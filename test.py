import os
import sys
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from collator import TestCollator
from evaluate import get_topk_results
from generation_trie import Trie
from models.decoder_only import DecoderOnlyForCausalLM
from utils import (
    computeTopNAccuracy,
    load_test_dataset,
    prefix_allowed_tokens_fn,
    print_results,
    set_seed,
)


def _parse_config_and_overrides(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    config_path: Optional[str] = None
    dotlist: List[str] = []

    for a in argv:
        if a in {"-h", "--help"}:
            print(
                "Usage examples:\n"
                "  python test.py config=path.yaml model.type=decoder_only model.ckpt_path=... \n"
                "  python test.py config=path.yaml dataset.data_path=... dataset.name=... test.filter_items=true\n"
                "\n"
                "Supported forms:\n"
                "  config=path.yaml\n"
                "  key=value (OmegaConf dotlist; supports nested keys like model.ckpt_path)\n"
                "\n"
                "NOT supported:\n"
                "  --ckpt_path ... / --gpu_id ... (legacy flags removed)\n"
            )
            raise SystemExit(0)

        if a.startswith("config="):
            config_path = a.split("=", 1)[1].strip()
            continue

        if a.startswith("--"):
            raise ValueError(
                f"Legacy flag '{a}' is not supported. "
                "Use dotlist 'key=value' overrides, e.g. model.ckpt_path=... global.gpu_id=0"
            )

        if "=" not in a:
            raise ValueError(f"Unrecognized argument: {a}. Expected key=value or config=...")

        k, v = a.split("=", 1)
        k = k.strip()
        if k == "config":
            config_path = v.strip()
        else:
            if k.startswith("+"):
                k = k[1:]
            dotlist.append(f"{k}={v.strip()}")

    return config_path, dotlist


def _apply_test_defaults(cfg):
    from omegaconf import OmegaConf  # type: ignore

    defaults = OmegaConf.create(
        {
            "global": {"seed": 42, "gpu_id": 0},
            "model": {
                "type": "t5",
                "ckpt_path": "",
                "base_model": "",
                "tokenizer_path": "",
            },
            "dataset": {
                "data_path": "",
                "name": "",
                "index_file": "_index.json",
                "train_file": None,
                "valid_file": None,
                "test_file": None,
            },
            "test": {
                "task": "seqrec",
                "batch_size": 2,
                "num_beams": 20,
                "max_new_tokens": 10,
                "sample_num": -1,
                "filter_items": False,
            },
        }
    )
    return OmegaConf.merge(defaults, cfg)


def _load_cfg_from_cli(argv: List[str]):
    from omegaconf import OmegaConf  # type: ignore

    config_path, dotlist = _parse_config_and_overrides(argv)

    base = OmegaConf.create({})
    if config_path:
        path = os.path.expandvars(os.path.expanduser(config_path))
        base = OmegaConf.load(path)

    cli = OmegaConf.from_dotlist(dotlist) if dotlist else OmegaConf.create({})
    merged = OmegaConf.merge(base, cli)
    merged = _apply_test_defaults(merged)
    return merged


def _load_tokenizer_for_test(cfg) -> T5Tokenizer:
    tokenizer_path = str(cfg.model.get("tokenizer_path") or "").strip()
    if tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    else:
        ckpt_path = str(cfg.model.get("ckpt_path") or "").strip()
        base_model = str(cfg.model.get("base_model") or "").strip()

        if ckpt_path:
            try:
                tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
            except Exception:
                if not base_model:
                    raise
                tokenizer = T5Tokenizer.from_pretrained(base_model)
        else:
            if not base_model:
                raise ValueError("Missing model.ckpt_path and model.base_model; cannot load tokenizer")
            tokenizer = T5Tokenizer.from_pretrained(base_model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    return tokenizer


def test(cfg):
    from omegaconf import OmegaConf  # type: ignore

    set_seed(int(cfg["global"].seed))
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda", int(cfg["global"].gpu_id))
    tokenizer = _load_tokenizer_for_test(cfg)

    model_type = str(cfg.model.type)
    ckpt_path = str(cfg.model.ckpt_path)

    test_data = load_test_dataset(cfg)
    all_items = test_data.get_all_items()

    if model_type == "t5":
        device_map = {"": int(cfg["global"].gpu_id)}
        model = T5ForConditionalGeneration.from_pretrained(
            ckpt_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        candidate_trie = Trie(
            [[tokenizer.pad_token_id] + tokenizer.encode(candidate) for candidate in all_items]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    elif model_type == "decoder_only":
        tokenizer.padding_side = "left"
        model = DecoderOnlyForCausalLM.from_pretrained(
            ckpt_path,
            low_cpu_mem_usage=True,
        ).to(device)
        candidate_trie = Trie([tokenizer.encode(candidate) for candidate in all_items])
        prefix_allowed_tokens = None

    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    collator = TestCollator(cfg, tokenizer)

    test_loader = DataLoader(
        test_data,
        batch_size=int(cfg.test.batch_size),
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("data num:", len(test_data))
    model.eval()

    with torch.no_grad():
        all_pred_list = []
        all_gold_list = []

        for _, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]

            if model_type == "t5":
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=int(cfg.test.max_new_tokens),
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=int(cfg.test.num_beams),
                    num_return_sequences=int(cfg.test.num_beams),
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            else:
                context_len = int(inputs["input_ids"].shape[1])

                def _prefix_allowed_tokens(batch_id, sentence):
                    sentence = sentence.tolist()
                    return candidate_trie.get(sentence[context_len:])

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=int(cfg.test.max_new_tokens),
                    prefix_allowed_tokens_fn=_prefix_allowed_tokens,
                    num_beams=int(cfg.test.num_beams),
                    num_return_sequences=int(cfg.test.num_beams),
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                suffix_ids = output_ids[:, context_len:]
                decoded = tokenizer.batch_decode(suffix_ids, skip_special_tokens=True)

            topk_res = get_topk_results(
                decoded,
                scores,
                targets,
                int(cfg.test.num_beams),
                all_items=all_items if bool(cfg.test.filter_items) else None,
            )
            all_pred_list.extend(topk_res)
            all_gold_list.extend(targets)

        test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20])
        print("=== End ===")
        print_results(None, None, test_results)


if __name__ == "__main__":
    cfg = _load_cfg_from_cli(sys.argv[1:])
    test(cfg)