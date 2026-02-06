import os
import sys
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from collator import TestCollator
from evaluate import get_topk_results
from generation_trie import Trie
from models.decoder_only import DecoderOnlyForCausalLM
from pkm.memory import HashingMemory
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
                "type": "t5_seq2seq",
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
            "pkm": {
                "t5_seq2seq": {
                    "pk_is_enabled": False,
                    "pk_encoder_layers": "",
                    "pk_decoder_layers": "",
                    "pk_mem_n_keys": 128,
                    "pk_mem_heads": 4,
                    "pk_mem_knn": 32,
                    "pk_mem_share_values": False,
                    "pk_mem_k_dim": 512,
                    "pk_mem_v_dim": -1,
                    "pk_swilu_projection": True,
                    "pk_value_fixed_lr": 0.001,
                    "pk_mem_gated": False,
                    "pk_peer_variant": False,
                    "pk_topk": 8,
                    "pk_mem_dim": None,
                }
            },
        }
    )
    return OmegaConf.merge(defaults, cfg)


def _parse_int_list(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    s = str(v).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def _inject_pkm_into_t5_seq2seq(model: T5ForConditionalGeneration, cfg) -> None:
    pkm_cfg = cfg.pkm.get("t5_seq2seq", {})
    if not bool(pkm_cfg.get("pk_is_enabled") or False):
        return

    encoder_layers = set(_parse_int_list(pkm_cfg.get("pk_encoder_layers") or ""))
    decoder_layers = set(_parse_int_list(pkm_cfg.get("pk_decoder_layers") or ""))

    d_model = int(model.config.d_model)

    def _replace_block_ffn(block, layer_id: int, which: str) -> None:
        if not hasattr(block, "layer") or len(block.layer) < 1:
            raise ValueError(f"Unexpected T5 block structure for {which} layer {layer_id}")

        ff_layer = block.layer[-1]
        if not hasattr(ff_layer, "DenseReluDense"):
            raise ValueError(
                f"Cannot find DenseReluDense in {which} layer {layer_id}. "
                f"Got ff_layer={type(ff_layer)} with attrs={sorted(dir(ff_layer))}"
            )

        ff_layer.DenseReluDense = HashingMemory(
            input_dim=d_model,
            output_dim=d_model,
            mem_n_keys=int(pkm_cfg.get("pk_mem_n_keys") or 128),
            mem_heads=int(pkm_cfg.get("pk_mem_heads") or 4),
            mem_knn=int(pkm_cfg.get("pk_mem_knn") or 32),
            mem_share_values=bool(pkm_cfg.get("pk_mem_share_values") or False),
            mem_k_dim=int(pkm_cfg.get("pk_mem_k_dim") or 512),
            mem_v_dim=int(pkm_cfg.get("pk_mem_v_dim") or -1),
            swilu_projection=bool(pkm_cfg.get("pk_swilu_projection") or True),
            value_fixed_lr=float(pkm_cfg.get("pk_value_fixed_lr") or 0.001),
            mem_gated=bool(pkm_cfg.get("pk_mem_gated") or False),
            peer_variant=bool(pkm_cfg.get("pk_peer_variant") or False),
            topk=pkm_cfg.get("pk_topk"),
            mem_dim=pkm_cfg.get("pk_mem_dim"),
            mem_input_dropout=0.0,
            mem_query_dropout=0.0,
            mem_value_dropout=0.0,
        )
        ff_layer.DenseReluDense.layer_id = int(layer_id)

    if encoder_layers:
        for i, block in enumerate(model.encoder.block):
            if i in encoder_layers:
                _replace_block_ffn(block, i, which="encoder")

    if decoder_layers:
        for i, block in enumerate(model.decoder.block):
            if i in decoder_layers:
                _replace_block_ffn(block, i, which="decoder")


def _load_t5_model_for_test(cfg, tokenizer: T5Tokenizer, device: torch.device) -> T5ForConditionalGeneration:
    ckpt_path = str(cfg.model.ckpt_path).strip()
    if not ckpt_path:
        raise ValueError("Missing model.ckpt_path for t5_seq2seq test")

    config = T5Config.from_pretrained(ckpt_path)
    model = T5ForConditionalGeneration(config)

    if len(tokenizer) != int(model.config.vocab_size):
        model.resize_token_embeddings(len(tokenizer))

    _inject_pkm_into_t5_seq2seq(model, cfg)

    weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Cannot find '{weights_path}'. "
            "Expected a HuggingFace Trainer-style checkpoint dir containing pytorch_model.bin."
        )

    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()
    return model


def _load_tokenizer_for_test(cfg) -> T5Tokenizer:
    tokenizer_path = str(cfg.model.tokenizer_path or "").strip()
    ckpt_path = str(cfg.model.ckpt_path or "").strip()
    base_model = str(cfg.model.base_model or "").strip()

    source = tokenizer_path or ckpt_path or base_model
    if not source:
        raise ValueError(
            "Missing tokenizer source. Provide one of: "
            "model.tokenizer_path, model.ckpt_path, model.base_model"
        )

    # Keep truncation/length consistent with training when available.
    tokenizer_kwargs = {}
    if hasattr(cfg, "train") and hasattr(cfg.train, "model_max_length") and cfg.train.model_max_length is not None:
        tokenizer_kwargs["model_max_length"] = int(cfg.train.model_max_length)

    tokenizer = T5Tokenizer.from_pretrained(source, **tokenizer_kwargs)
    return tokenizer


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

    if model_type in ("t5_seq2seq", "t5"):
        model = _load_t5_model_for_test(cfg, tokenizer, device)
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

            if model_type in ("t5_seq2seq", "t5"):
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