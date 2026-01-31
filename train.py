import argparse
import inspect
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from collator import Collator
from models.decoder_only import DecoderOnlyConfig, DecoderOnlyForCausalLM
from pkm.memory import HashingMemory
from utils import ensure_dir, load_datasets, set_seed


def _parse_config_and_overrides(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    config_path: Optional[str] = None
    dotlist: List[str] = []

    i = 0
    while i < len(argv):
        a = argv[i]

        if a in {"-h", "--help"}:
            print(
                "Usage examples:\n"
                "  torchrun ... train.py config=path.yaml train.output_dir=./ckpt\n"
                "  torchrun ... train.py --config path.yaml model.decoder_only.d_model=512\n"
                "\n"
                "Supported forms:\n"
                "  config=path.yaml / --config path.yaml / --config=path.yaml\n"
                "  key=value (OmegaConf dotlist; supports nested keys like train.output_dir)\n"
            )
            raise SystemExit(0)

        if a.startswith("config="):
            config_path = a.split("=", 1)[1].strip()
            i += 1
            continue
        if a == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a value")
            config_path = argv[i + 1].strip()
            i += 2
            continue
        if a.startswith("--config="):
            config_path = a.split("=", 1)[1].strip()
            i += 1
            continue

        if "=" in a:
            k, v = a.split("=", 1)
            k = k.strip()
            if k == "config":
                config_path = v.strip()
            else:
                if k.startswith("+"):
                    k = k[1:]
                dotlist.append(f"{k}={v.strip()}")
            i += 1
            continue

        raise ValueError(
            f"Unrecognized argument: {a}. "
            "Legacy flags like '--output_dir xxx' are not supported. "
            "Use 'train.output_dir=xxx' style dotlist overrides."
        )

    return config_path, dotlist


def _apply_train_defaults(cfg):
    from omegaconf import OmegaConf  # type: ignore

    defaults = OmegaConf.create(
        {
            "strategy": "t5_seq2seq",
            "global": {"seed": 42, "special_token_for_answer": None},
            "train": {
                "output_dir": "./ckpt/debug",
                "optim": "adamw_torch",
                "epochs": 4,
                "learning_rate": 2e-5,
                "batch_size": None,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "logging_step": 10,
                "warmup_ratio": 0.01,
                "lr_scheduler_type": "cosine",
                "save_and_eval_strategy": "epoch",
                "save_and_eval_steps": 1000,
                "weight_decay": 0.01,
                "model_max_length": 2048,
                "resume_from_checkpoint": None,
            },
            "dataset": {
                "tasks": "seqrec",
                "max_his_len": 20,
                "add_prefix": False,
                "his_sep": ", ",
                "only_train_response": False,
                "train_prompt_sample_num": "1",
                "train_data_sample_num": "-1",
                "valid_prompt_id": 0,
                "sample_valid": True,
                "valid_prompt_sample_num": 2,
                "data_path": "",
                "name": "",
                "index_file": "",
                "train_file": None,
                "valid_file": None,
                "test_file": None,
            },
            "model": {
                "t5_seq2seq": {"base_model": "", "tokenizer_max_length": 512},
                "decoder_only": {"base_model": "", "tokenizer_max_length": 512},
            },
            "pkm": {
                "decoder_only": {
                    "pk_is_enabled": False,
                    "pk_layers": "",
                    "pk_mem_n_keys": 128,
                    "pk_mem_heads": 4,
                    "pk_mem_knn": None,
                    "pk_mem_share_values": True,
                    "pk_mem_k_dim": 512,
                    "pk_mem_v_dim": -1,
                    "pk_swilu_projection": True,
                    "pk_value_fixed_lr": 0.001,
                    "pk_value_weight_decay": 0.0,
                    "pk_mem_gated": False,
                    "pk_peer_variant": False,
                    "pk_topk": 8,
                    "pk_mem_dim": None,
                    "pk_use_gating": False,
                }
            },
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)
    return cfg


def _load_cfg_from_cli(argv: List[str]):
    from omegaconf import OmegaConf  # type: ignore

    config_path, dotlist = _parse_config_and_overrides(argv)

    base = OmegaConf.create({})
    if config_path:
        path = os.path.expandvars(os.path.expanduser(config_path))
        base = OmegaConf.load(path)

    cli = OmegaConf.from_dotlist(dotlist) if dotlist else OmegaConf.create({})
    merged = OmegaConf.merge(base, cli)

    merged = _apply_train_defaults(merged)
    return merged


def _parse_int_list(v: Any):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    s = str(v).strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _resolve_effective_per_device_batch_size(cfg) -> int:
    train_cfg = cfg.train

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_size = max(world_size, 1)

    global_bs = train_cfg.get("batch_size", None)
    if global_bs is None:
        per_device = int(train_cfg.per_device_batch_size)
        if per_device <= 0:
            raise ValueError(f"train.per_device_batch_size must be > 0, got {per_device}")
        return per_device

    global_bs = int(global_bs)
    if global_bs <= 0:
        raise ValueError(f"train.batch_size must be > 0, got {global_bs}")

    if global_bs % world_size != 0:
        raise ValueError(
            f"train.batch_size ({global_bs}) must be divisible by WORLD_SIZE ({world_size})."
        )

    per_device = global_bs // world_size
    if per_device <= 0:
        raise ValueError(
            f"Derived per-device batch size must be > 0, got {per_device} "
            f"(train.batch_size={global_bs}, WORLD_SIZE={world_size})."
        )
    return per_device


def _build_training_arguments(cfg, ddp: bool) -> transformers.TrainingArguments:
    train_cfg = cfg.train
    global_cfg = cfg["global"]

    effective_per_device_bs = _resolve_effective_per_device_batch_size(cfg)

    training_args_kwargs = dict(
        seed=int(global_cfg.seed),
        per_device_train_batch_size=int(effective_per_device_bs),
        per_device_eval_batch_size=int(effective_per_device_bs),
        gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
        warmup_ratio=float(train_cfg.warmup_ratio),
        num_train_epochs=float(train_cfg.epochs),
        learning_rate=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
        lr_scheduler_type=str(train_cfg.lr_scheduler_type),
        logging_steps=int(train_cfg.logging_step),
        optim=str(train_cfg.optim),
        save_strategy=str(train_cfg.save_and_eval_strategy),
        eval_steps=int(train_cfg.save_and_eval_steps),
        save_steps=int(train_cfg.save_and_eval_steps),
        output_dir=str(train_cfg.output_dir),
        save_total_limit=5,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=None,
        eval_delay=1 if str(train_cfg.save_and_eval_strategy) == "epoch" else 2000,
    )

    ta_params = inspect.signature(transformers.TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = str(train_cfg.save_and_eval_strategy)
    else:
        training_args_kwargs["eval_strategy"] = str(train_cfg.save_and_eval_strategy)

    return transformers.TrainingArguments(**training_args_kwargs)


def train_t5_seq2seq(cfg) -> None:
    set_seed(int(cfg["global"].seed))
    ensure_dir(str(cfg.train.output_dir))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        from omegaconf import OmegaConf  # type: ignore

        print(OmegaConf.to_yaml(cfg, resolve=True))

    base_model = str(cfg.model.t5_seq2seq.base_model)
    tokenizer_max_length = int(cfg.model.t5_seq2seq.get("tokenizer_max_length") or 512)

    config = T5Config.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        base_model,
        model_max_length=tokenizer_max_length,
    )

    train_data, valid_data = load_datasets(cfg)

    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(str(cfg.train.output_dir))
        config.save_pretrained(str(cfg.train.output_dir))
        print(train_data[100])
        print(valid_data[100])

    collator = Collator(cfg, tokenizer)

    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if local_rank == 0:
        print(model)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=_build_training_arguments(cfg, ddp=ddp),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=str(cfg.train.output_dir))


def _build_optimizer_with_pkm(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    base_lr = float(cfg.train.learning_rate)
    base_wd = float(cfg.train.weight_decay)
    pk_value_wd = float(cfg.pkm.decoder_only.get("pk_value_weight_decay") or 0.0)

    regular_decay = []
    regular_no_decay = []
    pk_decay = []
    pk_no_decay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_pk_value = bool(getattr(p, "pk_value_param", False))
        no_decay = (p.ndim == 1) or name.endswith(".bias")

        if is_pk_value:
            if no_decay:
                pk_no_decay.append(p)
            else:
                pk_decay.append(p)
        else:
            if no_decay:
                regular_no_decay.append(p)
            else:
                regular_decay.append(p)

    param_groups = []
    if regular_decay:
        param_groups.append({"params": regular_decay, "lr": base_lr, "weight_decay": base_wd})
    if regular_no_decay:
        param_groups.append({"params": regular_no_decay, "lr": base_lr, "weight_decay": 0.0})

    pk_lr = None
    for p in pk_decay + pk_no_decay:
        maybe = getattr(p, "fixed_lr", None)
        if maybe is not None:
            pk_lr = float(maybe)
            break
    if pk_lr is None:
        pk_lr = base_lr

    if pk_decay:
        param_groups.append({"params": pk_decay, "lr": pk_lr, "weight_decay": pk_value_wd})
    if pk_no_decay:
        param_groups.append({"params": pk_no_decay, "lr": pk_lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)

class PKMDiagnosticsCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        model = kwargs.get("model", None)
        if model is None:
            return

        per_layer = []
        for m in model.modules():
            if isinstance(m, HashingMemory) and hasattr(m, "_last_stats"):
                layer_id = getattr(m, "layer_id", None)
                stats = getattr(m, "_last_stats", None)
                if not isinstance(stats, dict):
                    continue
                per_layer.append(stats)
                if layer_id is not None:
                    for k, v in stats.items():
                        logs[f"pkm/layer{layer_id}/{k}"] = v

        if per_layer:
            keys = sorted(set().union(*[d.keys() for d in per_layer]))
            for k in keys:
                vs = [d[k] for d in per_layer if k in d]
                if vs:
                    logs[f"pkm/{k}"] = float(sum(vs) / len(vs))

def train_decoder_only(cfg) -> None:
    set_seed(int(cfg["global"].seed))
    ensure_dir(str(cfg.train.output_dir))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        from omegaconf import OmegaConf  # type: ignore

        print(OmegaConf.to_yaml(cfg, resolve=True))

    model_cfg = cfg.model.decoder_only
    pkm_cfg = cfg.pkm.decoder_only

    base_model = str(model_cfg.base_model)

    tokenizer = T5Tokenizer.from_pretrained(
        base_model,
        model_max_length=int(cfg.train.model_max_length),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    train_data, valid_data = load_datasets(cfg)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    if local_rank == 0:
        print(f"add {add_num} new token.")

    pk_layers = _parse_int_list(pkm_cfg.get("pk_layers") or "")
    pk_use_gating = bool(pkm_cfg.get("pk_use_gating") or pkm_cfg.get("pk_mem_gated") or False)

    pk_topk = pkm_cfg.get("pk_topk")
    pk_mem_dim = pkm_cfg.get("pk_mem_dim")

    config = DecoderOnlyConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or 1,
        max_seq_len=int(model_cfg.max_seq_len),
        d_model=int(model_cfg.d_model),
        n_layers=int(model_cfg.n_layers),
        head_dim=model_cfg.head_dim,
        n_heads=int(model_cfg.n_heads),
        n_kv_heads=model_cfg.n_kv_heads,
        multiple_of=int(model_cfg.multiple_of),
        ffn_dim_multiplier=model_cfg.ffn_dim_multiplier,
        ffn_dim=int(model_cfg.ffn_dim),
        dropout=float(model_cfg.dropout),
        rope_theta=float(model_cfg.rope_theta),
        init_base_std=model_cfg.init_base_std,
        init_std_factor=str(model_cfg.init_std_factor),
        pk_is_enabled=bool(pkm_cfg.pk_is_enabled),
        pk_layers=pk_layers,
        pk_mem_n_keys=int(pkm_cfg.pk_mem_n_keys),
        pk_mem_heads=int(pkm_cfg.pk_mem_heads),
        pk_mem_knn=pkm_cfg.pk_mem_knn,
        pk_mem_share_values=bool(pkm_cfg.pk_mem_share_values),
        pk_mem_k_dim=int(pkm_cfg.pk_mem_k_dim),
        pk_mem_v_dim=int(pkm_cfg.pk_mem_v_dim),
        pk_swilu_projection=bool(pkm_cfg.pk_swilu_projection),
        pk_value_fixed_lr=float(pkm_cfg.pk_value_fixed_lr),
        pk_value_weight_decay=float(pkm_cfg.pk_value_weight_decay),
        pk_mem_gated=bool(pkm_cfg.pk_mem_gated or pk_use_gating),
        pk_peer_variant=bool(pkm_cfg.pk_peer_variant),
        pk_topk=int(pk_topk) if pk_topk is not None else 8,
        pk_mem_dim=int(pk_mem_dim) if pk_mem_dim is not None else None,
        pk_use_gating=pk_use_gating,
    )

    if local_rank == 0:
        tokenizer.save_pretrained(str(cfg.train.output_dir))
        config.save_pretrained(str(cfg.train.output_dir))

    collator = Collator(cfg, tokenizer)

    model = DecoderOnlyForCausalLM(config)
    model.to(device)

    optimizer = _build_optimizer_with_pkm(model, cfg)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=_build_training_arguments(cfg, ddp=ddp),
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[PKMDiagnosticsCallback()],
    )

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=str(cfg.train.output_dir))


def main() -> None:
    cfg = _load_cfg_from_cli(sys.argv[1:])

    strategy = str(cfg.get("strategy") or "t5_seq2seq")
    if strategy == "t5_seq2seq":
        train_t5_seq2seq(cfg)
        return

    if strategy == "decoder_only":
        train_decoder_only(cfg)
        return

    raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    main()