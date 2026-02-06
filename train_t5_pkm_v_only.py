import glob
import os
import sys
from typing import List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import transformers
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from collator import Collator
from pkm.memory import HashingMemory
from utils import ensure_dir, load_datasets, set_seed

# Reuse config parsing / defaults / TrainingArguments builder from existing train.py
from train import _build_training_arguments, _inject_pkm_into_t5_seq2seq, _load_cfg_from_cli


def _find_pytorch_model_bin(init_path: str) -> str:
    init_path = os.path.expandvars(os.path.expanduser(str(init_path)))

    if os.path.isfile(init_path):
        return init_path

    if not os.path.isdir(init_path):
        raise FileNotFoundError(f"train.init_from_ckpt not found: {init_path}")

    direct = os.path.join(init_path, "pytorch_model.bin")
    if os.path.isfile(direct):
        return direct

    ckpt_dirs = glob.glob(os.path.join(init_path, "checkpoint-*"))
    if ckpt_dirs:

        def _step(d: str) -> int:
            base = os.path.basename(d)
            try:
                return int(base.split("-", 1)[1])
            except Exception:
                return -1

        latest = max(ckpt_dirs, key=_step)
        cand = os.path.join(latest, "pytorch_model.bin")
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"Cannot find pytorch_model.bin under {init_path}. "
        "Expected either {init_path}/pytorch_model.bin or {init_path}/checkpoint-*/pytorch_model.bin"
    )


def _freeze_all_except_pkm_values(model: torch.nn.Module) -> Tuple[int, int, List[str]]:
    for p in model.parameters():
        p.requires_grad = False

    total = 0
    trainable = 0
    trainable_names: List[str] = []

    for name, p in model.named_parameters():
        total += p.numel()
        if bool(getattr(p, "pk_value_param", False)):
            p.requires_grad = True
            trainable += p.numel()
            trainable_names.append(name)

    return total, trainable, trainable_names


def _assert_only_pkm_values_trainable(model: torch.nn.Module, local_rank: int) -> None:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    assert trainable, "No trainable parameters found (expected PKM V / values.*)."

    bad = [(n, p) for n, p in trainable if not bool(getattr(p, "pk_value_param", False))]
    assert not bad, (
        "Found trainable params that are NOT PKM values (pk_value_param=False). "
        f"Examples: {[n for n, _ in bad[:20]]}"
    )

    # Extra safety: name pattern check (handle both normal and PEER variants)
    bad_name = []
    for n, _ in trainable:
        if (".values." in n) or (".values_u." in n) or (".values_v." in n):
            continue
        bad_name.append(n)

    assert not bad_name, (
        "Trainable params do not look like PKM V weights (expected names containing "
        "'.values.' / '.values_u.' / '.values_v.'). "
        f"Examples: {bad_name[:20]}"
    )

    if local_rank == 0:
        print(f"[ASSERT] Trainable params count: {len(trainable)} (all pk_value_param=True)")
        for n, p in trainable[:30]:
            print(f"[ASSERT] trainable: {n}  shape={tuple(p.shape)}")


def _build_optimizer_value_only(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    pkm_cfg = cfg.pkm.get("t5_seq2seq", {})

    base_lr = float(cfg.train.learning_rate)
    pk_value_wd = float(pkm_cfg.get("pk_value_weight_decay") or 0.0)

    pk_params = []
    pk_lr: Optional[float] = None

    for _, p in model.named_parameters():
        if not bool(getattr(p, "pk_value_param", False)):
            continue
        if not p.requires_grad:
            continue
        pk_params.append(p)

        maybe = getattr(p, "fixed_lr", None)
        if maybe is not None:
            pk_lr = float(maybe)

    if not pk_params:
        raise RuntimeError(
            "No PKM value parameters found to train. "
            "Check pkm.t5_seq2seq.pk_is_enabled=true and you injected PKM into T5."
        )

    if pk_lr is None:
        pk_lr = base_lr

    return torch.optim.AdamW(
        [{"params": pk_params, "lr": pk_lr, "weight_decay": pk_value_wd}]
    )


def train_t5_seq2seq_value_only(cfg) -> None:
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
    tokenizer_max_length = int(cfg.train.model_max_length)

    init_from_ckpt = cfg.train.get("init_from_ckpt", None)
    init_from_ckpt = str(init_from_ckpt) if init_from_ckpt else None

    if init_from_ckpt:
        init_dir = os.path.expandvars(os.path.expanduser(init_from_ckpt))
        config = T5Config.from_pretrained(init_dir)
        tokenizer = T5Tokenizer.from_pretrained(init_dir, model_max_length=tokenizer_max_length)
    else:
        config = T5Config.from_pretrained(base_model)
        tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=tokenizer_max_length)

    train_data, valid_data = load_datasets(cfg)

    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"add {add_num} new token.")
        print("data num:", len(train_data))
        tokenizer.save_pretrained(str(cfg.train.output_dir))
        config.save_pretrained(str(cfg.train.output_dir))
        print(train_data[100])
        print(valid_data[100])

    collator = Collator(cfg, tokenizer)

    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))

    _inject_pkm_into_t5_seq2seq(model, cfg)

    if init_from_ckpt:
        ckpt_bin = _find_pytorch_model_bin(init_from_ckpt)
        if local_rank == 0:
            print(f"Loading model weights from: {ckpt_bin}")
        state = torch.load(ckpt_bin, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if local_rank == 0:
            print(f"load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing (first 30):", missing[:30])
            if unexpected:
                print("  unexpected (first 30):", unexpected[:30])

    model.to(device)

    total_params, trainable_params, trainable_names = _freeze_all_except_pkm_values(model)

    if local_rank == 0:
        print(f"Total params: {total_params:,}")
        print(f"Trainable params (PKM V only): {trainable_params:,}")
        print(f"Trainable PKM V tensors: {len(trainable_names)}")

    _assert_only_pkm_values_trainable(model, local_rank=local_rank)

    optimizer = _build_optimizer_value_only(model, cfg)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=_build_training_arguments(cfg, ddp=ddp),
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, None),
    )
    model.config.use_cache = False

    # IMPORTANT: do not resume trainer state/optimizer state (optimizer param set changed)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_state()
    trainer.save_model(output_dir=str(cfg.train.output_dir))


def main() -> None:
    cfg = _load_cfg_from_cli(sys.argv[1:])

    strategy = str(cfg.get("strategy") or "t5_seq2seq")
    if strategy != "t5_seq2seq":
        raise ValueError("train_t5_pkm_v_only.py only supports strategy=t5_seq2seq")

    train_t5_seq2seq_value_only(cfg)


if __name__ == "__main__":
    main()