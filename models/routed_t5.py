"""
Cross-attention routing for T5.

Different decoder layers attend to different portions of the encoder output:
- "early layers" cross-attend only to the early-history tokens
- all other layers cross-attend only to the recent-history tokens

The split position (in tokens) is conveyed per-sample via the
``history_split_pos`` tensor that the collator adds to the batch.

Usage
-----
    from models.routed_t5 import enable_cross_attention_routing

    model = T5ForConditionalGeneration(config)
    enable_cross_attention_routing(model, early_layers={3})
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-layer mask construction
# ---------------------------------------------------------------------------

def build_per_layer_masks(
    encoder_attention_mask: torch.Tensor,   # [B, S] 1/0
    history_split_pos: torch.Tensor,        # [B]  — #early tokens per sample
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (early_mask, recent_mask) in *inverted* form ready for attention.

    Both tensors have shape ``[B, 1, 1, S]``.

    * ``early_mask``:  0 where the token is early & valid, large-neg elsewhere
    * ``recent_mask``: 0 where the token is recent & valid, large-neg elsewhere
    """
    B, S = encoder_attention_mask.shape
    device = encoder_attention_mask.device

    # Position indices [1, S]
    pos = torch.arange(S, device=device).unsqueeze(0)          # [1, S]
    split = history_split_pos.unsqueeze(1)                      # [B, 1]

    # Boolean masks — True means "allowed to attend"
    is_early  = pos < split                                     # [B, S]
    is_recent = pos >= split                                    # [B, S]

    base = encoder_attention_mask.bool()                        # [B, S]
    early_bool  = is_early  & base                              # [B, S]
    recent_bool = is_recent & base                              # [B, S]

    # Invert: 0.0 for attend, large-neg for masked
    neg_inf = torch.finfo(dtype).min
    early_inv  = torch.where(early_bool,  torch.tensor(0.0, device=device, dtype=dtype),
                             torch.tensor(neg_inf, device=device, dtype=dtype))
    recent_inv = torch.where(recent_bool, torch.tensor(0.0, device=device, dtype=dtype),
                             torch.tensor(neg_inf, device=device, dtype=dtype))

    return early_inv[:, None, None, :], recent_inv[:, None, None, :]


# ---------------------------------------------------------------------------
# Routing context — stores per-layer masks for the current forward pass
# ---------------------------------------------------------------------------

class _RoutingContext:
    """Thread-local-ish storage attached to the model instance."""

    def __init__(self) -> None:
        self.early_mask: Optional[torch.Tensor] = None
        self.recent_mask: Optional[torch.Tensor] = None
        self.active: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_cross_attention_routing(
    model: nn.Module,
    early_layers: Set[int],
) -> None:
    """Patch *model* (a ``T5ForConditionalGeneration``) so that different
    decoder layers use different encoder-attention masks when
    ``history_split_pos`` is present in the forward kwargs.

    The patching is non-destructive:
    * If ``history_split_pos`` is absent, behaviour is identical to vanilla T5.
    * No subclassing; only the model's ``forward`` and decoder-block hooks are
      touched.
    """

    ctx = _RoutingContext()
    # Attach to model so it won't be garbage-collected
    model._routing_ctx = ctx  # type: ignore[attr-defined]
    model._routing_early_layers = early_layers  # type: ignore[attr-defined]

    # -- 1. Install pre-hooks on each decoder block -------------------------
    # T5Block.forward signature (positions excluding self):
    #   0: hidden_states, 1: attention_mask, 2: position_bias,
    #   3: encoder_hidden_states, 4: encoder_attention_mask, ...
    # Some transformers versions pass encoder_attention_mask positionally,
    # others as a keyword. The hook handles both cases.
    _ENC_ATTN_MASK_POS = 4

    hooks = []
    for layer_idx, block in enumerate(model.decoder.block):
        is_early = layer_idx in early_layers

        def _make_hook(early: bool):
            def _hook(module, args, kwargs):
                if not ctx.active:
                    return None  # no-op when routing is off
                mask = ctx.early_mask if early else ctx.recent_mask
                if mask is None:
                    return None
                # During beam search, batch is expanded by num_beams.
                # Expand mask to match actual batch size.
                hidden_states = args[0]
                if mask.shape[0] != hidden_states.shape[0]:
                    expand_factor = hidden_states.shape[0] // mask.shape[0]
                    mask = mask.repeat_interleave(expand_factor, dim=0)
                if "encoder_attention_mask" in kwargs:
                    kwargs["encoder_attention_mask"] = mask
                elif len(args) > _ENC_ATTN_MASK_POS:
                    # Passed as positional arg — replace in-place
                    args = list(args)
                    args[_ENC_ATTN_MASK_POS] = mask
                    args = tuple(args)
                return args, kwargs
            return _hook

        h = block.register_forward_pre_hook(_make_hook(is_early), with_kwargs=True)
        hooks.append(h)

    model._routing_hooks = hooks  # type: ignore[attr-defined]

    # -- 2. Wrap model.forward to set up / tear down context ----------------
    original_forward = model.forward

    def _routed_forward(*args, **kwargs):
        split_pos = kwargs.pop("history_split_pos", None)

        if split_pos is not None and split_pos.sum() > 0:
            # Determine encoder_attention_mask from kwargs or the first
            # positional arg.  T5ForConditionalGeneration.forward signature:
            #   forward(input_ids, attention_mask, ...)
            enc_attn_mask = kwargs.get("attention_mask", None)
            if enc_attn_mask is None and len(args) >= 2:
                enc_attn_mask = args[1]
            if enc_attn_mask is None:
                # Fallback: all-ones
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                if input_ids is not None:
                    enc_attn_mask = torch.ones(
                        input_ids.shape[:2], device=input_ids.device, dtype=torch.long
                    )

            if enc_attn_mask is not None:
                ctx.early_mask, ctx.recent_mask = build_per_layer_masks(
                    enc_attn_mask, split_pos,
                    dtype=next(model.parameters()).dtype,
                )
                ctx.active = True

        try:
            return original_forward(*args, **kwargs)
        finally:
            ctx.active = False
            ctx.early_mask = None
            ctx.recent_mask = None

    model.forward = _routed_forward


@contextmanager
def routing_context(model: nn.Module, encoder_attention_mask: torch.Tensor,
                    history_split_pos: torch.Tensor):
    """Context manager to activate routing masks for ``model.generate()``.

    ``generate()`` doesn't forward unknown kwargs to ``forward()``, so we
    pre-activate the routing context before calling ``generate()``::

        with routing_context(model, attn_mask, split_pos):
            output = model.generate(input_ids=..., attention_mask=..., ...)
    """
    ctx: _RoutingContext = getattr(model, "_routing_ctx", None)
    if ctx is None or history_split_pos is None or history_split_pos.sum() == 0:
        yield
        return

    ctx.early_mask, ctx.recent_mask = build_per_layer_masks(
        encoder_attention_mask, history_split_pos,
        dtype=next(model.parameters()).dtype,
    )
    ctx.active = True
    try:
        yield
    finally:
        ctx.active = False
        ctx.early_mask = None
        ctx.recent_mask = None


# ---------------------------------------------------------------------------
# Auxiliary prediction head
# ---------------------------------------------------------------------------

class AuxPredictionHead(nn.Module):
    """Lightweight head that predicts target tokens from an intermediate
    decoder layer's hidden state, providing a direct learning signal for
    the early-history routing layer."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss from intermediate hidden states.

        Args:
            hidden_states: [B, T, D] — decoder hidden states after the early layer
            labels: [B, T] — target token ids (with -100 for padding)

        Returns:
            Scalar loss.
        """
        logits = self.proj(hidden_states)  # [B, T, V]
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )


def enable_aux_prediction_head(
    model: nn.Module,
    early_layers: Set[int],
) -> AuxPredictionHead:
    """Attach an auxiliary prediction head after the first early layer.

    Returns the ``AuxPredictionHead`` module (already registered as a
    sub-module on ``model`` so its parameters are included in
    ``model.parameters()``).

    The captured hidden state is stored on ``model._aux_hidden`` after each
    forward pass. The training loop should compute the aux loss via::

        if hasattr(model, '_aux_head') and model._aux_hidden is not None:
            aux_loss = model._aux_head(model._aux_hidden, labels)
    """
    d_model = model.config.d_model
    vocab_size = model.config.vocab_size
    head = AuxPredictionHead(d_model, vocab_size)

    # Register as a proper sub-module so optimizer picks up its params
    model.add_module("_aux_head", head)
    model._aux_hidden = None

    # Hook on the first early layer to capture hidden states
    target_layer = min(early_layers)
    block = model.decoder.block[target_layer]

    def _capture_hook(module, args, output):
        # T5Block.forward returns a tuple; first element is hidden_states
        hidden = output[0] if isinstance(output, tuple) else output
        model._aux_hidden = hidden

    block.register_forward_hook(_capture_hook)
    return head
