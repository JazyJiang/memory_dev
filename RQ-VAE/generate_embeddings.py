"""
Generate item embeddings using T5 encoder for RQ-VAE training.

Reads combine_tdcb_maps.npy (from 0_process.py), encodes each item's text
with T5-small encoder + mean pooling, and saves as .emb-t5-tdcb.npy.

Usage:
    python generate_embeddings.py \
        --tdcb_path ../data/info/Toys_and_Games_5_2016-10-2018-11_combine_tdcb_maps.npy \
        --output_path ../data/info/Toys_and_Games.emb-t5-tdcb.npy \
        --model_name google-t5/t5-small \
        --batch_size 128 \
        --device cuda:0
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdcb_path", type=str, required=True,
                        help="Path to combine_tdcb_maps.npy from 0_process.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for .emb-t5-tdcb.npy")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-small")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    comb = np.load(args.tdcb_path, allow_pickle=True).item()
    recid2combine = comb["recid2combine"]
    num_items = len(recid2combine)
    texts = [recid2combine[i] for i in range(num_items)]
    print(f"Loaded {num_items} items from {args.tdcb_path}")

    print(f"Loading {args.model_name} ...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5EncoderModel.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()

    embed_dim = model.config.d_model
    all_embeddings = np.zeros((num_items, embed_dim), dtype=np.float32)
    print(f"Embedding dim: {embed_dim}, output shape: ({num_items}, {embed_dim})")

    with torch.no_grad():
        for start in tqdm(range(0, num_items, args.batch_size), desc="Encoding"):
            end = min(start + args.batch_size, num_items)
            batch_texts = texts[start:end]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
            embeddings = mean_pooling(outputs.last_hidden_state, encoded.attention_mask)
            all_embeddings[start:end] = embeddings.cpu().numpy()

    np.save(args.output_path, all_embeddings)
    print(f"Saved embeddings to {args.output_path}, shape: {all_embeddings.shape}")


if __name__ == "__main__":
    main()
