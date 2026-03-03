#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

        return EventAccumulator
    except Exception as e:
        raise SystemExit(
            "Cannot import tensorboard EventAccumulator. "
            "Install TensorBoard first, e.g. `pip install tensorboard`. "
            f"Original error: {e}"
        )


def _list_image_tags(ea) -> List[str]:
    try:
        tags = ea.Tags().get("images", [])
        return [str(t) for t in tags]
    except Exception:
        return []


def export_images(
    logdir: Path,
    out_dir: Path,
    tags: Optional[List[str]] = None,
    tag_prefix: str = "",
    last_k: int = 0,
    stride: int = 1,
) -> List[str]:
    EventAccumulator = _import_event_accumulator()

    if not logdir.exists():
        raise SystemExit(f"logdir does not exist: {logdir}")

    if stride <= 0:
        stride = 1

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    all_tags = _list_image_tags(ea)
    if tags is None or len(tags) == 0:
        selected = all_tags
    else:
        selected = [t for t in tags if t in all_tags]

    if tag_prefix:
        selected = [t for t in selected if t.startswith(tag_prefix)]

    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for tag in selected:
        try:
            imgs = ea.Images(tag)
        except Exception:
            continue

        if not imgs:
            continue

        if last_k > 0:
            imgs_sel = imgs[-last_k:]
        else:
            imgs_sel = imgs

        imgs_sel = imgs_sel[::stride]

        safe_tag = tag.replace("/", "__")
        for j, img in enumerate(imgs_sel):
            step = int(getattr(img, "step", 0) or 0)
            out_path = out_dir / f"{safe_tag}.step{step}.i{j}.png"
            try:
                out_path.write_bytes(img.encoded_image_string)
            except Exception:
                continue
            saved.append(str(out_path))

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Export TensorBoard image summaries from events to PNG.")
    ap.add_argument("--logdir", required=True, help="TensorBoard logdir containing events.out.tfevents.*")
    ap.add_argument("--out_dir", required=True, help="Output directory for pngs")
    ap.add_argument(
        "--tags",
        default=None,
        help="Comma-separated image tags to export. If omitted, exports all image tags.",
    )
    ap.add_argument(
        "--tag_prefix",
        default="",
        help="Only export tags that start with this prefix (e.g. 'PKM/').",
    )
    ap.add_argument(
        "--last_k",
        type=int,
        default=0,
        help="If >0, only export last K images per tag. 0 means export all.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Export every Nth image per tag.",
    )

    args = ap.parse_args()

    logdir = Path(args.logdir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    tags = None
    if args.tags is not None and str(args.tags).strip() != "":
        tags = [t.strip() for t in str(args.tags).split(",") if t.strip() != ""]

    saved = export_images(
        logdir=logdir,
        out_dir=out_dir,
        tags=tags,
        tag_prefix=str(args.tag_prefix),
        last_k=int(args.last_k),
        stride=int(args.stride),
    )

    print(f"logdir: {logdir}")
    print(f"out_dir: {out_dir}")
    print(f"saved_png: {len(saved)}")
    if saved:
        print("last_png:", saved[-1])


if __name__ == "__main__":
    main()