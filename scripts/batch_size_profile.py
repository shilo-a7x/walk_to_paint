import argparse
import os
import sys

from src.utils.config import load_config


def _parse_batch_sizes(text: str):
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("No batch sizes provided.")
    return items


def main():
    parser = argparse.ArgumentParser(description="Batch size memory profiling")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1024,1280,1536,1792,2048,2304,2560,3072,3584,4096,4608,5120,6144,7168,8192",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="epinions",
        help="Dataset name to load config overrides",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to dataset_cache.pt (defaults to dataset.data_dir)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Forward/backward steps per batch size",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocab size (default: load from cache)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override num classes (default: load from cache)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    import time
    import torch
    import torch.nn.functional as F
    from src.model.lit_model import LitEdgeClassifier

    cfg = load_config(args.config, overrides=[f"dataset.name={args.dataset_name}"])
    cache_path = args.cache_path or os.path.join(cfg.dataset.data_dir, "dataset_cache.pt")

    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    # Use command-line overrides if provided, otherwise load from cache
    if args.vocab_size and args.num_classes:
        print("Using command-line parameter overrides...")
        cfg.model.vocab_size = args.vocab_size
        cfg.model.num_classes = args.num_classes
        # Use config defaults for these
        if not hasattr(cfg.model, 'pad_id') or cfg.model.pad_id is None:
            cfg.model.pad_id = 0
        if not hasattr(cfg.model, 'ignore_index') or cfg.model.ignore_index is None:
            cfg.model.ignore_index = -100
    else:
        # Try to load metadata from cache
        print("Loading cache metadata (this may take a minute for large files)...")
        try:
            cache_data = torch.load(cache_path, map_location="cpu")
            metadata = cache_data.get("metadata", {})
            cfg.model.vocab_size = args.vocab_size or metadata.get("vocab_size")
            cfg.model.num_classes = args.num_classes or metadata.get("num_classes")
            cfg.model.pad_id = metadata.get("pad_id", 0)
            cfg.model.ignore_index = metadata.get("ignore_index", -100)
            if "class_weights" in metadata:
                cfg.model.class_weights = metadata.get("class_weights")
            del cache_data
            print("Metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading cache: {e}")
            sys.exit(1)

    if any(v is None for v in [cfg.model.vocab_size, cfg.model.num_classes, cfg.model.pad_id, cfg.model.ignore_index]):
        raise ValueError("Missing vocab_size/num_classes/pad_id/ignore_index from cache metadata.")

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision(getattr(cfg, "float32_precision", "medium"))

    model = LitEdgeClassifier(cfg).to(device)
    model.train()

    seq_len = int(2 * cfg.dataset.max_walk_length + 1)
    num_classes = int(cfg.model.num_classes)
    ignore_index = int(cfg.model.ignore_index)
    vocab_size = int(cfg.model.vocab_size)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    props = torch.cuda.get_device_properties(device)
    print(f"Total VRAM: {props.total_memory / (1024 ** 3):.2f} GB")
    print(f"seq_len={seq_len}, vocab_size={vocab_size}, num_classes={num_classes}")

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    results = []
    best = None

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            
            for _ in range(args.steps):
                input_ids = torch.randint(0, vocab_size, (bs, seq_len), device=device, dtype=torch.long)
                attention_mask = torch.ones((bs, seq_len), device=device, dtype=torch.bool)
                labels = torch.randint(0, num_classes, (bs, seq_len), device=device, dtype=torch.long)
                if bs >= 16:
                    labels[: bs // 16, : seq_len // 8] = ignore_index

                logits = model.model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=ignore_index,
                    weight=model.class_weights.to(device),
                )
                loss.backward()

                model.zero_grad(set_to_none=True)
                del input_ids, attention_mask, labels, logits, loss

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            throughput = (bs * args.steps) / elapsed  # samples/second
            
            results.append((bs, "ok", peak_gb, throughput))
            best = (bs, peak_gb, throughput)
            print(f"bs={bs:5d} OK   peak_alloc={peak_gb:6.2f} GB   throughput={throughput:8.1f} samples/s   time={elapsed:.3f}s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                results.append((bs, "oom", peak_gb, 0.0))
                print(f"bs={bs:5d} OOM  peak_alloc={peak_gb:6.2f} GB")
                break
            raise

    print("\nSUMMARY")
    if best is None:
        print("No batch size succeeded.")
    else:
        print(f"Max successful batch size: {best[0]} (peak_alloc={best[1]:.2f} GB, throughput={best[2]:.1f} samples/s)")
        
        # Find batch size with best throughput
        ok_results = [(bs, mem, thr) for bs, status, mem, thr in results if status == "ok"]
        if ok_results:
            best_throughput = max(ok_results, key=lambda x: x[2])
            print(f"Best throughput: batch_size={best_throughput[0]} ({best_throughput[2]:.1f} samples/s, {best_throughput[1]:.2f} GB)")

    print("All results:")
    for bs, status, gb in results:
        print(f"  {bs:5d}  {status:>3}  {gb:6.2f} GB")


if __name__ == "__main__":
    main()
