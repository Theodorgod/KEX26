import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from loader import EventDataset
from model import EventCNN
import config
import argparse
import time
import csv
import json
import random
from pathlib import Path
import numpy as np
from preprocessing import get_num_channels


def parse_feature_dims(raw_dims):
    values = [token.strip() for token in raw_dims.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError("--feature-dims must provide exactly 3 comma-separated integers, e.g. 64,128,256")
    dims = tuple(int(v) for v in values)
    if min(dims) <= 0:
        raise ValueError("--feature-dims values must be positive")
    return dims


def make_loader(dataset, shuffle, batch_size, num_workers, pin_memory, prefetch_factor, persistent_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def train_one_epoch(loader, model, optimizer, criterion, scaler, device, amp_enabled, log_interval, grad_clip):
    model.train()
    running_loss = 0.0
    batches = 0

    for step, (voxels, labels) in enumerate(loader, start=1):
        voxels = voxels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(voxels)
            loss = criterion(pred, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
        batches += 1

        if log_interval > 0 and step % log_interval == 0:
            print(f"step {step}/{len(loader)} loss={loss.item():.5f}")

    return running_loss / max(1, batches)


def evaluate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for voxels, labels in loader:
            voxels = voxels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            pred = model(voxels)
            loss = criterion(pred, labels)

            running_loss += loss.item()
            batches += 1

            predicted = pred.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(1, batches)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def maybe_plot_history(history_rows, plot_file):
    if not plot_file:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"plot skipped: matplotlib unavailable ({exc})")
        return

    epochs = [row["epoch"] for row in history_rows]
    train_losses = [row["train_loss"] for row in history_rows]
    val_losses = [row["val_loss"] for row in history_rows]
    val_accs = [row["val_acc"] for row in history_rows]

    has_val = any(v is not None for v in val_losses)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="train_loss", marker="o")
    if has_val:
        val_loss_plot = [float("nan") if v is None else v for v in val_losses]
        axes[0].plot(epochs, val_loss_plot, label="val_loss", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_val:
        val_acc_plot = [float("nan") if v is None else (100.0 * v) for v in val_accs]
        axes[1].plot(epochs, val_acc_plot, label="val_acc", marker="o")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    if has_val:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(plot_file)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved plot to {plot_path}")


def run_benchmark(full_dataset, model, optimizer, criterion, scaler, device, amp_enabled, subset_size):
    subset_size = min(max(1, subset_size), len(full_dataset))
    subset_indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    subset_loader = make_loader(
        subset_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    print(f"benchmark: running 1 epoch on subset_size={subset_size} (full={len(full_dataset)})")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    benchmark_loss = train_one_epoch(
        subset_loader,
        model,
        optimizer,
        criterion,
        scaler,
        device,
        amp_enabled,
        log_interval=0,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    scale_factor = len(full_dataset) / subset_size
    estimated_epoch_seconds = elapsed * scale_factor
    return benchmark_loss, elapsed, estimated_epoch_seconds


def main():
    parser = argparse.ArgumentParser(description="Train EventCNN")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to train on")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files to use")
    parser.add_argument("--val-max-files", type=int, default=None, help="Optional cap on validation files")
    parser.add_argument("--benchmark", action="store_true", help="Run subset benchmark and estimate full training time")
    parser.add_argument("--benchmark-size", type=int, default=128, help="Number of samples used in benchmark subset")
    parser.add_argument("--log-interval", type=int, default=50, help="Print every N steps (0 disables step logs)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible comparisons")
    parser.add_argument("--output", default="model.pth", help="Path for trained model checkpoint")
    parser.add_argument("--best-output", default=None, help="Path for best validation model checkpoint")
    parser.add_argument("--history-file", default="training_history.csv", help="CSV file with per-epoch metrics")
    parser.add_argument("--summary-file", default="training_summary.json", help="JSON summary file")
    parser.add_argument("--plot-file", default="training_curves.png", help="PNG output for loss/accuracy curves")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum relative validation-loss improvement")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate to prevent overfitting")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization weight decay")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: use config.LR)")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau", "cosine"],
        default="reduce_on_plateau",
        help="LR scheduler: none | reduce_on_plateau | cosine",
    )
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau reduction factor")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR floor for scheduler")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon for CrossEntropyLoss (e.g. 0.1 reduces overconfident spikes)")
    parser.add_argument("--feature-dims", default="64,128,256", help="Comma-separated channel widths for model stages")
    parser.add_argument("--blocks-per-stage", type=int, default=2, help="Residual blocks per model stage")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: use config.BATCH_SIZE)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (default: use config.NUM_WORKERS)")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Batches prefetched per worker (default: use config.PREFETCH_FACTOR)")
    parser.add_argument("--time-bins", type=int, default=None, help="Number of temporal bins for time_bins preprocessing")
    parser.add_argument("--delta-t", type=int, default=None, help="Window duration in microseconds for all window modes using DELTA_T")
    parser.add_argument(
        "--input-normalization",
        choices=["none", "maxabs", "zscore"],
        default=None,
        help="Per-sample normalization applied after preprocessing",
    )
    parser.add_argument("--bbox-jitter-us", type=int, default=0, help="Train-only random bbox time-shift in microseconds (augmentation)")
    parser.add_argument("--active-slice-stride-us", type=int, default=None, help="Stride (us) when scanning candidate active windows")
    parser.add_argument("--active-slice-top-k", type=int, default=1, help="Use random window among top-k most active slices during training")
    parser.add_argument("--cache-preprocessed", action="store_true", help="Cache deterministic preprocessed tensors to disk for faster later epochs")
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "val_acc"],
        default="val_loss",
        help="Metric used for best checkpoint and early stopping",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["event_frame", "polarity_frame", "time_bins", "tbr", "time_surface"],
        default=None,
        help="Override preprocessing method from config",
    )
    parser.add_argument(
        "--time-surface-decay",
        type=float,
        default=None,
        help="Exponential decay lambda for time_surface (default: config.TIME_SURFACE_DECAY=3.0)",
    )
    parser.add_argument(
        "--window-mode",
        choices=["full", "first_slice", "dense", "bbox", "active_slice", "random"],
        default=None,
        help="How to select the time window from each recording (default: config.WINDOW_MODE)",
    )
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config.DATASET_SEED = seed

    feature_dims = parse_feature_dims(args.feature_dims)
    blocks_per_stage = max(1, int(args.blocks_per_stage))

    if args.preprocessing is not None:
        config.PREPROCESSING_METHOD = args.preprocessing
    if args.window_mode is not None:
        config.WINDOW_MODE = args.window_mode
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.prefetch_factor is not None:
        config.PREFETCH_FACTOR = args.prefetch_factor
    if args.time_bins is not None:
        config.TIME_BINS = args.time_bins
    if args.delta_t is not None:
        config.DELTA_T = max(1, int(args.delta_t))
    if args.input_normalization is not None:
        config.INPUT_NORMALIZATION = args.input_normalization
    if args.time_surface_decay is not None:
        config.TIME_SURFACE_DECAY = float(args.time_surface_decay)

    if config.PREPROCESSING_METHOD != "time_bins" and args.time_bins is not None:
        print("NOTE: --time-bins is ignored unless --preprocessing time_bins")
    config.BBOX_JITTER_US = max(0, int(args.bbox_jitter_us))
    if args.active_slice_stride_us is not None:
        config.ACTIVE_SLICE_STRIDE_US = max(1, int(args.active_slice_stride_us))
    config.ACTIVE_SLICE_TOP_K = max(1, int(args.active_slice_top_k))
    if args.cache_preprocessed:
        config.CACHE_PREPROCESSED = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU (very slow)")
    print(f"Device: {device}")
    print(f"seed={seed}")
    print(
        f"loader: batch_size={config.BATCH_SIZE} num_workers={config.NUM_WORKERS} "
        f"prefetch_factor={config.PREFETCH_FACTOR} persistent_workers={config.PERSISTENT_WORKERS} "
        f"pin_memory={config.PIN_MEMORY and torch.cuda.is_available()}"
    )
    print(
        f"window_mode={config.WINDOW_MODE} bbox_jitter_us={config.BBOX_JITTER_US} "
        f"active_slice_stride_us={config.ACTIVE_SLICE_STRIDE_US} active_slice_top_k={config.ACTIVE_SLICE_TOP_K} "
        f"cache_preprocessed={config.CACHE_PREPROCESSED} delta_t={config.DELTA_T} "
        f"input_norm={config.INPUT_NORMALIZATION} time_surface_decay={config.TIME_SURFACE_DECAY}"
    )

    train_dataset = EventDataset(split=args.split, max_files=args.max_files)
    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    val_dataset = None
    val_loader = None
    try:
        val_dataset = EventDataset(split="val", max_files=args.val_max_files)
        val_loader = make_loader(
            val_dataset,
            shuffle=False,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            prefetch_factor=config.PREFETCH_FACTOR,
            persistent_workers=config.PERSISTENT_WORKERS,
        )
    except RuntimeError as exc:
        print(f"validation disabled: {exc}")

    in_channels = get_num_channels(config.PREPROCESSING_METHOD, config.TIME_BINS)
    num_classes = len(train_dataset.class_to_index)
    model = EventCNN(
        in_channels,
        num_classes=num_classes,
        dropout=args.dropout,
        feature_dims=feature_dims,
        blocks_per_stage=blocks_per_stage,
    ).to(device)
    learning_rate = args.lr if args.lr is not None else config.LR
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if args.selection_metric == "val_acc" else "min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_min,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )
    else:
        scheduler = None

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"device={device.type} amp={amp_enabled} train_samples={len(train_dataset)} train_batches={len(train_loader)}")
    print(f"dropout={args.dropout} weight_decay={args.weight_decay} lr={learning_rate} scheduler={args.lr_scheduler} grad_clip={args.grad_clip} label_smoothing={args.label_smoothing}")
    print(f"model_feature_dims={feature_dims} blocks_per_stage={blocks_per_stage}")
    print(f"preprocessing={config.PREPROCESSING_METHOD} in_channels={in_channels} num_classes={num_classes}")
    if val_loader is not None:
        print(f"val_samples={len(val_dataset)} val_batches={len(val_loader)}")

    if args.benchmark:
        bench_loss, subset_seconds, est_epoch_seconds = run_benchmark(
            train_dataset,
            model,
            optimizer,
            criterion,
            scaler,
            device,
            amp_enabled,
            args.benchmark_size,
        )
        est_total_seconds = est_epoch_seconds * max(1, args.epochs)
        print(f"benchmark_subset_loss={bench_loss:.5f}")
        print(f"benchmark_subset_time={subset_seconds:.2f}s")
        print(f"estimated_full_epoch_time={est_epoch_seconds / 60:.2f} min")
        print(f"estimated_total_time_for_{args.epochs}_epochs={est_total_seconds / 3600:.2f} h")

    best_output = args.best_output
    if not best_output:
        output_path = Path(args.output)
        best_output = str(output_path.with_name(f"{output_path.stem}_best{output_path.suffix}"))

    history_rows = []
    best_val_loss = None
    best_val_acc = None
    best_metric_value = None
    best_epoch = 0
    no_improve_count = 0
    stopped_early = False

    history_path = Path(args.history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="", encoding="utf-8") as history_fp:
        writer = csv.DictWriter(
            history_fp,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "lr",
                "epoch_seconds",
                "is_best",
                "no_improve_count",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            train_loss = train_one_epoch(
                train_loader,
                model,
                optimizer,
                criterion,
                scaler,
                device,
                amp_enabled,
                args.log_interval,
                args.grad_clip,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - t0

            val_loss = None
            val_acc = None
            is_best = False

            if val_loader is not None:
                val_loss, val_acc = evaluate(val_loader, model, criterion, device)
                current_metric = val_loss if args.selection_metric == "val_loss" else val_acc
                if best_metric_value is None:
                    is_best = True
                else:
                    if args.selection_metric == "val_loss":
                        improvement = best_metric_value - current_metric
                        required_improvement = best_metric_value * args.min_delta
                        is_best = improvement > required_improvement
                    else:
                        improvement = current_metric - best_metric_value
                        required_improvement = max(1e-8, best_metric_value) * args.min_delta
                        is_best = improvement > required_improvement

                if is_best:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_metric_value = current_metric
                    best_epoch = epoch
                    no_improve_count = 0
                    torch.save(model.state_dict(), best_output)
                else:
                    no_improve_count += 1
            else:
                # Without validation data, always keep the latest weights as best.
                best_epoch = epoch
                torch.save(model.state_dict(), best_output)

            if scheduler is not None and val_loss is not None:
                sched_metric = val_acc if args.selection_metric == "val_acc" else val_loss
                if args.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(sched_metric)
                else:
                    scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "val_acc": None if val_acc is None else float(val_acc),
                "lr": float(current_lr),
                "epoch_seconds": float(epoch_seconds),
                "is_best": int(is_best),
                "no_improve_count": no_improve_count,
            }
            history_rows.append(row)
            writer.writerow(row)
            history_fp.flush()

            next_lr = optimizer.param_groups[0]["lr"]
            lr_tag = f" lr={next_lr:.2e}" if next_lr != current_lr else ""
            if val_loss is None:
                print(
                    f"epoch {epoch}/{args.epochs} train_loss={train_loss:.5f} "
                    f"time={epoch_seconds:.2f}s{lr_tag}"
                )
            else:
                print(
                    f"epoch {epoch}/{args.epochs} train_loss={train_loss:.5f} "
                    f"val_loss={val_loss:.5f} val_acc={100.0 * val_acc:.2f}% "
                    f"no_improve={no_improve_count}/{args.patience} time={epoch_seconds:.2f}s{lr_tag}"
                )

            if val_loader is not None and no_improve_count >= args.patience:
                stopped_early = True
                print(f"early stopping at epoch {epoch} (best_epoch={best_epoch}, patience={args.patience})")
                break

    torch.save(model.state_dict(), args.output)
    print(f"saved final model to {args.output}")
    print(f"saved best model to {best_output}")

    maybe_plot_history(history_rows, args.plot_file)

    summary = {
        "device": device.type,
        "amp_enabled": amp_enabled,
        "preprocessing": config.PREPROCESSING_METHOD,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": 0 if val_dataset is None else len(val_dataset),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history_rows),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "selection_metric": args.selection_metric,
        "best_metric_value": best_metric_value,
        "lr_scheduler": args.lr_scheduler,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "model_feature_dims": list(feature_dims),
        "model_blocks_per_stage": int(blocks_per_stage),
        "window_mode": config.WINDOW_MODE,
        "active_slice_stride_us": int(config.ACTIVE_SLICE_STRIDE_US),
        "active_slice_top_k": int(config.ACTIVE_SLICE_TOP_K),
        "delta_t": int(config.DELTA_T),
        "input_normalization": config.INPUT_NORMALIZATION,
        "time_surface_decay": float(config.TIME_SURFACE_DECAY),
        "label_smoothing": float(args.label_smoothing),
        "output": str(Path(args.output)),
        "best_output": str(Path(best_output)),
        "history_file": str(history_path),
        "plot_file": str(Path(args.plot_file)),
        "seed": int(seed),
    }

    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as summary_fp:
        json.dump(summary, summary_fp, indent=2)
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()