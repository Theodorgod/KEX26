import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from loader import EventDataset
from model import EventCNN
import config
import argparse
import time


def make_loader(dataset, shuffle):
    kwargs = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY and torch.cuda.is_available(),
    }
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def train_one_epoch(loader, model, optimizer, criterion, scaler, device, amp_enabled, log_interval):
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        batches += 1

        if log_interval > 0 and step % log_interval == 0:
            print(f"step {step}/{len(loader)} loss={loss.item():.5f}")

    return running_loss / max(1, batches)


def run_benchmark(full_dataset, model, optimizer, criterion, scaler, device, amp_enabled, subset_size):
    subset_size = min(max(1, subset_size), len(full_dataset))
    subset_indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    subset_loader = make_loader(subset_dataset, shuffle=False)

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
    parser.add_argument("--benchmark", action="store_true", help="Run subset benchmark and estimate full training time")
    parser.add_argument("--benchmark-size", type=int, default=128, help="Number of samples used in benchmark subset")
    parser.add_argument("--log-interval", type=int, default=50, help="Print every N steps (0 disables step logs)")
    parser.add_argument("--output", default="model.pth", help="Path for trained model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = EventDataset(split=args.split, max_files=args.max_files)
    loader = make_loader(dataset, shuffle=True)

    model = EventCNN(config.TIME_BINS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"device={device.type} amp={amp_enabled} samples={len(dataset)} batches={len(loader)}")

    if args.benchmark:
        bench_loss, subset_seconds, est_epoch_seconds = run_benchmark(
            dataset,
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

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        epoch_loss = train_one_epoch(
            loader,
            model,
            optimizer,
            criterion,
            scaler,
            device,
            amp_enabled,
            args.log_interval,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} loss={epoch_loss:.5f} time={epoch_seconds:.2f}s")

    torch.save(model.state_dict(), args.output)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()