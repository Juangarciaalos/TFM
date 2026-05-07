import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.logger import PerformanceLogger
from common.memory import (
    disable_memory_snapshot,
    enable_memory_snapshot,
    save_memory_diagnostics,
)
from common.paths import build_csv_path, get_diagnostics_dir

from load_image_net import load_tiny_imagenet
from ResNet import ResNet50, ResNet101


MODEL_FOLDER = "ResNet"


def build_model(model_name: str, device: torch.device):
    """
    Construye el modelo ResNet seleccionado y lo mueve al dispositivo.

    Args:
        model_name (str): nombre del modelo. Puede ser 'resnet50' o 'resnet101'.
        device (torch.device): dispositivo de ejecución.

    Returns:
        torch.nn.Module: modelo ResNet inicializado.
    """
    if model_name == "resnet50":
        print("ResNet50 seleccionado")
        return ResNet50(num_classes=200).to(device)

    if model_name == "resnet101":
        print("ResNet101 seleccionado")
        return ResNet101(num_classes=200).to(device)

    raise ValueError(f"Modelo no reconocido: {model_name}")


def train(args):
    """
    Ejecuta el entrenamiento de ResNet.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    snapshot_enabled = False
    if args.memory_snapshot:
        snapshot_enabled = enable_memory_snapshot(max_entries=args.snapshot_max_entries)

    filename_base = f"{args.model}_{args.name}_S{args.image_size}_{args.job_id}"
    csv_filename = build_csv_path(MODEL_FOLDER, filename_base)

    train_loader, _ = load_tiny_imagenet(
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    model = build_model(args.model, device)

    if args.checkpointing:
        model.use_checkpointing = True
        print("Gradient checkpointing activado")

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    if args.amp:
        print("AMP activado")

    logger = PerformanceLogger(csv_filename)

    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

    print(
        f"Iniciando entrenamiento: {args.model} | "
        f"Res: {args.image_size}x{args.image_size} | "
        f"AMP: {args.amp} | "
        f"Ckpt: {args.checkpointing}"
    )

    stop_training = False
    max_batches = args.snapshot_batches if args.memory_snapshot else args.max_batches

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            else:
                batch_start_time = time.perf_counter()

            logger.log(epoch, batch_idx, "batch_start")

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp and use_cuda):
                    output = model(data)
                    loss = criterion(output, target)

                logger.log(epoch, batch_idx, "after_forward")

                scaler.scale(loss).backward()
                logger.log(epoch, batch_idx, "after_backward")

                scaler.step(optimizer)
                scaler.update()

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"|OOM DETECTADO en Batch {batch_idx}|")
                    logger.log(epoch, batch_idx, "OOM_CRASH")

                    if use_cuda:
                        torch.cuda.empty_cache()

                    stop_training = True
                    break

                raise

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event)
            else:
                batch_time = (time.perf_counter() - batch_start_time) * 1000

            logger.log(epoch, batch_idx, "after_step", batch_time=batch_time)

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {batch_time:.2f}ms"
                )

            if batch_idx >= max_batches:
                break

        print(
            f"Epoch {epoch} finalizada en "
            f"{(time.time() - epoch_start_time):.2f}s"
        )

        if stop_training:
            break

    if args.memory_snapshot and use_cuda:
        diagnostics_dir = get_diagnostics_dir(MODEL_FOLDER)

        save_memory_diagnostics(
            output_dir=diagnostics_dir,
            model_name=args.model,
            config_name=args.name,
            job_id=args.job_id,
        )

        if snapshot_enabled:
            disable_memory_snapshot()


def parse_args():
    """
    Define y parsea los argumentos.
    """
    parser = argparse.ArgumentParser(description="Entrenamiento de ResNet")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101"],
    )
    parser.add_argument(
        "--name",
        type=str,
        default="base",
        choices=[
            "base",
            "opt",
            "alloc_test",
            "alloc_test_opt",
            "opt_snapshot",
            "alloc_snapshot",
            "cuda_async",
            "cuda_async_opt",
        ],
    )
    parser.add_argument("--job_id", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")

    parser.add_argument("--memory_snapshot", action="store_true")
    parser.add_argument("--snapshot_batches", type=int, default=30)
    parser.add_argument("--snapshot_max_entries", type=int, default=20000)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())