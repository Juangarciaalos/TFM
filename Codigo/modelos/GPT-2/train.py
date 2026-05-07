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

from GPT2 import gpt2_base, gpt2_large, gpt2_medium
from load_wikitext2 import load_wikitext2


MODEL_FOLDER = "GPT2"


def build_model(model_name: str, use_sdpa: bool, device: torch.device):
    """
    Construye el modelo GPT-2 seleccionado y lo mueve al dispositivo.

    Args:
        model_name (str): nombre del modelo.
        use_sdpa (bool): activa SDPA en los bloques de atención.
        device (torch.device): dispositivo de ejecución.

    Returns:
        torch.nn.Module: modelo GPT-2 inicializado.
    """
    if model_name == "gpt2_base":
        model = gpt2_base(use_sdpa=use_sdpa)

    elif model_name == "gpt2_medium":
        model = gpt2_medium(use_sdpa=use_sdpa)

    elif model_name == "gpt2_large":
        model = gpt2_large(use_sdpa=use_sdpa)

    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    return model.to(device)


def train(args):
    """
    Ejecuta el entrenamiento de GPT-2.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    snapshot_enabled = False
    if args.memory_snapshot:
        snapshot_enabled = enable_memory_snapshot(max_entries=args.snapshot_max_entries)

    filename_base = f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}"
    csv_filename = build_csv_path(MODEL_FOLDER, filename_base)

    train_loader, _ = load_wikitext2(
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = build_model(
        model_name=args.model,
        use_sdpa=args.sdpa,
        device=device,
    )

    if args.sdpa:
        print("SDPA activado en la atención de GPT-2")

    if args.checkpointing:
        model.use_checkpoint = True
        print("Gradient checkpointing activado")

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    criterion = nn.CrossEntropyLoss(ignore_index=50256)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    if args.amp:
        print("AMP activado")

    logger = PerformanceLogger(csv_filename)

    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

    print(
        f"Iniciando GPT-2 | "
        f"Modelo: {args.model} | "
        f"MaxLen: {args.max_length} | "
        f"AMP: {args.amp} | "
        f"Checkpointing: {args.checkpointing} | "
        f"SDPA: {args.sdpa}"
    )

    model.train()

    stop_training = False
    max_batches = args.snapshot_batches if args.memory_snapshot else args.max_batches

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            else:
                batch_start_time = time.perf_counter()

            logger.log(epoch, batch_idx, "batch_start")

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp and use_cuda):
                    inputs = input_ids[:, :-1]
                    targets = input_ids[:, 1:].contiguous()

                    logits = model(inputs)

                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )

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
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {batch_time:.2f}ms"
                )

            if batch_idx >= max_batches:
                break

        print(
            f"Epoch {epoch} completado en "
            f"{(time.time() - epoch_start_time):.2f} segundos"
        )

        if stop_training:
            break

    if use_cuda:
        print(torch.cuda.memory_summary(device=device))

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
    parser = argparse.ArgumentParser(description="Entrenamiento de GPT-2")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2_base",
        choices=["gpt2_base", "gpt2_medium", "gpt2_large"],
    )
    parser.add_argument(
        "--name",
        type=str,
        default="base",
        choices=[
            "base",
            "opt",
            "opt_snapshot",
            "sdpa",
            "sdpa_opt",
            "sdpa_snapshot",
            "sdpa_opt_snapshot",
        ],
    )
    parser.add_argument("--job_id", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--sdpa", action="store_true")

    parser.add_argument("--memory_snapshot", action="store_true")
    parser.add_argument("--snapshot_batches", type=int, default=30)
    parser.add_argument("--snapshot_max_entries", type=int, default=20000)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())