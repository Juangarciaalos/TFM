import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

#Permite importar common/ aunque el script se ejecute desde modelos/BERT/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.logger import PerformanceLogger
from common.memory import (
    disable_memory_snapshot,
    enable_memory_snapshot,
    report_gpu_memory,
    save_memory_diagnostics,
)
from common.paths import build_csv_path, get_diagnostics_dir

from BERT import BERTBase, BERTLarge
from load_glue_sst2 import load_glue_sst2


MODEL_FOLDER = "BERT"


def build_model(model_name: str, device: torch.device):
    """
    Construye el modelo BERT seleccionado y lo mueve al dispositivo indicado.

    Args:
        model_name (str): nombre del modelo. Puede ser 'bert_base' o 'bert_large'.
        device (torch.device): dispositivo de ejecución.

    Returns:
        torch.nn.Module: modelo BERT inicializado.
    """
    if model_name == "bert_base":
        print("BERT Base seleccionado")
        return BERTBase(num_classes=2).to(device)

    if model_name == "bert_large":
        print("BERT Large seleccionado")
        return BERTLarge(num_classes=2).to(device)

    raise ValueError(f"Modelo no reconocido: {model_name}")


def build_optimizer(args, model):
    """
    Construye el optimizador según la configuración.

    En la configuración normal se usa AdamW de PyTorch en FP32.
    Si se activa --optim_8bit, se usa AdamW8bit de bitsandbytes.

    Args:
        args: argumentos de línea de comandos.
        model (torch.nn.Module): modelo a optimizar.

    Returns:
        torch.optim.Optimizer: optimizador configurado.
    """
    if args.optim_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Bitsandbytes no está instalado."
            ) from exc

        print("Optimizador AdamW de 8-bit (bitsandbytes)")
        return bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)

    print("Optimizador AdamW estándar (PyTorch FP32)")
    return optim.AdamW(model.parameters(), lr=2e-5)


def train(args):
    """
    Ejecuta el entrenamiento de BERT.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    snapshot_enabled = False
    if args.memory_snapshot:
        snapshot_enabled = enable_memory_snapshot(max_entries=args.snapshot_max_entries)

    filename_base = f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}"
    csv_filename = build_csv_path(MODEL_FOLDER, filename_base)

    train_loader, _ = load_glue_sst2(
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = build_model(args.model, device)

    if args.checkpointing:
        model.use_checkpoint = True
        print("Gradient checkpointing activado")

    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    if args.amp:
        print("AMP activado")

    logger = PerformanceLogger(csv_filename)

    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

    print(
        f"Iniciando entrenamiento: {args.model} | "
        f"AMP: {args.amp} | "
        f"Checkpointing: {args.checkpointing} | "
        f"MaxLen: {args.max_length}"
    )

    stop_training = False
    max_batches = args.snapshot_batches if args.memory_snapshot else args.max_batches

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
            else:
                batch_start_time = time.perf_counter()

            logger.log(epoch, batch_idx, "batch_start")

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp and use_cuda):
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)

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

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {batch_time:.2f}ms"
                )
                report_gpu_memory()

            if batch_idx >= max_batches:
                break

        print(
            f"Epoch {epoch} completado en "
            f"{(time.time() - epoch_start_time):.2f} segundos"
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
    parser = argparse.ArgumentParser(description="Entrenamiento de BERT")

    parser.add_argument(
        "--model",
        type=str,
        default="bert_base",
        choices=["bert_base", "bert_large"],
    )
    parser.add_argument(
        "--name",
        type=str,
        default="base",
        choices=["base", "opt", "opt_8bit", "base_32bit", "opt_snapshot"],
    )
    parser.add_argument("--job_id", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--optim_8bit", action="store_true")

    parser.add_argument("--memory_snapshot", action="store_true")
    parser.add_argument("--snapshot_batches", type=int, default=30)
    parser.add_argument("--snapshot_max_entries", type=int, default=20000)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())