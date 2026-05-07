import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import csv
import time
import pickle

from GPT2 import gpt2_base, gpt2_large, gpt2_medium
from load_wikitext2 import load_wikitext2

def report_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Memoria GPU - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Peak: {max_peak:.2f} MB")

def get_output_root():
    return os.environ.get("TFM_OUTPUT_DIR", "salidas")

def enable_memory_snapshot():
    if not torch.cuda.is_available():
        print("CUDA no está disponible.")
        return False

    try:
        torch.cuda.memory._record_memory_history(max_entries=20000)
        print("Historial de memoria CUDA activado")
        return True
    except TypeError:
        try:
            torch.cuda.memory._record_memory_history()
            print("Historial de memoria CUDA activado")
            return True
        except Exception as e:
            print(f"No se pudo activar el historial de memoria CUDA: {e}")
            return False
    except Exception as e:
        print(f"No se pudo activar el historial de memoria CUDA: {e}")
        return False
    
def disable_memory_snapshot():
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Historial de memoria CUDA desactivado")
    except Exception as e:
        print(f"No se pudo desactivar el historial de memoria CUDA: {e}")

def save_memory_diagnostics(output_dir, model_name, config_name, job_id):
    if not torch.cuda.is_available():
        return

    os.makedirs(output_dir, exist_ok=True)

    snapshot_path = os.path.join(
        output_dir,
        f"{model_name}_{config_name}_{job_id}_memory_snapshot.pickle"
    )

    summary_path = os.path.join(
        output_dir,
        f"{model_name}_{config_name}_{job_id}_memory_summary.txt"
    )

    snapshot_saved = False

    if hasattr(torch.cuda.memory, "_dump_snapshot"):
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)

            if os.path.exists(snapshot_path) and os.path.getsize(snapshot_path) > 0:
                snapshot_saved = True
                print(f"Snapshot de memoria guardado con _dump_snapshot en: {snapshot_path}")
            else:
                print("_dump_snapshot no generó un archivo válido.")

        except Exception as e:
            print("No se pudo guardar el snapshot con _dump_snapshot.")
            print(f"Tipo de error: {type(e).__name__}")
            print(f"Detalle: {repr(e)}")

    if not snapshot_saved and hasattr(torch.cuda.memory, "_snapshot"):
        try:
            snapshot = torch.cuda.memory._snapshot()

            with open(snapshot_path, "wb") as f:
                pickle.dump(snapshot, f)

            if os.path.exists(snapshot_path) and os.path.getsize(snapshot_path) > 0:
                snapshot_saved = True
                print(f"Snapshot de memoria guardado con _snapshot + pickle en: {snapshot_path}")
            else:
                print("_snapshot + pickle no generó un archivo válido.")

        except Exception as e:
            print("No se pudo guardar el snapshot con _snapshot + pickle.")
            print(f"Tipo de error: {type(e).__name__}")
            print(f"Detalle: {repr(e)}")

    if not snapshot_saved:
        print("No se pudo guardar ningún snapshot pickle.")

    try:
        with open(summary_path, "w") as f:
            f.write(torch.cuda.memory_summary())
        print(f"Resumen de memoria guardado en: {summary_path}")
    except Exception as e:
        print("No se pudo guardar memory_summary.")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Detalle: {repr(e)}")
class PerformanceLogger:
    def __init__(self, filename):
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "event", "allocated_mb", "reserved_mb", "max_peak_mb", "batch_time_ms"])

    def log(self, epoch, batch, event, batch_time=0.0):
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        max_peak = torch.cuda.max_memory_allocated() / (1024**2)
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, event, f"{allocated:.2f}", f"{reserved:.2f}", f"{max_peak:.2f}", f"{batch_time:.4f}"])

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snapshot_enabled = False

    if args.memory_snapshot:
        snapshot_enabled = enable_memory_snapshot()

    output_root = get_output_root()
    model_output_dir = os.path.join(output_root, "GPT2")

    csv_dir = os.path.join(model_output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = os.path.join(
        csv_dir,
        f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}.csv"
    )

    # Cargador de datos (Wikitext)
    train_loader, _ = load_wikitext2(batch_size=args.batch_size, max_length=args.max_length)

    if args.model == "gpt2_base":
        model = gpt2_base(use_sdpa=args.sdpa).to(device)
    elif args.model == "gpt2_medium":
        model = gpt2_medium(use_sdpa=args.sdpa).to(device)
    elif args.model == "gpt2_large":
        model = gpt2_large(use_sdpa=args.sdpa).to(device)
    else:
        raise ValueError(f"Modelo no reconocido: {args.model}")

    if args.sdpa:
        print("SDPA activado en la atención de GPT-2")
    
    if args.checkpointing:
        model.use_checkpoint = True

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    #ignorar el índice del pad_token para que no penalice por fallar en predecir el relleno.
    criterion = nn.CrossEntropyLoss(ignore_index=50256) 
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    logger = PerformanceLogger(csv_filename)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Iniciando GPT-2 | Modelo: {args.model} | MaxLen: {args.max_length}")

    model.train()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)

            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            logger.log(epoch, batch_idx, "batch_start")

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):

                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:].contiguous()
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            logger.log(epoch, batch_idx, "after_forward")

            scaler.scale(loss).backward()
            logger.log(epoch, batch_idx, "after_backward")
            
            scaler.step(optimizer)
            scaler.update()
            
            end_event.record()
            torch.cuda.synchronize()
            batch_time = start_event.elapsed_time(end_event)
            
            logger.log(epoch, batch_idx, "after_step", batch_time=batch_time)

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
                
            max_batches = args.snapshot_batches if args.memory_snapshot else 100
            if batch_idx >= max_batches:
                break

    print(torch.cuda.memory_summary(device=device))

    if args.memory_snapshot and torch.cuda.is_available():
        diagnostics_dir = os.path.join(model_output_dir, "diagnosticos")
        save_memory_diagnostics(
            output_dir=diagnostics_dir,
            model_name=args.model,
            config_name=args.name,
            job_id=args.job_id
        )

        if snapshot_enabled:
            disable_memory_snapshot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2_base", choices=["gpt2_base", "gpt2_medium", "gpt2_large"])
    parser.add_argument("--name", type=str, default="base", choices=["base", "opt", "opt_snapshot", "sdpa", "sdpa_opt", "sdpa_opt_snapshot"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--memory_snapshot", action="store_true")
    parser.add_argument("--snapshot_batches", type=int, default=30)
    parser.add_argument("--sdpa", action="store_true")

    args = parser.parse_args()
    train(args)