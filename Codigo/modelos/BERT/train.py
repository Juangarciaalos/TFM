import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
import argparse
import sys
import os
import csv
import pickle
import time

from BERT import BERTBase, BERTLarge
from load_glue_sst2 import load_glue_sst2

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
        torch.cuda.memory._record_memory_history(True)
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
        torch.cuda.memory._record_memory_history(False)
        print("Historial de memoria CUDA desactivado")
    except Exception as e:
        print(f"No se pudo desactivar el historial de memoria CUDA: {e}")

def save_memory_diagnostics(output_dir, model_name, config_name, job_id):
    """
    Guarda información avanzada del CUDA caching allocator:
    - Snapshot visualizable con pytorch.org/memory_viz
    - Resumen textual de memoria CUDA
    """
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

    try:
        if hasattr(torch.cuda.memory, "_dump_snapshot"):
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"Snapshot de memoria guardado con _dump_snapshot en: {snapshot_path}")

        elif hasattr(torch.cuda.memory, "_snapshot"):
            snapshot = torch.cuda.memory._snapshot()

            with open(snapshot_path, "wb") as f:
                pickle.dump(snapshot, f)

            print(f"Snapshot de memoria guardado con _snapshot + pickle en: {snapshot_path}")

        else:
            print("Esta versión de PyTorch no tiene _dump_snapshot ni _snapshot. No se puede guardar el pickle.")

    except Exception as e:
        print("No se pudo guardar el snapshot de memoria.")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Detalle: {repr(e)}")

    try:
        with open(summary_path, "w") as f:
            f.write(torch.cuda.memory_summary())
        print(f"Resumen de memoria guardado en: {summary_path}")
    except Exception as e:
        print(f"No se pudo guardar memory_summary: {e}")

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
    model_output_dir = os.path.join(output_root, "BERT")

    csv_dir = os.path.join(model_output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = os.path.join(
        csv_dir,
        f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}.csv"
    )

    #cargar dataloader
    train_loader, _ = load_glue_sst2(batch_size=args.batch_size, max_length=args.max_length)

    if args.model == "bert_base":
        model = BERTBase(num_classes=2).to(device)
        print("BERT Base seleccionado")
    elif args.model == "bert_large":
        model = BERTLarge(num_classes=2).to(device)
        print("BERT Large seleccionado")
    else:
        print("Modelo no reconocido, BERT Base seleccionado.")
        model = BERTBase(num_classes=2).to(device)

    if args.checkpointing:
        model.use_checkpoint = True
        print("Gradient Checkpointing activado")

    if args.optim_8bit:
        print("Optimizador AdamW de 8-bits (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)
    else:
        print("Optimizador AdamW estándar (PyTorch FP32)")
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)


    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.amp:
        print("AMP activado")

    logger = PerformanceLogger(csv_filename)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Iniciando entrenamiento: {args.model} | AMP: {args.amp} | Checkpointing: {args.checkpointing}")

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
           
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            
            logger.log(epoch, batch_idx, "batch_start")

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                
                logger.log(epoch, batch_idx, "after_forward")

                scaler.scale(loss).backward()
                logger.log(epoch, batch_idx, "after_backward")
                
                scaler.step(optimizer)
                scaler.update()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"|OOM DETECTADO en Batch {batch_idx}|")
                    logger.log(epoch, batch_idx, "OOM_CRASH")
                    
                    torch.cuda.empty_cache() 
                    break 
                else:    
                    raise e 
            
            end_event.record()
            torch.cuda.synchronize()
            batch_time = start_event.elapsed_time(end_event)
            
            logger.log(epoch, batch_idx, "after_step", batch_time=batch_time)

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
                report_gpu_memory()
                

            max_batches = args.snapshot_batches if args.memory_snapshot else 300
            if batch_idx >= max_batches:
                break

        print(f"Epoch {epoch} completado en {(time.time() - epoch_start_time):.2f} segundos")
    
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
    parser.add_argument("--model", type=str, default="bert_base", choices=["bert_base", "bert_large"])
    parser.add_argument("--name", type=str, default="base", choices=["base", "opt", "opt_8bit", "base_32bit", "opt_snapshot"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--optim_8bit", action="store_true")
    parser.add_argument("--memory_snapshot", action="store_true")
    parser.add_argument("--snapshot_batches", type=int, default=30)
    args = parser.parse_args()
    train(args)