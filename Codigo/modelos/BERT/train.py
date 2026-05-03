import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
import argparse
import sys
import os
import csv
import time

from BERT import BERTBase, BERTLarge
from load_glue_sst2 import load_glue_sst2

def report_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Memoria GPU - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Peak: {max_peak:.2f} MB")

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
    os.makedirs(f"salidas/BERT_{args.job_id}/csv", exist_ok=True)
    csv_filename = f"salidas/BERT_{args.job_id}/csv/{args.model}_{args.name}_L{args.max_length}_{args.job_id}.csv"

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

            optimizer.zero_grad()

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
                
            if batch_idx >= 300: break 
        print(f"Epoch {epoch} completado en {(time.time() - epoch_start_time):.2f} segundos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert_base", choices=["bert_base", "bert_large"])
    parser.add_argument("--name", type=str, default="base", choices=["base", "opt", "opt_8bit", "base_32bit"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--optim_8bit", action="store_true")
    args = parser.parse_args()
    train(args)