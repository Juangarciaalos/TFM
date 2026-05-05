import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import csv
import time

from GPT2 import gpt2_base, gpt2_large, gpt2_medium
from load_wikitext2 import load_wikitext2

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
    os.makedirs(f"salidas/GPT2_{args.job_id}/csv", exist_ok=True)
    csv_filename = f"salidas/GPT2_{args.job_id}/csv/{args.model}_{args.name}_L{args.max_length}_{args.job_id}.csv"

    # Cargador de datos (Wikitext)
    train_loader, _ = load_wikitext2(batch_size=args.batch_size, max_length=args.max_length)

    model = gpt2_large().to(device) if args.model == "gpt2_large" else gpt2_base().to(device)

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

            optimizer.zero_grad()
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
                
            if batch_idx >= 100: break 
    print(torch.cuda.memory_summary(device=device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2_base", choices=["gpt2_base", "gpt2_medium", "gpt2_large"])
    parser.add_argument("--name", type=str, default="base", choices=["base", "opt"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()
    train(args)