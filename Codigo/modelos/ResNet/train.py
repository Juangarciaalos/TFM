import torch
import torch.nn as nn
import torch.optim as optim
from load_image_net import load_tiny_imagenet
from ResNet import ResNet50, ResNet101
import csv
import time
import sys
import argparse
import os

def report_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Memoria GPU - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Peak: {max_peak:.2f} MB")

class PerformanceLogger:
    def __init__(self, filename):
        self.filename = filename
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
    os.makedirs(f"salidas/resnet_{args.job_id}/csv", exist_ok=True)
    
    csv_filename = f"salidas/resnet_{args.job_id}/csv/{args.model}_{args.name}_S{args.image_size}_{args.job_id}.csv"

    train_loader, _ = load_tiny_imagenet(batch_size=args.batch_size, image_size=args.image_size)

    if args.model == "resnet50":
        model = ResNet50(num_classes=200).to(device)
        print("ResNet50 seleccionado")
    elif args.model == "resnet101":
        model = ResNet101(num_classes=200).to(device)
        print("ResNet101 seleccionado")
    else:
        print("Modelo no reconocido, Resnet50 seleccionado.")
        model = ResNet50(num_classes=200).to(device)

    if args.checkpointing:
        model.use_checkpointing = True
        print("Gradient checkpointing activado")

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.amp:
        print("AMP activado")

    logger = PerformanceLogger(filename=csv_filename)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Iniciando entrenamiento: {args.model} | Res: {args.image_size}x{args.image_size} | AMP: {args.amp} | Ckpt: {args.checkpointing}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            torch.cuda.reset_peak_memory_stats()
            start_event.record()
            
            logger.log(epoch, batch_idx, "batch_start")
            optimizer.zero_grad()
            
            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = model(data)
                    loss = criterion(output, target)
                
                logger.log(epoch, batch_idx, "after_forward")
                
                scaler.scale(loss).backward()
                logger.log(epoch, batch_idx, "after_backward")
                
                scaler.step(optimizer)
                scaler.update()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n| --- ¡OOM CRASH DETECTADO en Batch {batch_idx}! --- |")
                    logger.log(epoch, batch_idx, "OOM_CRASH")
                    torch.cuda.empty_cache() 
                    break 
                else:
                    raise e

            end_event.record()
            torch.cuda.synchronize()
            batch_time = start_event.elapsed_time(end_event)
            
            logger.log(epoch, batch_idx, "after_step", batch_time=batch_time)

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
            
            if batch_idx >= 100: break

        print(f">> Epoch {epoch} finalizada en {time.time() - epoch_start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101"]) 
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--name", type=str, default="base", choices=["base", "opt", "alloc_test", "alloc_test_opt"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224) 
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    train(args)