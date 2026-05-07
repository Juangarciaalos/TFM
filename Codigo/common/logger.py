import os
import csv
import torch


class PerformanceLogger:
    """
    Logger CSV para registrar eventos de entrenamiento y memoria CUDA.
    """

    HEADER = [
        "epoch",
        "batch",
        "event",
        "allocated_mb",
        "reserved_mb",
        "max_peak_mb",
        "batch_time_ms",
    ]

    def __init__(self, filename: str):
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)

    def log(self, epoch: int, batch: int, event: str, batch_time: float = 0.0) -> None:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            max_peak = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            allocated = 0.0
            reserved = 0.0
            max_peak = 0.0

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                batch,
                event,
                f"{allocated:.2f}",
                f"{reserved:.2f}",
                f"{max_peak:.2f}",
                f"{batch_time:.4f}",
            ])