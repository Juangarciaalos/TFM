import os
import pickle
import torch


def report_gpu_memory() -> None:
    """
    Imprime el estado actual de memoria GPU gestionada por PyTorch.
    """
    if not torch.cuda.is_available():
        print("CUDA no disponible.")
        return

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_peak = torch.cuda.max_memory_allocated() / (1024**2)

    print(
        f"Memoria GPU - Allocated: {allocated:.2f} MB | "
        f"Reserved: {reserved:.2f} MB | "
        f"Max Peak: {max_peak:.2f} MB"
    )


def enable_memory_snapshot(max_entries: int = 20000) -> bool:
    """
    Activa el historial de memoria CUDA para poder generar snapshots.

    Se intenta primero con max_entries.
    Si no está disponible en la versión de PyTorch usada, se intenta una llamada
    más antigua como fallback.
    """
    if not torch.cuda.is_available():
        print("CUDA no está disponible.")
        return False

    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        print(f"Historial de memoria CUDA activado con max_entries={max_entries}")
        return True

    except TypeError:
        try:
            torch.cuda.memory._record_memory_history()
            print("Historial de memoria CUDA activado con firma antigua")
            return True
        except Exception as e:
            print(f"No se pudo activar el historial de memoria CUDA: {e}")
            return False

    except Exception as e:
        print(f"No se pudo activar el historial de memoria CUDA: {e}")
        return False


def disable_memory_snapshot() -> None:
    """
    Desactiva el historial de memoria CUDA.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Historial de memoria CUDA desactivado")
    except Exception as e:
        print(f"No se pudo desactivar el historial de memoria CUDA: {e}")


def save_memory_diagnostics(
    output_dir: str,
    model_name: str,
    config_name: str,
    job_id: str,
) -> None:
    """
    Guarda diagnósticos de memoria CUDA:
    - Snapshot pickle para ver en la página de memory_viz.
    - Resumen textual con torch.cuda.memory_summary().
    """
    if not torch.cuda.is_available():
        print("CUDA no disponible.")
        return

    os.makedirs(output_dir, exist_ok=True)

    snapshot_path = os.path.join(
        output_dir,
        f"{model_name}_{config_name}_{job_id}_memory_snapshot.pickle",
    )

    summary_path = os.path.join(
        output_dir,
        f"{model_name}_{config_name}_{job_id}_memory_summary.txt",
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
            print(f"Error: {type(e).__name__}")
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
            print(f"Error: {type(e).__name__}")
            print(f"Detalle: {repr(e)}")

    if not snapshot_saved:
        print("No se pudo guardar ningún snapshot pickle.")

    try:
        with open(summary_path, "w") as f:
            f.write(torch.cuda.memory_summary())
        print(f"Resumen de memoria guardado en: {summary_path}")

    except Exception as e:
        print("No se pudo guardar memory_summary.")
        print(f"Error: {type(e).__name__}")
        print(f"Detalle: {repr(e)}")