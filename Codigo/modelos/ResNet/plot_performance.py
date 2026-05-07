import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.paths import build_csv_path, build_plot_base_path
from common.plotting import (
    compute_plot_metrics,
    load_performance_csv,
    plot_memory_usage,
)


MODEL_FOLDER = "ResNet"


def get_model_title(model_name: str) -> str:
    """
    Devuelve el nombre del modelo seleccionado.
    """
    mapping = {
        "resnet50": "ResNet-50",
        "resnet101": "ResNet-101",
    }

    return mapping.get(model_name, model_name)


def get_config_title(config_name: str) -> str:
    """
    Devuelve el nombre de la configuración utilizada.
    """
    mapping = {
        "base": "Baseline (FP32)",
        "opt": "Optimizado (AMP+CKPT)",
        "alloc_test": "Allocator test (max_split_size_mb=128)",
        "alloc_test_opt": "Allocator test + AMP+CKPT",
        "opt_snapshot": "Optimizado (AMP+CKPT) Snapshot",
        "alloc_snapshot": "Allocator Snapshot (max_split_size_mb=128)",
        "cuda_async": "Allocator backend cudaMallocAsync",
        "cuda_async_opt": "cudaMallocAsync + AMP+CKPT",
    }

    return mapping.get(config_name, config_name)


def build_filename_base(args) -> str:
    """
    Construye el nombre base para el CSV y las gráficas.
    """
    return f"{args.model}_{args.name}_S{args.image_size}_{args.job_id}"


def generate_plot(args) -> None:
    """
    Genera las gráficas de memoria para una ejecución de ResNet.

    Se generan dos imágenes:
    - Vista completa: *_full.png
    - Vista ampliada de los primeros eventos: *_zoom.png
    """
    filename_base = build_filename_base(args)

    input_csv = build_csv_path(MODEL_FOLDER, filename_base)
    output_base = build_plot_base_path(MODEL_FOLDER, filename_base)

    try:
        df = load_performance_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {input_csv}")
        return

    metrics = compute_plot_metrics(df)

    title_model = get_model_title(args.model)
    title_config = get_config_title(args.name)
    estado = " (OOM)" if metrics["hubo_oom"] else ""

    base_title = (
        f"{title_model} {title_config} - "
        f"Res: {args.image_size}x{args.image_size}{estado}\n"
        f"Peak Allocated: {metrics['peak_alloc']:.0f}MB | "
        f"Peak Reserved: {metrics['peak_reserved']:.0f}MB | "
        f"Avg Time: {metrics['avg_time']:.1f}ms"
    )

    print(f"Generando gráfica FULL y ZOOM para {args.model}...")

    plot_memory_usage(
        df=df,
        output_base=output_base,
        base_title=base_title,
        peak_mem=metrics["peak_mem"],
        hubo_oom=metrics["hubo_oom"],
        zoom_points=40,
        legend_loc="upper left",
    )

    print("Gráficas guardadas.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generador de gráficas para ResNet")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "resnet101"],
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
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
    parser.add_argument("--image_size", type=int, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    generate_plot(parse_args())