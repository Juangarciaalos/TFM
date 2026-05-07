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


MODEL_FOLDER = "BERT"


def get_model_title(model_name: str) -> str:
    """
    Devuelve el nombre del modelo seleccionado.
    """
    mapping = {
        "bert_base": "BERT-Base",
        "bert_large": "BERT-Large",
    }

    return mapping.get(model_name, model_name)


def get_config_title(config_name: str) -> str:
    """
    Devuelve el nombre de la configuración utilizada.
    """
    mapping = {
        "base": "Baseline (FP32)",
        "opt": "Optimizado (AMP+CKPT)",
        "base_32bit": "Baseline AdamW 32-bit",
        "opt_8bit": "AdamW 8-bit",
        "opt_snapshot": "Optimizado (AMP+CKPT) Snapshot",
    }

    return mapping.get(config_name, config_name)


def build_filename_base(args) -> str:
    """
    Construye el nombre base para el CSV y las gráficas.
    """
    return f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}"


def generate_individual_plots(args) -> None:
    """
    Genera las gráficas de memoria para una ejecución de BERT.

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
        f"{title_model} {title_config} - L={args.max_length}{estado}\n"
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
        zoom_points=20,
        legend_loc="upper right",
    )

    print("Gráficas guardadas.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generador de gráficas para BERT")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bert_base", "bert_large"],
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=["base", "opt", "opt_8bit", "base_32bit", "opt_snapshot"],
    )
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    generate_individual_plots(parse_args())