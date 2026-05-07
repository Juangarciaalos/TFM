import math
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def load_performance_csv(input_csv: str) -> pd.DataFrame:
    """
    Carga un CSV y convierte las columnas numéricas.
    """
    df = pd.read_csv(input_csv)

    numeric_columns = [
        "allocated_mb",
        "reserved_mb",
        "max_peak_mb",
        "batch_time_ms",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df["global_step"] = range(len(df))

    return df


def compute_plot_metrics(df: pd.DataFrame) -> dict:
    """
    Calcula métricas agregadas para mostrar en el título de las gráficas.
    """
    step_times = df[df["event"] == "after_step"]["batch_time_ms"]

    if len(step_times) > 10:
        avg_time = step_times.iloc[10:].mean()
    else:
        avg_time = step_times.mean()

    if avg_time is None or (isinstance(avg_time, float) and math.isnan(avg_time)):
        avg_time = 0.0

    hubo_oom = "OOM_CRASH" in df["event"].values

    peak_alloc = df["max_peak_mb"].max()
    peak_reserved = df["reserved_mb"].max()

    if peak_alloc is None or math.isnan(float(peak_alloc)):
        peak_alloc = 0.0

    if peak_reserved is None or math.isnan(float(peak_reserved)):
        peak_reserved = 0.0

    return {
        "avg_time": float(avg_time),
        "hubo_oom": bool(hubo_oom),
        "peak_alloc": float(peak_alloc),
        "peak_reserved": float(peak_reserved),
        "peak_mem": float(peak_alloc),
    }


def plot_memory_usage(
    df: pd.DataFrame,
    output_base: str,
    base_title: str,
    peak_mem: float,
    hubo_oom: bool,
    zoom_points: int = 20,
    legend_loc: str = "upper right",
) -> None:
    """
    Genera dos gráficas:
    - Vista completa.
    - Vista zoom con los primeros eventos.

    Se muestran: reserved_mb, allocated_mb ymax_peak_mb
    """
    sns.set_theme(style="whitegrid")

    def save_plot(dataframe: pd.DataFrame, suffix: str, is_zoom: bool = False) -> None:
        plt.figure(figsize=(12, 6))

        plt.plot(
            dataframe["global_step"],
            dataframe["reserved_mb"],
            label="Reserved (CUDA Cache)",
            color="#ff7f0e",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            marker="." if is_zoom else None,
        )

        plt.plot(
            dataframe["global_step"],
            dataframe["allocated_mb"],
            label="Allocated (Uso Real)",
            color="#1f77b4",
            linewidth=2,
            marker="o" if is_zoom else None,
        )

        plt.plot(
            dataframe["global_step"],
            dataframe["max_peak_mb"],
            label="Max Peak Allocated",
            color="#2ca02c",
            linestyle=":",
            linewidth=2,
        )

        plt.fill_between(
            dataframe["global_step"],
            dataframe["allocated_mb"],
            color="#1f77b4",
            alpha=0.15,
        )

        for _, row in dataframe.iterrows():
            if is_zoom and row["event"] in ["after_forward", "after_backward"]:
                label = "Fwd" if "forward" in row["event"] else "Bwd"
                plt.annotate(
                    label,
                    (row["global_step"], row["allocated_mb"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )

            if row["event"] == "OOM_CRASH":
                crash_mem = row["max_peak_mb"]
                plt.plot(
                    row["global_step"],
                    crash_mem,
                    marker="*",
                    color="red",
                    markersize=16,
                    markeredgecolor="black",
                    zorder=5,
                )
                plt.annotate(
                    f"OOM\n{crash_mem:.0f} MB",
                    (row["global_step"], crash_mem),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                )

        title_color = "darkred" if hubo_oom else "black"

        plt.title(
            f"{base_title} - {suffix.upper()}",
            fontsize=14,
            fontweight="bold",
            color=title_color,
        )

        plt.ylabel("Memoria GPU (MB)", fontsize=12)
        plt.xlabel("Pasos del Entrenamiento (Eventos)", fontsize=12)
        plt.legend(loc=legend_loc, frameon=True, fontsize=10)
        plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)

        if hubo_oom and peak_mem > 0:
            plt.ylim(0, peak_mem * 1.15)

        plt.tight_layout()

        plot_output = f"{output_base}_{suffix}.png"
        plt.savefig(plot_output, dpi=300, bbox_inches="tight")
        plt.close()

    save_plot(df, "full", is_zoom=False)

    zoom_len = min(zoom_points, len(df))
    save_plot(df.head(zoom_len), "zoom", is_zoom=True)