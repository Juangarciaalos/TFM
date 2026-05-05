import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

def get_output_root():
    return os.environ.get("TFM_OUTPUT_DIR", "salidas")

def build_paths(output_root, model_folder, filename_without_ext):
    model_output_dir = os.path.join(output_root, model_folder)
    csv_dir = os.path.join(model_output_dir, "csv")
    plots_dir = os.path.join(model_output_dir, "graficas")

    os.makedirs(plots_dir, exist_ok=True)

    input_csv = os.path.join(csv_dir, f"{filename_without_ext}.csv")
    output_base = os.path.join(plots_dir, filename_without_ext)

    return input_csv, output_base

def generate_individual_plots(args):
    output_root = get_output_root()
    model_output_dir = os.path.join(output_root, "GPT2")

    csv_dir = os.path.join(model_output_dir, "csv")
    plots_dir = os.path.join(model_output_dir, "graficas")

    os.makedirs(plots_dir, exist_ok=True)

    filename_base = f"{args.model}_{args.name}_L{args.max_length}_{args.job_id}"

    input_csv, output_base = build_paths(
        get_output_root(),
        "GPT2",
        filename_base
    )
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {input_csv}")
        return

    df['allocated_mb'] = pd.to_numeric(df['allocated_mb'], errors='coerce')
    df['reserved_mb'] = pd.to_numeric(df['reserved_mb'], errors='coerce')
    df['batch_time_ms'] = pd.to_numeric(df['batch_time_ms'], errors='coerce')
    df['max_peak_mb'] = pd.to_numeric(df['max_peak_mb'], errors='coerce')
    df = df.dropna()

    df['global_step'] = range(len(df))
    
    if len(df) > 10:
        avg_time = df['batch_time_ms'][10:].mean()
    else:
        avg_time = df['batch_time_ms'].mean()

    hubo_oom = 'OOM_CRASH' in df['event'].values

    peak_alloc = df['max_peak_mb'].max()
    peak_reserved = df['reserved_mb'].max()
    peak_mem = peak_alloc
        
    title_model = "GPT-2 Base" if args.model == "gpt2_base" else "GPT-2 Medium" if args.model == "gpt2_medium" else "GPT-2 Large"
    if args.name == "base":
        title_config = "Baseline (FP32)"
    elif args.name == "opt":
        title_config = "Optimizado (AMP+CKPT)"
    elif args.name == "opt_snapshot":
        title_config = "Optimizado (AMP+CKPT) Snapshot"
    else:
        title_config = args.name

    estado = " (OOM)" if hubo_oom else ""

    base_title = (
        f"{title_model} {title_config} - L={args.max_length}{estado}\n"
        f"Peak Allocated: {peak_alloc:.0f}MB | "
        f"Peak Reserved: {peak_reserved:.0f}MB | "
        f"Avg Time: {avg_time:.1f}ms"
)
    sns.set_theme(style="whitegrid")

    def save_plot(dataframe, suffix, is_zoom=False):
        plt.figure(figsize=(12, 6))
        
        plt.plot(dataframe['global_step'], dataframe['reserved_mb'], 
                 label='Reserved (CUDA Cache)', color='#ff7f0e', 
                 linestyle='--', alpha=0.7, linewidth=1.5,
                 marker='.' if is_zoom else None)
        
        plt.plot(dataframe['global_step'], dataframe['allocated_mb'], 
                 label='Allocated (Uso Real)', color='#1f77b4', 
                 linewidth=2, marker='o' if is_zoom else None)
        
        plt.plot(dataframe['global_step'], dataframe['max_peak_mb'],
                 label='Max Peak Allocated', color='#2ca02c',
                 linestyle=':', linewidth=2)
        
        plt.fill_between(dataframe['global_step'], dataframe['allocated_mb'], color='#1f77b4', alpha=0.15)
        
        if is_zoom:
            for _, row in dataframe.iterrows():
                if row['event'] in ['after_forward', 'after_backward']:
                    label = 'Fwd' if 'forward' in row['event'] else 'Bwd'
                    plt.annotate(label, (row['global_step'], row['allocated_mb']),
                                 textcoords="offset points", xytext=(0,10), ha='center', 
                                 fontsize=8, fontweight='bold', color='black')

        plt.title(f"{base_title} - {suffix.upper()}", fontsize=14, fontweight='bold')
        plt.ylabel("Memoria GPU (MB)", fontsize=12)
        plt.xlabel("Pasos del Entrenamiento (Eventos)", fontsize=12)
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
        if hubo_oom:
            plt.ylim(0, peak_mem * 1.15) 
        plt.tight_layout()
        
        plot_output = f"{output_base}_{suffix}.png"
        plt.savefig(plot_output, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Generando gráfica FULL para {args.model}...")
    save_plot(df, "full", is_zoom=False)
    
    print(f"Generando gráfica ZOOM para {args.model}...")
    zoom_points = 20 if len(df) > 20 else len(df)
    save_plot(df.head(zoom_points), "zoom", is_zoom=True)
    
    print(f"Gráficas guardadas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de gráficas para GPT-2")
    parser.add_argument("--model", type=str, required=True, choices=["gpt2_base", "gpt2_medium", "gpt2_large"])
    parser.add_argument("--name", type=str, required=True, choices=["base", "opt", "opt_snapshot"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()
    generate_individual_plots(args)