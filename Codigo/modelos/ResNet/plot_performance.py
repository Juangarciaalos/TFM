import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
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

def generate_plot(args):

    output_root = get_output_root()
    model_output_dir = os.path.join(output_root, "ResNet")

    csv_dir = os.path.join(model_output_dir, "csv")
    plots_dir = os.path.join(model_output_dir, "graficas")

    os.makedirs(plots_dir, exist_ok=True)

    filename_base = f"{args.model}_{args.name}_S{args.image_size}_{args.job_id}"

    input_csv, output_base = build_paths(
        get_output_root(),
        "ResNet",
        filename_base
    )
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {input_csv}")
        return

    df['allocated_mb'] = pd.to_numeric(df['allocated_mb'], errors='coerce')
    df['reserved_mb'] = pd.to_numeric(df['reserved_mb'], errors='coerce')
    df['max_peak_mb'] = pd.to_numeric(df['max_peak_mb'], errors='coerce')
    df['batch_time_ms'] = pd.to_numeric(df['batch_time_ms'], errors='coerce')
    df = df.dropna()

    df['global_step'] = range(len(df))
    
    step_times = df[df["event"] == "after_step"]["batch_time_ms"]

    if len(step_times) > 10:
        avg_time = step_times.iloc[10:].mean()
    else:
        avg_time = step_times.mean()

    hubo_oom = 'OOM_CRASH' in df['event'].values

    peak_alloc = df['max_peak_mb'].max()
    peak_reserved = df['reserved_mb'].max()
    peak_mem = peak_alloc

    title_model = "ResNet-50" if args.model == "resnet50" else "ResNet-101"
    
    if args.name == "base":
        title_config = "Baseline (FP32)"
    elif args.name == "opt":
        title_config = "Optimizado (AMP+CKPT)"
    elif args.name == "alloc_test":
        title_config = "Allocator test (max_split_size_mb=128)"
    elif args.name == "alloc_test_opt":
        title_config = "Allocator test + AMP+CKPT"
    elif args.name == "opt_snapshot":
        title_config = "Optimizado (AMP+CKPT) Snapshot"
    elif args.name == "alloc_snapshot":
        title_config = "Allocator Snapshot (max_split_size_mb=128)"
    elif args.name == "cuda_async":
        title_config = "Allocator backend cudaMallocAsync"
    elif args.name == "cuda_async_opt":
        title_config = "cudaMallocAsync + AMP+CKPT"
    else:
        title_config = args.name

    estado = " (OOM)" if hubo_oom else ""

    sns.set_theme(style="whitegrid")

    def save_plot(dataframe, suffix, is_zoom=False):
        plt.figure(figsize=(12, 6))
        
        plt.plot(dataframe['global_step'], dataframe['reserved_mb'], 
                 label='Reserved (CUDA Cache)', color='#ff7f0e', 
                 linestyle='--', alpha=0.7, marker='.' if is_zoom else None)
        
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
                                 fontsize=8, fontweight='bold')
                                 
        for _, row in dataframe.iterrows():
            if row['event'] == 'OOM_CRASH':
                crash_mem = row['max_peak_mb']
                plt.plot(row['global_step'], crash_mem, marker='o', color='red', markersize=8, zorder=5)
                plt.annotate('OOM', (row['global_step'], crash_mem),
                             textcoords="offset points", xytext=(0,8), ha='center', 
                             fontsize=9, color='red', fontweight='bold')

        plt.title(
            f"{title_model} {title_config} - Res: {args.image_size}x{args.image_size}{estado}\n"
            f"Peak Allocated: {peak_alloc:.0f}MB | Peak Reserved: {peak_reserved:.0f}MB | "
            f"Avg Time: {avg_time:.1f}ms"
        )

        plt.ylabel("Memoria GPU (MB)")
        plt.xlabel("Pasos (Eventos)")
        
        plt.legend(loc='upper left')
        
        if hubo_oom:
            plt.ylim(0, peak_mem * 1.10)
            
        plt.tight_layout()
        plt.savefig(f"{output_base}_{suffix}.png", dpi=300)
        plt.close()


    save_plot(df, "full", is_zoom=False)
    
    zoom_points = 40 if len(df) > 40 else len(df)
    save_plot(df.head(zoom_points), "zoom", is_zoom=True)
    
    print(f"Gráficas guardadas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de gráficas para ResNet")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    args = parser.parse_args()
    generate_plot(args)