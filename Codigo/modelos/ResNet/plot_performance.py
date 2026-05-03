import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def generate_plot(args):

    os.makedirs(f"salidas/resnet_{args.job_id}/graficas", exist_ok=True)

    input_csv = f"salidas/resnet_{args.job_id}/csv/{args.model}_{args.name}_S{args.image_size}_{args.job_id}.csv"
    output_base = f"salidas/resnet_{args.job_id}/graficas/{args.model}_{args.name}_S{args.image_size}_{args.job_id}"
    
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
    
    hubo_oom = 'OOM_CRASH' in df['event'].values
    
    if len(df) > 10:
        avg_time = df['batch_time_ms'][10:].mean()
        peak_mem = df['max_peak_mb'].max() if hubo_oom else df['allocated_mb'].max()
    else:
        avg_time = df['batch_time_ms'].mean()
        peak_mem = df['max_peak_mb'].max() if hubo_oom else df['allocated_mb'].max()
        
    title_model = "ResNet-50" if args.model == "resnet50" else "ResNet-101"
    title_config = "Baseline (FP32)" if args.name == "base" else "Optimizado (AMP+CKPT)"
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

        plt.title(f"{title_model} {title_config} - Res: {args.image_size}x{args.image_size}{estado}\nPeak: {peak_mem:.0f}MB | Avg Time: {avg_time:.1f}ms", fontsize=14)
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