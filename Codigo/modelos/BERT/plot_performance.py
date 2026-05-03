import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

def generate_individual_plots(args):

    os.makedirs(f"salidas/BERT_{args.job_id}/graficas", exist_ok=True)

    input_csv = f"salidas/BERT_{args.job_id}/csv/{args.model}_{args.name}_L{args.max_length}_{args.job_id}.csv"
    output_base = f"salidas/BERT_{args.job_id}/graficas/{args.model}_{args.name}_L{args.max_length}_{args.job_id}"
    
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
        
    title_model = "BERT-Base" if args.model == "bert_base" else "BERT-Large"
    title_config = "Baseline (FP32)" if args.name == "base" else "Optimizado (AMP+CKPT)"
    
    estado = "¡OUT OF MEMORY!" if hubo_oom else "Completado"
    base_title = f"{title_model} ({title_config}) - L={args.max_length}\nEstado: {estado} | Peak: {peak_mem:.0f}MB | Time Batch (Avg): {avg_time:.1f}ms"

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
        
        plt.fill_between(dataframe['global_step'], dataframe['allocated_mb'], color='#1f77b4', alpha=0.15)
        
        for _, row in dataframe.iterrows():
            if is_zoom and row['event'] in ['after_forward', 'after_backward']:
                label = 'Fwd' if 'forward' in row['event'] else 'Bwd'
                plt.annotate(label, (row['global_step'], row['allocated_mb']),
                             textcoords="offset points", xytext=(0,10), ha='center', 
                             fontsize=8, fontweight='bold', color='black')
                             
            if row['event'] == 'OOM_CRASH':
                crash_mem = row['max_peak_mb']
                plt.plot(row['global_step'], crash_mem, marker='*', color='red', markersize=20, markeredgecolor='black')
                plt.annotate(f'OOM CRASH!\n{crash_mem:.0f} MB', 
                             (row['global_step'], crash_mem),
                             textcoords="offset points", xytext=(0,15), ha='center', 
                             fontsize=12, fontweight='bold', color='red',
                             bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.9))

        title_color = 'darkred' if hubo_oom else 'black'
        plt.title(f"{base_title} - {suffix.upper()}", fontsize=14, fontweight='bold', color=title_color)
        
        if hubo_oom:
            plt.ylim(0, peak_mem * 1.15) 
        plt.xlabel("Pasos del Entrenamiento (Eventos)", fontsize=12)
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
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
    parser = argparse.ArgumentParser(description="Generador de gráficas para BERT")
    parser.add_argument("--model", type=str, required=True, choices=["bert_base", "bert_large"])
    parser.add_argument("--name", type=str, required=True, choices=["base", "opt", "opt_8bit", "base_32bit"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()
    generate_individual_plots(args)