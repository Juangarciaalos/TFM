import os


def get_output_root() -> str:
    """
    Devuelve el directorio raíz de salida script.

    Si existe la variable de entorno TFM_OUTPUT_DIR, se usa esa ruta.
    En caso contrario, se usa 'salidas' como valor por defecto.
    """
    return os.environ.get("TFM_OUTPUT_DIR", "salidas")


def get_model_output_dir(model_folder: str) -> str:
    """
    Devuelve la carpeta de salida de un modelo concreto.

    Ejemplos:
        get_model_output_dir("BERT")  -> salidas/.../BERT
        get_model_output_dir("GPT2")  -> salidas/.../GPT2
        get_model_output_dir("ResNet")-> salidas/.../ResNet
    """
    return os.path.join(get_output_root(), model_folder)


def ensure_dir(path: str) -> str:
    """
    Crea un directorio si no existe y devuelve la misma ruta.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_csv_dir(model_folder: str) -> str:
    """
    Devuelve la carpeta 'csv' de un modelo.
    """
    return ensure_dir(os.path.join(get_model_output_dir(model_folder), "csv"))


def get_plots_dir(model_folder: str) -> str:
    """
    Devuelve la carpeta 'graficas' de un modelo.
    """
    return ensure_dir(os.path.join(get_model_output_dir(model_folder), "graficas"))


def get_diagnostics_dir(model_folder: str) -> str:
    """
    Devuelve la carpeta 'diagnosticos' de un modelo.
    """
    return ensure_dir(os.path.join(get_model_output_dir(model_folder), "diagnosticos"))


def build_csv_path(model_folder: str, filename_without_ext: str) -> str:
    """
    Construye la ruta completa de un CSV.
    """
    return os.path.join(get_csv_dir(model_folder), f"{filename_without_ext}.csv")


def build_plot_base_path(model_folder: str, filename_without_ext: str) -> str:
    """
    Construye la ruta base para guardar gráficas.
    """
    return os.path.join(get_plots_dir(model_folder), filename_without_ext)