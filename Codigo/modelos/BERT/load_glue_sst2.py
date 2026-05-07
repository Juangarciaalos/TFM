from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def load_glue_sst2(
    batch_size=16,
    max_length=128,
    tokenizer_name="bert-base-uncased",
    shuffle_train=True,
    num_workers=0,
    pin_memory=True,
):
    """
    Carga el dataset GLUE/SST-2 para clasificación binaria de sentimiento.

    Este cargador se utiliza con BERT. Convierte las
    oraciones del dataset SST-2 en tensores de identificadores de tokens
    mediante el tokenizador de BERT.

    Args:
        batch_size (int): tamaño de batch usado en los DataLoaders.
        max_length (int): longitud máxima de secuencia. Las frases se truncan
            o rellenan hasta este tamaño.
        tokenizer_name (str): nombre del tokenizador de Hugging Face.
        shuffle_train (bool): si True, mezcla el conjunto de entrenamiento.
        num_workers (int): número de procesos auxiliares para carga de datos.
        pin_memory (bool): si True, activa memoria fijada para acelerar
            transferencias CPU-GPU cuando se usa CUDA.

    Returns:
        tuple[DataLoader, DataLoader]:
            train_loader: DataLoader del conjunto de entrenamiento.
            val_loader: DataLoader del conjunto de validación.
    """
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def encode(examples):
        """
        Tokeniza un batch de frases de SST-2.
        """
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = dataset["train"].map(encode, batched=True)
    val_dataset = dataset["validation"].map(encode, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader