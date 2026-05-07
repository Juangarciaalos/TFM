from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer


def load_wikitext2(
    batch_size=2,
    max_length=256,
    tokenizer_name="gpt2",
    min_text_length=10,
    shuffle_train=True,
    num_workers=0,
    pin_memory=True,
):
    """
    Carga el dataset WikiText-2 para el entrenamiento de GPT-2.

    El texto se tokeniza con el tokenizador de GPT-2 y se devuelve en forma
    de tensores de identificadores de tokens.

    Args:
        batch_size (int): tamaño de batch usado en los DataLoaders.
        max_length (int): longitud máxima de secuencia.
        tokenizer_name (str): nombre del tokenizador de Hugging Face.
        min_text_length (int): longitud mínima de texto no vacío para conservar
            una muestra.
        shuffle_train (bool): si True, mezcla el conjunto de entrenamiento.
        num_workers (int): número de procesos auxiliares para carga de datos.
        pin_memory (bool): si True, activa memoria fijada para acelerar
            transferencias CPU-GPU cuando se usa CUDA.

    Returns:
        tuple[DataLoader, DataLoader]:
            train_loader: DataLoader del conjunto de entrenamiento.
            val_loader: DataLoader del conjunto de prueba usado como validación.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def keep_valid_text(example):
        """
        Elimina textos vacíos o demasiado cortos para el entrenamiento.
        """
        return len(example["text"].strip()) > min_text_length

    def encode(examples):
        """
        Tokeniza un batch de textos de WikiText-2.
        """
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.filter(keep_valid_text)

    train_dataset = dataset["train"].map(encode, batched=True)
    val_dataset = dataset["test"].map(encode, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids"])
    val_dataset.set_format(type="torch", columns=["input_ids"])

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