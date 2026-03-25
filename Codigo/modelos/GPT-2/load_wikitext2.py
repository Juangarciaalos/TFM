from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_wikitext2(batch_size=2):
    """
    Carga el dataset WikiText-2 para lenguaje (GPT-2).
    :param batch_size: tamaño del batch para DataLoader
    :return: train_loader, val_loader (DataLoaders para WikiText-2)
    """
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    #Cargar tokenizer de GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    #Tokenizar textos
    def encode(examples):
        return tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True)

    train_dataset = dataset['train'].map(encode, batched=True)
    val_dataset = dataset['test'].map(encode, batched=True)

    #Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
