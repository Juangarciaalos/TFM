from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_wikitext2(batch_size=2, max_length=256):
    """
    Carga el dataset WikiText-2 para lenguaje (GPT-2).
    :param batch_size: tamaño del batch para DataLoader
    :param max_length: longitud máxima de los tokens
    :return: train_loader, val_loader (DataLoaders para WikiText-2)
    """
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    #Cargar tokenizer de GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token

    def encode(examples):
        return tokenizer(examples['text'], 
                         padding='max_length', 
                         truncation=True, 
                         max_length=max_length)


    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)

    train_dataset = dataset['train'].map(encode, batched=True)
    val_dataset = dataset['test'].map(encode, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
