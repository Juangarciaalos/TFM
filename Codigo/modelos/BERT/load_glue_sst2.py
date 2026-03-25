from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_glue_sst2(batch_size=16, max_length=128):
    """
    Carga el dataset GLUE (SST-2) para entrenamiento y validación.
    :param batch_size: tamaño del batch para DataLoader
    :param max_length: longitud máxima de los tokens
    :return: train_loader, val_loader (DataLoaders para SST-2)
    """
    
    #Cargar el dataset SST-2 de GLUE
    dataset = load_dataset("glue", "sst2")

    #Cargar tokenizer de BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Tokenizar oraciones
    def encode(examples):
        return tokenizer(examples['sentence'],
                        padding='max_length',
                        truncation=True,
                        max_length=max_length)
    
    train_dataset = dataset['train'].map(encode, batched=True)
    val_dataset = dataset['validation'].map(encode, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    #Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
