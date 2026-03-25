import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_tiny_imagenet(batch_size=64, num_workers=4, data_dir='../../datasets/tiny-imagenet-200'):
    """
    Carga Tiny ImageNet para entrenamiento con ResNet-50.
    :param batch_size: tamaño del batch para DataLoader
    :param num_workers: número de trabajadores
    :param data_dir: directorio donde sse almacenan los datos
    :return: train_loader, val_loader (DataLoaders para entrenamiento y validación)
    """
    
    transform = transforms.Compose([
        transforms.Resize(256),             
        transforms.CenterCrop(224),         
        transforms.ToTensor(),               
        transforms.Normalize(                #Normalización con valores de ImageNet
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)

    #Crear DataLoaders de entrenamiento y validación
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
