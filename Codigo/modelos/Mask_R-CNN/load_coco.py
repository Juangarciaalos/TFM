import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

def load_coco(batch_size=2, data_dir='./datasets/coco/coco2017'):
    """
    Carga el dataset COCO para Mask R-CNN.
    :param batch_size: tamaño del batch para DataLoader
    :param data_dir: directorio donde están los datos de COCO
    :return: train_loader, val_loader (DataLoaders para COCO)
    """

    #Definir transformaciones para las imágenes
    transform = transforms.Compose([transforms.ToTensor()])

    #Cargar los datasets de COCO
    train_dataset = CocoDetection(root=f'{data_dir}/train2017', annFile=f'{data_dir}/annotations/instances_train2017.json', transform=transform)
    val_dataset = CocoDetection(root=f'{data_dir}/val2017', annFile=f'{data_dir}/annotations/instances_val2017.json', transform=transform)

    #Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader