import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_tiny_imagenet_transform(image_size=224):
    """
    Construye las transformaciones usadas para Tiny ImageNet.

    Se redimensiona la imagen manteniendo una proporción similar a la usada
    habitualmente en ImageNet y después se aplica un CenterCrop al tamaño
    objetivo. Finalmente, se convierte a tensor y se normaliza con las medias
    y desviaciones estándar de ImageNet.

    Args:
        image_size (int): resolución final de las imágenes cuadradas.

    Returns:
        torchvision.transforms.Compose: transformaciones de preprocesado.
    """
    resize_size = int(image_size * (256 / 224))

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def load_tiny_imagenet(
    batch_size=64,
    num_workers=4,
    data_dir="../../datasets/tiny-imagenet-200",
    image_size=224,
    shuffle_train=True,
    pin_memory=True,
):
    """
    Carga Tiny ImageNet mediante ImageFolder.

    Este cargador con ResNet. El dataset debe estar organizado en carpetas 
    compatibles con torchvision.datasets.ImageFolder:

        data_dir/
        ├── train/
        │   ├── class_1/
        │   ├── class_2/
        │   └── ...
        └── val/
            ├── class_1/
            ├── class_2/
            └── ...

    Args:
        batch_size (int): tamaño de batch usado en los DataLoaders.
        num_workers (int): número de procesos auxiliares para carga de datos.
        data_dir (str): ruta raíz del dataset Tiny ImageNet.
        image_size (int): resolución final de entrada al modelo.
        shuffle_train (bool): si True, mezcla el conjunto de entrenamiento.
        pin_memory (bool): si True, activa memoria fijada para acelerar
            transferencias CPU-GPU cuando se usa CUDA.

    Returns:
        tuple[DataLoader, DataLoader]:
            train_loader: DataLoader del conjunto de entrenamiento.
            val_loader: DataLoader del conjunto de validación.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform = build_tiny_imagenet_transform(image_size=image_size)

    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform,
    )

    val_dataset = ImageFolder(
        root=val_dir,
        transform=transform,
    )

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