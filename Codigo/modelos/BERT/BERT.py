import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class BERTLayer(nn.Module):
    """
    Bloque encoder básico de BERT.

    Cada bloque está formado por:
    - Self-Attention multi-cabeza.
    - Conexión residual.
    - Normalización.
    - Red feed-forward.
    - Dropout.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        """
        Inicializa una capa encoder de BERT.

        Args:
            hidden_dim (int): dimensión oculta de las representaciones.
            num_heads (int): número de cabezas de atención.
            dropout (float): probabilidad de dropout.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor de entrada con forma
                [batch_size, seq_len, hidden_dim].

        Returns:
            Tensor: salida transformada con la misma forma que la entrada.
        """
        residual = x

        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            need_weights=False
        )

        x = self.norm1(residual + self.dropout(attn_out))

        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x


class BERTEmbeddings(nn.Module):
    """
    Capa de embeddings de BERT.
    """

    def __init__(self, vocab_size, hidden_dim, max_position=512, dropout=0.1):
        """
        Inicializa la capa de embeddings.

        Args:
            vocab_size (int): tamaño del vocabulario.
            hidden_dim (int): dimensión de los embeddings.
            max_position (int): longitud máxima de secuencia soportada.
            dropout (float): probabilidad de dropout.
        """
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        Args:
            input_ids (Tensor): tensor con forma [batch_size, seq_len].

        Returns:
            Tensor: embeddings con forma [batch_size, seq_len, hidden_dim].
        """
        seq_length = input_ids.size(1)

        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        )

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
        )

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BERT(nn.Module):
    """
    Modelo BERT simplificado para clasificación de texto.
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        num_classes=2,
        dropout=0.1
    ):
        """
        Inicializa el modelo BERT.

        Args:
            vocab_size (int): tamaño del vocabulario.
            hidden_dim (int): dimensión oculta del modelo.
            num_layers (int): número de capas encoder.
            num_heads (int): número de cabezas de atención.
            num_classes (int): número de clases de salida.
            dropout (float): probabilidad de dropout.
        """
        super().__init__()

        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.layers = nn.ModuleList([
            BERTLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.use_checkpoint = False

    def forward(self, input_ids):
        """
        Args:
            input_ids (Tensor): tensor de tokens con forma [batch_size, seq_len].

        Returns:
            Tensor: logits de clasificación con forma [batch_size, num_classes].
        """
        x = self.embeddings(input_ids)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        cls_token_output = x[:, 0, :]

        logits = self.classifier(cls_token_output)
        return logits


def BERTBase(num_classes=2):
    """
    Construye una configuración BERT-Base.

    Args:
        num_classes (int): número de clases de salida.

    Returns:
        BERT: modelo BERT-Base simplificado.
    """
    return BERT(
        vocab_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        num_classes=num_classes
    )


def BERTLarge(num_classes=2):
    """
    Construye una configuración  BERT-Large.

    Args:
        num_classes (int): número de clases de salida.

    Returns:
        BERT: modelo BERT-Large simplificado.
    """
    return BERT(
        vocab_size=30522,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        num_classes=num_classes
    )