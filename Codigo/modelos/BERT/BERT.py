import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

#Bloque básico de BERT: Self-Attention + Feed-Forward + Normalización + Residuals
class BERTLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        #Principal cuello de botella, escalado O(hidden_dim^2)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        #Expande la dimension a 4 veces el hidden_dim, luego la reduce de nuevo. Escalado exponencial tambien.
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        #Layer Normalization y Residuals
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(residual + self.dropout(attn_out))
        
        residual = x
        ff_out = self.ff(x)
        x = self.norm2(residual + self.dropout(ff_out))
        
        return x

#Capa de embeddings, combina embeddings de palabras y posiciones
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position=512):
        super().__init__()
        #definir el significado de la palabra
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        #Dónde está la palabra, BERT no sabe el orden si no se indica explícitamente
        self.position_embeddings = nn.Embedding(max_position, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        #Se crea un tensor de posiciones [0, 1, 2, ..., seq_length]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        #Se suman ambos embeddings
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.norm(embeddings)
        return self.dropout(embeddings)
    
#Clase principal de BERT, combina embeddings + encoders + cabezal de clasificación
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, num_classes=2):
        super(BERT, self).__init__()
        
        #1 Capa de Embeddings
        self.embeddings = BERTEmbeddings(vocab_size, hidden_dim)
        
        #2 Encoders
        self.layers = nn.ModuleList([
            BERTLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        #3 Cabezal de Clasificación (para SST-2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        #Flag de control para activar/desactivar gradient checkpointing
        self.use_checkpoint = False

    def forward(self, input_ids):
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
    return BERT(vocab_size=30522, hidden_dim=768, num_layers=12, num_heads=12, num_classes=num_classes)

def BERTLarge(num_classes=2):
    return BERT(vocab_size=30522, hidden_dim=1024, num_layers=24, num_heads=16, num_classes=num_classes)