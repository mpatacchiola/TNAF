import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn

class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_ar_attn_mask: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_ar_attn_mask = use_ar_attn_mask
        
        # Attention block
        self.ln_1 = norm_layer(embedding_dim)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embedding_dim) got {input.shape}")
        
        if(self.use_ar_attn_mask):
            # In the Pytorch MultiheadAttention module "True" means "not allowed to attend"
            attn_mask = torch.triu(torch.ones(input.shape[1], input.shape[1]), diagonal=1).to(torch.bool).to(input.device)
        else:
            attn_mask = None
        
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embedding_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                embedding_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(embedding_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embedding_dim) got {input.shape}")
        
        #print(input.shape, self.pos_embedding.shape)
        
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class Transformer(nn.Module):

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        mlp_dim: int,
        representation_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        out_dim: int = 1000,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.out_dim = out_dim
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            embedding_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length
        
        # Aggregate the head layer (output)
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["pre_logits"] = nn.Linear(embedding_dim, representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(representation_size, out_dim)
        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, x: torch.Tensor):
        # Input must be: [batch, seq_length, embedding_dims]
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, embedding_dim) got {x.shape}")

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) # -> [batch, seq_length+1, embedding_dims]
        x = x[:, 0:-1, :] # Remove last element of sequence to get -> [batch, seq_length, embedding_dims]

        x = self.encoder(x)

        x = self.heads(x)

        return x


def main():
    # Testing locally the module
    my_transformer = Transformer(
        seq_length = 4,
        num_layers = 2,
        num_heads = 4,
        embedding_dim = 16,
        mlp_dim = 32,
        dropout = 0.0,
        attention_dropout = 0.0,
        out_dim = 128,
        representation_size = 64,
    )
    
    transformer_tot_params = sum(p.numel() for p in my_transformer.parameters())
    print(f"Tot-params: {transformer_tot_params}")

    x_dummy = torch.randn([32, 4, 16])
    with torch.no_grad():
        out = my_transformer(x_dummy)
    print("out-shape", out.shape)

if __name__ == "__main__":
    main()
