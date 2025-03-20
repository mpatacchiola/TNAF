import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
import random
from methods.transformer import Transformer


class BNAFLayer(nn.Module):
    def __init__(self, w_in_dim: int, input_dim: int, output_dim: int):
        super().__init__()
        self.w_net = nn.Sequential(nn.Linear(w_in_dim, 2*w_in_dim), nn.Tanh(), nn.Linear(2*w_in_dim, input_dim*output_dim))
        self.b_net = nn.Sequential(nn.Linear(w_in_dim, 2*w_in_dim), nn.Tanh(), nn.Linear(2*w_in_dim, output_dim))
        self.act_fn = torch.nn.Softplus(beta=1, threshold=20)
        self.reset_parameters()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def reset_parameters(self):
        nn.init.normal_(self.w_net[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w_net[0].bias)
        nn.init.normal_(self.w_net[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w_net[2].bias)
        nn.init.normal_(self.b_net[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.b_net[0].bias)
        nn.init.zeros_(self.b_net[2].weight)
        nn.init.zeros_(self.b_net[2].bias)

    def forward(self, input, w_embeddings, logj, block_input=None):
        w1 = self.w_net(w_embeddings).view(w_embeddings.shape[0],
                                                        w_embeddings.shape[1],
                                                        self.input_dim,
                                                        self.output_dim)
        b1 = self.b_net(w_embeddings)
        output = torch.einsum('bij,bijk->bik', input, torch.exp(w1)) + b1
        if logj is None:
            logj = torch.logsumexp(w1, dim=-2)
        else:
            logj = torch.logsumexp(w1 + logj[..., None], dim=-2)
        return output, logj


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, w_embeddings, logj, block_input=None):
        output = F.tanh(input)
        tanh_der = -2 * (input - math.log(2) + torch.nn.functional.softplus(-2 * input))
        return output, tanh_der + logj


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, w_embeddings, logj, block_input=None):
        output = F.sigmoid(input)
        return output, F.logsigmoid(input) + F.logsigmoid(-input) + logj


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, w_embeddings, logj, block_input=None):
        return input, logj


class ResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, w_embeddings, logj, block_input=None):
        return input + block_input, F.softplus(logj)


class BNAF(nn.Module):
    def __init__(self, w_in_dim: int, hidden_dim: int, last_layer=False, num_hidden_layers=1):
        super().__init__()
        intermediate_layers = []
        for _ in range(num_hidden_layers - 1):
            intermediate_layers.append(BNAFLayer(w_in_dim, hidden_dim, hidden_dim))
            intermediate_layers.append(Tanh())

        self.layers = nn.Sequential(BNAFLayer(w_in_dim, 1, hidden_dim),
                                     Tanh(),
                                    *intermediate_layers,
                                    BNAFLayer(w_in_dim, hidden_dim, 1))
        self.last_layer = last_layer

        if self.last_layer:
            self.layers.append(Sigmoid())
        else:
            self.layers.append(ResidualLayer())

    def forward(self, input, w_embeddings):
        logj = None
        output = input
        for module in self.layers:
            output, logj = module(output, w_embeddings, logj, input)
        return output, logj
 

class TNAF(nn.Module):

    def __init__(self, seq_length: int, num_layers: int, num_heads: int, embedding_dim: int, mlp_dim: int, bnaf_dim: int,
                 representation_size: int, out_dim: int,  dropout: float=0.0, attention_dropout: float=0.0,
                 last_layer=False, bnaf_hidden_layers=1):
        super().__init__()

        self.proj_net = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
        )
    
        self.transformer = Transformer(
            seq_length = seq_length,
            num_layers = num_layers,
            num_heads = num_heads,
            embedding_dim = embedding_dim,
            mlp_dim = mlp_dim,
            dropout = dropout,
            attention_dropout = attention_dropout,
            out_dim = out_dim,
            representation_size = representation_size,
        )
    
        self.bnaf = BNAF(w_in_dim=out_dim,
                         hidden_dim=bnaf_dim,
                         last_layer=last_layer,
                         num_hidden_layers=bnaf_hidden_layers
                         )

    def forward(self, x: Tensor, clamp: float=1e-12) -> Tensor:
        n_dims = x.shape[1] # sequence length
        # Project and Forward in Transformer
        w_embeddings = self.transformer(self.proj_net(x))
        # Forward in the BNAF
        y, logderivative = self.bnaf(x, w_embeddings)
        logsum_dydx = logderivative.sum(dim=(-2, -1))
        # logsum_dydx = 0.0
        # for idx in range(n_dims):
        #     boolean_mask = torch.full_like(y[..., 0], fill_value=False)
        #     boolean_mask[:,idx] = True
        #     dy_dxi = torch.autograd.grad(y[..., 0], x, grad_outputs=boolean_mask, create_graph=True)[0][:,idx,0]
        #     # dy_dxi = torch.clamp(dy_dxi, min=1e-15)
        #     assert (dy_dxi<0).sum()==0
        #     logsum_dydx += torch.log(dy_dxi)
        # TODO check that logsum_dydx has shape == batch_size
        return y, logsum_dydx


class PermutationLayer(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.permutation = torch.flip(torch.arange(n_dim), dims=(0,))
    def forward(self, x: Tensor, clamp: float=1e-12) -> Tensor:
        return x[:, self.permutation], 0.0

class TNAFSequential(nn.Module):
    def __init__(self, num_tnafs=1, n_dims=5, dropout=0.2, num_layers=4, num_heads=4, embedding_dim=16,
                 mlp_dim=64, bnaf_dim=128, attention_dropout=0.0, out_dim=128,
                 representation_size=256, bnaf_hidden_layers=1):
        super().__init__()

        self.my_net = nn.ModuleList([TNAF(
            seq_length=n_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            bnaf_dim=bnaf_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            out_dim=out_dim,
            representation_size=representation_size,
            last_layer=num_tnafs == 1,
            bnaf_hidden_layers=bnaf_hidden_layers
        )])
        for i in range(num_tnafs - 1):
            self.my_net.append(PermutationLayer(n_dims))
            self.my_net.append(TNAF(
                seq_length=n_dims,
                num_layers=num_layers,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                mlp_dim=mlp_dim,
                bnaf_dim=bnaf_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                out_dim=out_dim,
                representation_size=representation_size,
                last_layer=i == num_tnafs - 2,
                bnaf_hidden_layers=bnaf_hidden_layers
            ))

    def forward(self, x: Tensor, clamp: float=1e-12) -> Tensor:
        logsum_dydx = 0.0
        output = x
        output.requires_grad_(True)

        for module in self.my_net:
            output, logsum_dydx_ = module(output, clamp)
            logsum_dydx += logsum_dydx_
        return output[..., 0], logsum_dydx
