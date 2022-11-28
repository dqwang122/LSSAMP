import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

from .distribute import all_reduce


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))   # [0, 1]

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)        # [batch_size, emb_dim, 1]
        num_arbitrary_dims = len(ctx.dims) - 2  
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, p=2, dim=1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


class NearestEmbedEMA(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = embeddings_dim
        self.n_emb = num_embeddings
        self.emb_dim = embeddings_dim
        embed = torch.rand(embeddings_dim, num_embeddings)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(embeddings_dim))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            print(x.size())
            flatten = x.permute(1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        return result, argmin


class EMAQuantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.999, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)
            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # diff = (quantize.detach() - input).pow(2).mean()
        z_q = input + (quantize - input).detach()
        embed_ind = embed_ind.unsqueeze(-1)


        return z_q, embed_ind, input, quantize

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class EMAQuantizeList(nn.Module):
    def __init__(self, args, dim, n_embed, decay=0.999, eps=1e-5, num=1, step=1):
        super().__init__()

        self.args = args
        self.dim = dim
        self.num = num
        self.step = step
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(x, dim), stride=step, padding='same') for x in args.vqvae_list])
        self.vq_list = nn.ModuleList([EMAQuantize(dim, n_embed, decay, eps) for _ in range(num)])

        if self.args.code_mode == 'sum':
            self.gate_proj = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num)])
        else:
            self.proj = nn.Linear(num * dim, dim)

    def forward(self, input):
        """ param: input [batch_size, seq_len, latent_dim] """
        conv_input = input.unsqueeze(1)
        conv_output = [F.relu(conv(conv_input)) for conv in self.convs]            # kernel_sizes * (batch_size, Co=dim, seq_len, latent_dim)
        vq_input = [x.view(-1, self.dim) for x in conv_output]
        vq_output = [vq(x) for (x,vq) in zip(vq_input, self.vq_list)]

        z_q, argmin, inputs, quantizes = zip(*vq_output)
        N = len(z_q)

        if self.args.code_mode == 'sum':
            gate_input = z_q[0]
            gate = torch.cat([gp(gate_input.view(-1, self.dim)) for gp in self.gate_proj], dim=-1)
            gate = F.softmax(gate, dim=-1)
            z_q = torch.stack([z_q[i] * gate[:,i].unsqueeze(-1) for i in range(N)], dim=-1)
            z_q = torch.sum(z_q, dim=-1)
        else:
            z_q = torch.cat(z_q, dim=-1)
            z_q = self.proj(z_q)
        
        inputs = torch.cat(inputs, dim=-1)
        quantizes = torch.cat(quantizes, dim=-1)
        argmin = torch.cat(argmin, dim=-1)          # [bz * seq_len, sub_book]
        
        return z_q, argmin, inputs, quantizes

    def embed_code(self, embed_id):
        """ param: embed_id : [batch_size, seq_len, sub_book] """
        bz, seq_len, sub_book = embed_id.size()
        embed_id = embed_id.view(bz*seq_len, -1).permute(1, 0)            # [sub_book, batch_size * seq_len]
        quantize_list = []
        for i in range(len(embed_id)):
            quantize = self.vq_list[i].embed_code(embed_id[i])
            quantize_list.append(quantize)   
        
        if self.args.code_mode == 'sum':
            gate_input = quantize_list[0]
            gate = torch.cat([gp(gate_input.view(-1, self.dim)) for gp in self.gate_proj], dim=-1)
            gate = F.softmax(gate, dim=-1)
            # print(gate)
            z_q = torch.stack([quantize_list[i] * gate[:,i].unsqueeze(-1) for i in range(sub_book)], dim=-1)
            z_q = torch.sum(z_q, dim=-1)
        else:
            quantizes = torch.cat(quantize_list, dim=-1)     #  [batch_size * seq_len, sub_book * dim]
            z_q = self.proj(quantizes)
        
        return z_q

        




