import torch
from torch import nn
import math
from jaxtyping import Float, Bool
from collections.abc import Iterable
from einops import rearrange, reduce, einsum, repeat
import numpy.typing as npt
import numpy as np
import os
from typing import IO, Any, BinaryIO
from cs336_basics.tokenizer import tokenizer


class MyLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):

        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))

        sigma = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = sigma,
            a = -3 * sigma,
            b = 3 * sigma
        )



    def forward(self, x: torch.Tensor)->torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class MyEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device= device, dtype= dtype))

        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )



    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:

        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_a = torch.sqrt(self.eps + torch.mean(x ** 2, dim=-1, keepdim= True))

        result = (x / RMS_a) * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):


    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 64 * math.ceil(8 / 3 * d_model / 64)

        self.d_ff = d_ff

        self.weight1 = MyLinear(d_model, self.d_ff, device=device, dtype=dtype)

        self.weight2 = MyLinear(self.d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = MyLinear(d_model, self.d_ff, device=device, dtype=dtype)


    def forward(self, in_features: torch.Tensor)->torch.Tensor:

        t = self.weight1(in_features)
        silu = t * torch.sigmoid(t)
        no_silu = self.weight3(in_features)

        result = self.weight2(silu * no_silu)


        return result

class RoPE(nn.Module):

    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()

        assert d_k % 2 == 0

        freq_ids = torch.arange(0, d_k, 2, device=device)
        freq = 1 / theta ** (freq_ids / d_k)

        position = torch.arange(0, max_seq_len, 1, device = device)

        angle = einsum(freq, position, "freq, pos -> pos freq")

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None)-> torch.Tensor:

        if token_positions is None:
            seq_len = x.size(-2)
            token_positions = torch.arange(0, seq_len, 1, device=x.device)

        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x_pair = rearrange(x, "... (pair split) -> ... pair split", split = 2)
        
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        result = torch.stack([rot_even, rot_odd], dim = -1)
        result = rearrange(result, "... pair split -> ... (pair split)")

        return result


def Softmax(x: torch.Tensor, dim: int):
    max_elem = torch.amax(x, dim = dim, keepdim=True)
    x_stable = x - max_elem
    exp_x = torch.exp(x_stable)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, "... queries d_k"],
        K: Float[torch.Tensor, "... keys d_k"],
        V: Float[torch.Tensor, "... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
)->Float[torch.Tensor, "... queries d_v"]:

    Q_KT = einsum(Q, K, "... q d_k, ... keys d_k -> ... q keys")

    d_k = Q.size(-1)

    scores = Q_KT / math.sqrt(d_k)

    if mask is None:
        mask = torch.tril(torch.ones(Q.size(-2), K.size(-2), dtype=torch.bool, device=Q.device))

    scores = scores.masked_fill(~mask, float("-inf"))
    softmax_scores = Softmax(scores, dim=-1)

    result = einsum(softmax_scores, V, "... q keys, ... keys d_v -> ... q d_v")

    return result


class Multihead_self_attention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device = None,
            dtype = None,
    ):
        super().__init__()

        self.WQ = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WK = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WV = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WO = MyLinear(d_model, d_model, device=device, dtype=dtype)

        assert d_model % num_heads == 0
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor)->torch.Tensor:
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        V = rearrange(V, "... s (h d_v) -> ... h s d_v", h = self.num_heads)

        multi_head = scaled_dot_product_attention(Q, K, V)
        multi_head = rearrange(multi_head, "... h s d_v->... s (h d_v)")

        attention = self.WO(multi_head)

        return attention


class Multihead_self_attention_with_rope(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float,
            device = None,
            dtype = None,
    ):
        super().__init__()

        self.WQ = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WK = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WV = MyLinear(d_model, d_model, device=device, dtype=dtype)
        self.WO = MyLinear(d_model, d_model, device=device, dtype=dtype)

        assert d_model % num_heads == 0
        self.num_heads = num_heads

        self.rope = RoPE(theta, d_model//num_heads, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None)->torch.Tensor:
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        V = rearrange(V, "... s (h d_v) -> ... h s d_v", h = self.num_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        multi_head = scaled_dot_product_attention(Q, K, V)
        multi_head = rearrange(multi_head, "... h s d_v->... s (h d_v)")

        attention = self.WO(multi_head)

        return attention

class Transformer_block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device = None,
            dtype = None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multihead_attention = Multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta,device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor)->torch.Tensor:
        x_norm1 = self.norm1(x)
        x_multihead_attention = self.multihead_attention(x_norm1)
        x = x + x_multihead_attention
        x_norm2 = self.norm2(x)
        x_ff = self.ff(x_norm2)

        result = x + x_ff

        return result

class Transformer_lm(nn.Module):
    def __init__(
            self,
            vocab_size: int, 
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.embedding1 = MyEmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = MyLinear(d_model, vocab_size, device=device, dtype= dtype)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.embedding1(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.linear(x)

        return x

def cross_entropy(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    x_max = torch.amax(x, -1, keepdim=True)

    sum_log = x_max.squeeze(-1) + torch.log(torch.exp(x - x_max).sum(dim = -1))

    correct_logit = torch.gather(x, -1, index = y.unsqueeze(-1)).squeeze(-1)

    result = sum_log - correct_logit

    return result.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")

        defaults = {
            "lr" : lr,
            "beta" : betas,
            "eps" : eps,
            "lam" : weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            lam = group["lam"]

            for p in group["params"]:
                if p .grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                if len(state) == 0:
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p)


                else:
                    m = state["m"]
                    v = state["v"]

                m = beta[0]*m + (1-beta[0])*grad
                v = beta[1]*v + (1-beta[1])*(grad * grad)


                p_lr = lr * math.sqrt(1 - beta[1] ** t) / (1 - beta[0] ** t)
                p.data = p.data - lr * lam * p.data
                p.data = p.data - p_lr * m / (torch.sqrt(v) + eps)

                state["t"] = t+1
                state["m"] = m
                state["v"] = v

        return loss

                
def Lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,

):

    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5*(1 + math.cos((it-warmup_iters) / (cosine_cycle_iters-warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate

    return lr

def Gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float)->None:
    eps = 1e-6
    total_square_norm = 0.0
    parameters = list(parameters)
    for p in parameters:
        if p.grad is not None:
            total_square_norm += torch.sum(p.grad * p.grad)

    total_norm = torch.sqrt(total_square_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
    return


def Get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str | torch.device, 
)->tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(0, len(dataset) - context_length, batch_size)

    inputs = np.stack([dataset[i: i+context_length] for i in starts])

    targets = np.stack([dataset[i+1:i+context_length+1] for i in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, 
        iteration: int, 
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
)-> int:

    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]

def decode(
        model: Transformer_lm,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        device: torch.device | None = None
):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    
    prompt = torch.tensor(prompt, dtype=torch.long, device=device)
    cnt = 0

    model.eval()
    with torch.inference_mode():
        while(cnt < max_new_tokens):

            outputs = model(prompt)

            logits = outputs[-1]
            logits /= temperature

            prob = Softmax(logits, -1)

            sort_prob, indices = torch.sort(prob, descending=True)
            cumsum_prob = torch.cumsum(sort_prob, -1)

            mask = torch.roll(cumsum_prob >= top_p, 1)
            mask[0] = False

            sort_prob = sort_prob.masked_fill(mask, 0.0)

            sort_prob = sort_prob / sort_prob.sum()

            sample_indices = torch.multinomial(sort_prob, 1)
            next_token = indices[sample_indices]

            prompt = torch.cat([prompt, next_token], dim=0)
            cnt += 1

            if next_token.item() == eos_token_id:
                break

    return prompt.tolist()
        

    
    
    

