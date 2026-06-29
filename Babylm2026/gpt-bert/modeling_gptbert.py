import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from configuration_gptbert import GptBertConfig


# ── Internals (from model_extra.py) ──────────────────────────────────────────

class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.Tensor().to(full_tensor.device)
        ret.set_(full_tensor[:x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


class DWAModules(nn.Module):
    def __init__(self, hidden_size, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(i + 2)) for i in range(n_blocks)])
        self.accumulator = None
        self._init_weights()

    def _init_weights(self):
        for module in self.alphas:
            module.data.zero_()
            module.data[-1] = 1.0

    def init_accumulator(self, x):
        self.accumulator = (torch.zeros((self.n_blocks + 1, *x.shape), device=x.device, dtype=x.dtype), None)
        self.accumulator = apply_inplace_set(self.accumulator, 0, x)

    def forward(self, x, block_idx):
        self.accumulator = apply_inplace_set(self.accumulator, block_idx + 1, x)
        x = torch.tensordot(self.alphas[block_idx], self.accumulator[1], dims=1)
        return x


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate, approximate='tanh')


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.in_proj_vg = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)

        position_indices = (
            torch.arange(config.max_position_embeddings).unsqueeze(1)
            - torch.arange(config.max_position_embeddings).unsqueeze(0)
        )
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self._init_weights()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            mid - 1,
            torch.abs(relative_pos).clamp(max=max_position - 1),
        )
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)).int() + mid
        return torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()

    def _init_weights(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        for proj in [self.in_proj_qk, self.in_proj_vg, self.out_proj]:
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            proj.bias.data.zero_()

    def forward(self, hidden_states, attention_mask, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = (
                torch.arange(query_len).unsqueeze(1) - torch.arange(query_len).unsqueeze(0)
            )
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)
        value, gate = self.in_proj_vg(hidden_states).chunk(2, dim=2)
        gate = F.gelu(gate)

        pos = self.in_proj_qk(self.dropout(relative_embedding))
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)
        query_pos, key_pos = pos.chunk(2, dim=-1)
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.head_size)
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.head_size)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)
        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)

        context = torch.bmm(attention_probs.flatten(0, 1), value)
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)
        context = context * gate
        context = self.post_layer_norm(context)
        context = self.out_proj(context)
        return self.dropout(context)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob),
        )
        std = math.sqrt(2.0 / (5.0 * config.hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_layers = nn.ModuleList([Attention(config) for _ in range(config.num_hidden_layers)])
        self.mlp_layers = nn.ModuleList([FeedForward(config) for _ in range(config.num_hidden_layers)])
        self.dwa_modules = DWAModules(config.hidden_size, config.num_hidden_layers * 2)

        for i, layer in enumerate(self.mlp_layers):
            layer.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, x, attention_mask, relative_embedding):
        self.dwa_modules.init_accumulator(x)
        for i, (attn, ff) in enumerate(zip(self.attention_layers, self.mlp_layers)):
            x = x + attn(x, attention_mask, relative_embedding)
            x = self.dwa_modules(x, block_idx=i * 2)
            x = x + ff(x)
            x = self.dwa_modules(x, block_idx=i * 2 + 1)
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0)),
        )
        std = math.sqrt(2.0 / (5.0 * config.hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.nonlinearity[-1].weight = subword_embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels):
        x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        return self.nonlinearity(x)


# ── HuggingFace-compatible wrappers ──────────────────────────────────────────

class GptBertPreTrainedModel(PreTrainedModel):
    config_class = GptBertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass  # weights are initialised inside each submodule


class GptBertModel(GptBertPreTrainedModel):
    """Bare GPT-BERT encoder outputting raw hidden states."""

    def __init__(self, config: GptBertConfig):
        super().__init__(config)
        self.embedding = Embedding(config)
        self.transformer = Encoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        static_embeddings, relative_embedding = self.embedding(input_ids)
        hidden_states = self.transformer(
            static_embeddings,
            attention_mask.unsqueeze(1).bool(),
            relative_embedding,
        )
        return BaseModelOutput(last_hidden_state=hidden_states)


class GptBertForMaskedLM(GptBertPreTrainedModel):
    """GPT-BERT with a masked-language-modelling head (EMA checkpoint compatible)."""

    def __init__(self, config: GptBertConfig):
        super().__init__(config)
        self.embedding = Embedding(config)
        self.transformer = Encoder(config)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        static_embeddings, relative_embedding = self.embedding(input_ids)
        hidden_states = self.transformer(
            static_embeddings,
            attention_mask.unsqueeze(1).bool(),
            relative_embedding,
        )

        loss = None
        logits = None
        if labels is not None:
            logits = self.classifier(hidden_states, labels)
            gold = labels.flatten()
            gold = gold[gold != -100]
            loss = F.cross_entropy(logits, gold)

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)
