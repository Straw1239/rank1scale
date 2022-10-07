import torch
import torch.fx
from itertools import *
import copy
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.jit as jit
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import re
import torch.optim as optim
import torch.cuda.amp
import matplotlib.pyplot as plt

def cast_signed(x, bits):
  return torch.clamp(torch.round(x), -(1 << bits)/2,  ((1 << bits)/2 -1))

def quantize(x, offset, scale, bits):
  return cast_signed((x - offset) / scale * (1 << bits),  bits)

def dmd_multiply(x, y, quantize):
  xd1, xm, xd2 = x
  yd1, ym, yd2 = y
  
  mid = xd1 * yd2
  scale = torch.max(mid, -1)[0]#torch.exp(torch.mean(torch.log(mid)))
  mid /= scale.unsqueeze(-1)
  corrections = torch.sqrt(mid)
 
  xm = quantize(xm*corrections.unsqueeze(-2))
  ym = quantize(ym*corrections.unsqueeze(-1))
  #print(xd2.shape, scale.shape)
  return yd1, torch.matmul(xm, ym), xd2 * scale.unsqueeze(-1)

def quantize_dmd_alternating(x, bits, qscale, iters=2):
  scales = [torch.ones(x.shape[:-2]+(x.shape[i-2],), device=x.device) for i in range(2)][::-1]
  for i in range(iters):
    dim = i % 2
    scale = qscale(x, dim - 2)
    if i == 0 and iters > 2:
      scale = torch.sqrt(scale)
    x = x / scale.unsqueeze(dim - 2)
    #print(x.shape, scale.shape, scales[0].shape, scales[1].shape, dim)
    scales[dim] *= scale
  return scales[0] , quantize(x, 0, 2, bits) , scales[1] / (1 << (bits - 1))

def max_scale(x, dim=0, eps=1e-4):
  m, _ = torch.max(torch.abs(x), dim)
  return torch.clamp(m, min=eps)

def max_scale_rounded_power(x, dim=0, eps=1e-2, pow2=1):
  return torch.exp2(torch.ceil(torch.log2(max_scale(x, dim, eps))/pow2)*pow2)


def quantize_dmd_mse_gd(x, bits, iters=10):
  logscales = [torch.zeros(x.shape[:-2]+(x.shape[i-2],), requires_grad=True) for i in range(2)][::-1]
  xabs = x.abs().detach()
  opt = torch.optim.Rprop(logscales)
  for i in range(iters):
    opt.zero_grad()
    scales = [torch.exp(s) for s in logscales]
    effscales = scales[0].unsqueeze(-2) * scales[1].unsqueeze(-1)
    diff = xabs - effscales
    clip_err = sqr(F.relu(diff)).mean()
    quantize_err = sqr((diff < 0) * effscales / (1 << bits)).mean() / 12
    #print(clip_err, quantize_err)
    #print((diff < 0).float().mean())
    total_err = clip_err + quantize_err
    total_err.backward()
    opt.step()
  return scales[0], quantize(x / effscales, 0, 2, bits), scales[1] / (1 << (bits - 1))

def recreate_from_dmd(s1, m, s2):
  return s1.unsqueeze(-2) * m * s2.unsqueeze(-1)

def sqr(x):
  return x * x

model = torch.hub.load('huggingface/pytorch-transformers', 'modelForCausalLM', 'gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
full_model = copy.deepcopy(model)

def quantized_if_needed(x, bits):
  if len(x.shape) == 1:
    return x
  return recreate_from_dmd(*quantize_dmd_alternating(x, bits, max_scale, 4))

class QMatmul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, y, bits):
    xq = recreate_from_dmd(*quantize_dmd_alternating(x, bits, max_scale_rounded_power))
    yq = recreate_from_dmd(*quantize_dmd_alternating(y, bits, max_scale_rounded_power))
    ctx.save_for_backward(xq, yq)
    ctx.bits = bits
    return torch.matmul(xq, yq)
  def backward(ctx, grad_output):
    xq, yq = ctx.saved_tensors
    bits = ctx.bits
    gq = recreate_from_dmd(*quantize_dmd_alternating(grad_output, bits, max_scale_rounded_power))
    return torch.matmul(gq, yq.transpose(-1, -2)), torch.matmul(xq.transpose(-1, -2), gq), None
    #return torch.matmul(grad_output, y.transpose(-1, -2)), torch.matmul(x.transpose(-1, -2), grad_output), None
qmatmul = QMatmul.apply


class QConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, mod, bits):
        super().__init__()
        self.bits = bits
        self.weight = mod.weight
        self.bias = mod.bias
        self.nf = mod.nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = qmatmul(x.view(-1, x.size(-1)), self.weight, self.bits) + self.bias
        x = x.view(size_out)
        return x


class QuantizedMLP(nn.Module):
    def __init__(self, mlp, bits):
        super().__init__()
        self.c_fc = QConv1D(mlp.c_fc, bits)
        self.c_proj = QConv1D(mlp.c_proj, bits)
        self.act = mlp.act

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

def quantize_gpt(gpt, bits):
  gpt = copy.deepcopy(gpt)
  for block in gpt.transformer.h:
    block.mlp = QuantizedMLP(block.mlp, bits)
    block.attn = QAttention(block.attn, bits)
  return gpt

class QAttention(nn.Module):
    def __init__(self, attn, bits):
        super().__init__()
        self.bits = bits
        self.register_buffer("bias", attn.bias),
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = attn.scale_attn_weights
        self.is_cross_attention = attn.is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = attn.scale_attn_by_inverse_layer_idx
        self.layer_idx = attn.layer_idx
        self.reorder_and_upcast_attn = attn.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = QConv1D(attn.c_attn, bits)
            self.q_attn = QConv1D(attn.q_attn, bits)
        else:
            self.c_attn = QConv1D(attn.c_attn, bits)
        self.c_proj = QConv1D(attn.c_proj, bits)

        self.attn_dropout = attn.attn_dropout
        self.resid_dropout = attn.attn_dropout

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = qmatmul(query, key.transpose(-1, -2), self.bits)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = qmatmul(attn_weights, value, self.bits)
        
        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
                attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past = None,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        use_cache = False,
        output_attentions = False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

with open('wikitext-103-raw/wiki.train.raw', 'r') as f:
    train = f.read()
heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'
train=  re.split(heading_pattern, train)

tokenizer.pad_token = tokenizer.eos_token
tokens = tokenizer(train[:50000], truncation=True, return_overflowing_tokens=True, padding=True, return_tensors='pt')

def pdot(x, y):
  return sum([(a*b).sum() for a, b in zip(x, y)])

model = GPT2LMHeadModel(model.config).cuda()
model = quantize_gpt(model, 8)

batch_size = 8
results = []
opt = torch.optim.Adam(model.parameters(), 1e-4)
base_lr = 1e-4
scaler = torch.cuda.amp.GradScaler()
for i in range(5199, 12171):
  data = tokens[i*batch_size:(i+1)*batch_size]
  with torch.cuda.amp.autocast():
    loss = model(data, labels=data)['loss']
  if i % 50 == 0:
    print(loss)
  results.append(float(loss))
  scaler.scale(loss).backward()
  scaler.step(opt)
  scaler.update()
  opt.zero_grad()

