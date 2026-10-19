"""Safe Transformers attention bridge: compatible encoder work only.

Unsupported semantics fall back to Transformers' own SDPA implementation. The
bridge never removes a real mask or turns causal attention into non-causal work.
"""
from collections import Counter
import math
import torch
from .runtime import flash_attn_demo


class TransformersAttentionBridge:
    def __init__(self, backend='ffi'):
        self.backend = backend
        self.cute_calls = 0
        self.fallbacks = Counter()

    def _reason(self,module,q,k,v,mask,dropout,scaling,is_causal,kwargs):
        causal = getattr(module,'is_causal',True) if is_causal is None else is_causal
        if causal:
            return 'causal'
        if mask is not None:
            return 'attention_mask'
        if dropout or module.training:
            return 'training_or_dropout'
        if kwargs.get('cache') is not None or kwargs.get('position_bias') is not None:
            return 'cache_or_position_bias'
        if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype or not q.is_cuda:
            return 'dtype_or_device'
        if q.ndim!=4 or k.ndim!=4 or v.ndim!=4 or k.shape!=v.shape:
            return 'layout'
        if q.shape[0]!=k.shape[0] or q.shape[2]!=k.shape[2] or q.shape[-1]!=k.shape[-1]:
            return 'different_sequence_or_batch'
        if q.shape[2]%64 or q.shape[-1] not in (64,128) or q.shape[1]%k.shape[1]:
            return 'unsupported_tile_or_heads'
        if scaling is not None and (not isinstance(scaling,(int,float)) or
                not math.isclose(scaling,q.shape[-1]**-0.5,rel_tol=1e-6)):
            return 'custom_scale'
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
            return 'autograd'
        return None

    def __call__(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        reason=self._reason(module,query,key,value,attention_mask,dropout,scaling,is_causal,kwargs)
        if reason is not None:
            self.fallbacks[reason]+=1
            return sdpa_attention_forward(
                module,query,key,value,attention_mask,dropout=dropout,
                scaling=scaling,is_causal=is_causal,**kwargs,
            )
        # Transformers uses B,H,L,D; the kernel uses B,L,H,D. These contiguous
        # views do not alter mask/causal semantics, and often already share layout
        # with the original dense Q/K/V projections.
        q,k,v=[t.transpose(1,2).contiguous() for t in (query,key,value)]
        out=flash_attn_demo(q,k,v,backend=self.backend)
        self.cute_calls+=1
        return out,None

    def stats(self):
        return {'cute_calls':self.cute_calls,'fallback_calls':sum(self.fallbacks.values()),
                'fallback_reasons':dict(self.fallbacks)}

    def reset_stats(self):
        self.cute_calls=0
        self.fallbacks.clear()


def register_transformers_attention(bridge,name='cute_ampere'):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface,sdpa_mask
    # Avoid the reserved flash_* namespace: Transformers routes those names
    # through its FlashAttention lazy importer instead of a plain custom bridge.
    # Register BOTH interfaces. An unknown mask backend in Transformers can
    # bypass mask creation entirely; registering only the attention is unsafe.
    AttentionInterface.register(name,bridge)
    AttentionMaskInterface.register(name,sdpa_mask)
    return name
