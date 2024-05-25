# it is python implementation of rotational positional embedding

'''why RoPE?
because it decays with the relative distance increased,
which is desired for natural language encoding

Llama uses RoPE instead of absolute positional embedding'''

import torch
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class RoPE_config:
    batch_size=4
    block_size=2
    num_heads=2
    d_model=8