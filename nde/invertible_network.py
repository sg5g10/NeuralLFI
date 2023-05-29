import jax
import flax
import jax.numpy as jnp
import flax.linen as nn
from jax._src.lax.lax import real
from jax import random
from jax.experimental import checkify
from typing import Sequence, Callable

class CouplingNet(nn.Module):
  features: Sequence[int]
  n_out: int 
  def setup(self):
    self.layers = [nn.Dense(feat,kernel_init=nn.initializers.glorot_uniform(),
                            bias_init=nn.initializers.zeros) for feat in self.features]
    self.output = nn.Dense(self.n_out,kernel_init=nn.initializers.glorot_uniform(),
                           bias_init=nn.initializers.zeros)

  def __call__(self, x, y):
    z = jnp.concatenate((x,y), axis=-1)
    for i, lyr in enumerate(self.layers):
      z = lyr(z)
      if i != len(self.layers) - 1:
        z = nn.elu(z)
    z = self.output(z)
    z = nn.elu(z)
    return z

class ConditionalInvertibleBlock(nn.Module):
    hidden_features: Sequence[int]
    alpha: float
    theta_dim: int
    permute: bool  
    key: random.PRNGKey

    def setup(self):
        self.n_out1 = self.theta_dim // 2
        self.n_out2 = self.theta_dim // 2 if self.theta_dim % 2 == 0 else self.theta_dim // 2 + 1
        self.s1 = CouplingNet(self.hidden_features, self.n_out1)
        self.t1 = CouplingNet(self.hidden_features, self.n_out1)
        self.s2 = CouplingNet(self.hidden_features, self.n_out2)
        self.t2 = CouplingNet(self.hidden_features, self.n_out2)
    
        if self.permute:
            self.permutation_vec = random.permutation(self.key,self.theta_dim)    
        
        
    def __call__(self, theta, y, inverse=False, log_det_J=True):
        if not inverse:
            if self.permute:
                theta = jnp.take(theta, self.permutation_vec, axis=-1)
            u1, u21, u22 = jnp.split(theta, [self.n_out1, self.n_out2], axis=-1)
            u2 = jnp.concatenate((u21, u22), axis=-1)
            s1 = self.s1(u2, y)
            if self.alpha is not None:
                s1 = (2. * self.alpha / jnp.pi) *jnp.arctan(s1 / self.alpha)
            t1 = self.t1(u2, y)
            v1 = u1 * jnp.exp(s1) + t1
            s2 = self.s2(v1, y)
            if self.alpha is not None:
                s2 = (2. * self.alpha / jnp.pi) * jnp.arctan(s2 / self.alpha)
            t2 = self.t2(v1, y)
            v2 = u2 * jnp.exp(s2) + t2
            v = jnp.concatenate((v1, v2), axis=-1)
            return v, jnp.sum(s1, axis=-1) + jnp.sum(s2, axis=-1)
        else:
            v1, v21, v22 = jnp.split(theta, [self.n_out1, self.n_out2], axis=-1)
            v2 = jnp.concatenate((v21, v22), axis=-1)
            s2 = self.s2(v1, y)
            if self.alpha is not None:
                s2 = (2. * self.alpha / jnp.pi) * jnp.arctan(s2 / self.alpha)
            u2 = (v2 - self.t2(v1, y)) * jnp.exp(-s2)
            s1 = self.s1(u2, y)
            if self.alpha is not None:
                s1 = (2. * self.alpha / jnp.pi) * jnp.arctan(s1 / self.alpha)
            u1 = (v1 - self.t1(u2, y)) * jnp.exp(-s1)
            u = jnp.concatenate((u1, u2), axis=-1)

            if self.permute:
                u = jnp.take(u, jnp.argsort(self.permutation_vec), axis=-1)                
            return u

class DeepConditionalDensityModel(nn.Module):
    theta_dim: int
    key: random.PRNGKey
    hidden_features: Sequence[int]
    n_blocks: int = 4
    alpha: float = 1.9  
    permute: bool = False
    summary_nw: Callable = None
    
    def setup(self):
        keys = random.split(self.key, self.n_blocks)
        self.cINNs = [ConditionalInvertibleBlock(self.hidden_features,\
            self.alpha, self.theta_dim, self.permute, \
            keys[block]) for block in range(self.n_blocks)]

    def forward(self, theta, y):
        z = theta
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, y)
            log_det_Js.append(log_det_J)
            log_det_J = sum(log_det_Js)
        return z, log_det_J

    def inverse(self, theta, y):       
        for cINN in reversed(self.cINNs):
            theta = cINN(theta, y, inverse=True)        
        return theta

    def __call__(self, y, theta=None, inverse=False, sampling_key=None, n_samples=None):
        if self.summary_nw is not None:
            y = self.summary_nw(y)
        if inverse:
            z_normal_samples = random.normal(sampling_key, \
                    shape=(n_samples, y.shape[0], self.theta_dim))
            theta_samples = self.inverse(z_normal_samples, jnp.stack([y] * n_samples))
            return theta_samples
        else:
            if theta is None:
                checkify.check(theta is not None, "Theta input needed")
            return self.forward(theta, y)