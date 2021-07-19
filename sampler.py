# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_sI39MBHGRnt20inesIty9_uiUVUdJ4X
"""

# Commented out IPython magic to ensure Python compatibility.
import os
if 'TPU_NAME' in os.environ:
  import requests
  if 'TPU_DRIVER_MODE' not in globals():
    url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
    resp = requests.post(url)
    TPU_DRIVER_MODE = 1


  from jax.config import config
  config.FLAGS.jax_xla_backend = "tpu_driver"
  config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
  print('Registered TPU:', config.FLAGS.jax_backend_target)
else:
  print('No TPU detected. Can be changed under "Runtime/Change runtime type".')

import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)

import numpy as np
import torch
import librosa
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import jax
import jax.tools.colab_tpu
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Any, Tuple
import functools
import torch
from flax.serialization import to_bytes, from_bytes
import tensorflow as tf
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from scipy import integrate
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import soundfile
import librosa.display
import IPython.display as ipd
import random
import argparse
import os 

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=25.0)
parser.add_argument('--num_steps', type=int, default=500)
parser.add_argument('--pc_num_steps', type=int, default=500)
parser.add_argument('--signal_to_noise_ratio', type=float, default=0.16)
parser.add_argument('--etol', type=float, default=1e-5)
parser.add_argument('--sample_batch_size', type=int, default=64)
parser.add_argument('--sample_no', type=int, default=25)
# args = parser.parse_args(args=[]) # required for notebooks

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x):    
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""  
  output_dim: int  
  
  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.output_dim)(x)[:, None, None, :]    


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture.
  
  Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
  """
  marginal_prob_std: Any
  channels: Tuple[int] = (32, 64, 128, 256)
  embed_dim: int = 256
  
  @nn.compact
  def __call__(self, x, t): 
    # The swish activation function
    act = nn.swish
    # Obtain the Gaussian random feature embedding for t   
    embed = act(nn.Dense(self.embed_dim)(
        GaussianFourierProjection(embed_dim=self.embed_dim)(t)))
        
    # Encoding path
    h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID',
                   use_bias=False)(x)    
    ## Incorporate information from t
    h1 += Dense(self.channels[0])(embed)
    ## Group normalization
    h1 = nn.GroupNorm(4)(h1)    
    h1 = act(h1)
    h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID',
                   use_bias=False)(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = nn.GroupNorm()(h2)        
    h2 = act(h2)
    h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID',
                   use_bias=False)(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = nn.GroupNorm()(h3)
    h3 = act(h3)
    h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID',
                   use_bias=False)(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = nn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(h4)    
    ## Skip connection from the encoding path
    h += Dense(self.channels[2])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)    
    h = nn.GroupNorm()(h)  
    h = act(h)
    h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
        jnp.concatenate([h, h1], axis=-1)
    )

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """      
  return jnp.sqrt((sigma**(2 * t) - 1.) / 2. / jnp.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return sigma**t

def score_fn(score_model, params, x, t):
  return score_model.apply(params, x, t)

def noise_removal(sample, threshold=-35.0):
  """ Post process generated melspectograms
  """
  p = np.array(sample)

  DB = librosa.amplitude_to_db(p, ref=np.max)
  DB_noise_removed = np.where(DB > threshold, DB, -80)
  return DB, DB_noise_removed

def viz(sample):
  """ Visualize the melspectrogram
  """
  sampling_rate = 16000  
  call_with_noise, call_wo_noise = noise_removal(sample)  
  
  librosa.display.specshow(call_wo_noise, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='log');
  plt.colorbar(format='%+2.0f dB');

def audio(sample, noise_threshold=-35.0):
  sampling_rate = 16000
  
  call_with_noise, call_wo_noise = noise_removal(sample, threshold=noise_threshold)  
  call_wo_noise = librosa.db_to_amplitude(call_wo_noise)
  back_audio = librosa.feature.inverse.mel_to_audio(call_wo_noise, sr=sampling_rate)
  soundfile.write('audio.wav', back_audio, samplerate=sampling_rate, subtype='FLOAT')
  return back_audio

sigma =  args.sigma
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
pmap_score_fn = jax.pmap(score_fn, static_broadcasted_argnums=(0, 1))

num_steps =  args.num_steps
signal_to_noise_ratio = args.signal_to_noise_ratio  
pc_num_steps = args.pc_num_steps
error_tolerance = args.etol

sample_batch_size = args.sample_batch_size
sampler = ode_sampler

## Load the pre-trained checkpoint from disk.
score_model = ScoreNet(marginal_prob_std_fn)
fake_input = jnp.ones((sample_batch_size, 28, 313, 1))
fake_time = jnp.ones((sample_batch_size, ))
rng = jax.random.PRNGKey(0)
params = score_model.init({'params': rng}, fake_input, fake_time)
optimizer = flax.optim.Adam().create(params)
with tf.io.gfile.GFile('cwd/ckpt.flax', 'rb') as fin:
    optimizer = from_bytes(optimizer, fin.read())

## Generate samples using the specified sampler.
rng, step_rng = jax.random.split(rng)
samples = sampler(rng,
                score_model, 
                optimizer.target,
                marginal_prob_std_fn,
                diffusion_coeff_fn, 
                sample_batch_size)

## Sample visualization.
samples = jnp.transpose(samples.reshape((-1, 28, 313, 1)), (0, 3, 1, 2))
# %matplotlib inline
sample_grid = make_grid(torch.tensor(np.asarray(samples)), nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()

viz(jnp.mean(samples[args.sample_no], 0))
ipd.Audio(audio(jnp.mean(samples[args.sample_no], 0)), rate=16000)