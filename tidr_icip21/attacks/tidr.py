""" Script for TI-DR attack.
"""
import copy
import numpy as np
from torch.autograd import Variable
import torch
import scipy.stats as st
from scipy import ndimage

import pdb


class TIDR(object):
  def __init__(self, model, 
                decay_factor=1, prob=0.5,
                epsilon=16./255, steps=40, step_size=2./255, 
                image_resize=330, dr_weight=0.1,
                random_start=False):
    """
    Paper link: https://arxiv.org/pdf/1803.06978.pdf
    """

    self._epsilon = epsilon
    self._steps = steps
    self._step_size = step_size
    self._rand = random_start
    self._model = copy.deepcopy(model)
    self._loss_fn = torch.nn.CrossEntropyLoss().cuda()
    self._decay_factor = decay_factor
    self._prob = prob
    self._image_resize = image_resize
    self._dr_weight = dr_weight

    kernel = self.gkern(15, 3).astype(np.float32)
    self._stack_kernel = np.stack([kernel, kernel, kernel])

  def __call__(self, X_nat: torch.Tensor,
        y: torch.Tensor, internal: tuple=()) -> torch.Tensor:
    """
    Given examples (X_nat, y), returns adversarial
    examples within epsilon of X_nat in l_infinity norm.
    """
    X_nat_np = X_nat.detach().cpu().numpy()
    for p in self._model.parameters():
      p.requires_grad = False
    
    self._model.eval()
    if self._rand:
      X = X_nat_np + np.random.uniform(-self._epsilon,
                                        self._epsilon,
                                        X_nat_np.shape).astype('float32')
    else:
      X = np.copy(X_nat_np)
    
    momentum = 0
    for _ in range(self._steps):
      X_var = Variable(torch.from_numpy(X).cuda(),
                       requires_grad=True, volatile=False)
      y_var = y.cuda()

      rnd = np.random.rand()
      if rnd < self._prob:
        transformer = _tranform_resize_padding(
            X.shape[-2], X.shape[-1], self._image_resize,
            resize_back=True)
        X_trans_var = transformer(X_var)
      else:
        X_trans_var = X_var

      layers, scores = self._model(X_trans_var, internal=internal)

      dr_loss = 0.0
      for layer_idx, target_layer in enumerate(layers):
        temp_dr_loss = -1 * target_layer.var()
        dr_loss += temp_dr_loss
      
      loss = self._loss_fn(scores, y_var) + self._dr_weight * dr_loss
      self._model.zero_grad()
      loss.backward()
      grad = X_var.grad.data.cpu().numpy()
      grad = self.depthwise_conv2d(grad, self._stack_kernel)
      X_var.grad.zero_()
      velocity = grad / np.mean(np.absolute(grad), axis=(1, 2, 3))
      momentum = self._decay_factor * momentum + velocity

      X += self._step_size * np.sign(momentum)
      X = np.clip(X, X_nat_np - self._epsilon, X_nat_np + self._epsilon)
      X = np.clip(X, 0, 1) # ensure valid pixel range
    return torch.from_numpy(X)

  @staticmethod
  def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

  @staticmethod
  def depthwise_conv2d(in1, stack_kernel):
    ret = []
    for temp_in in in1:
      # numpy convolve operates differently to CNN conv, 
      # however they are the same when keernel is symetric.
      temp_out = ndimage.convolve(temp_in, stack_kernel, mode='constant')
      ret.append(temp_out)
    ret = np.array(ret)
    return ret


class _tranform_resize_padding(torch.nn.Module):
  def __init__(self, image_h, image_w, image_resize, resize_back=False):
    super(_tranform_resize_padding, self).__init__()
    self.shape = [image_h, image_w]
    self._image_resize = image_resize
    self.resize_back = resize_back

  def __call__(self, input_tensor):
    assert self.shape[0] < self._image_resize \
      and self.shape[1] < self._image_resize
    rnd = np.random.randint(self.shape[1], self._image_resize)
    input_upsample = torch.nn.functional.interpolate(
        input_tensor, size=(rnd, rnd), mode='nearest')
    h_rem = self._image_resize - rnd
    w_rem = self._image_resize - rnd
    pad_top = np.random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padder = torch.nn.ConstantPad2d(
        (pad_left, pad_right, pad_top, pad_bottom), 0.0)
    input_padded = padder(input_upsample)
    if self.resize_back:
      input_padded_resize = torch.nn.functional.interpolate(
          input_padded, size=self.shape, mode='nearest')
      return input_padded_resize
    else:
      return input_padded