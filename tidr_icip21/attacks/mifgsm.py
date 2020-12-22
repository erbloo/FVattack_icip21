import copy
import numpy as np
from torch.autograd import Variable
import torch

import pdb

class MIFGSM(object):
  def __init__(self, model, decay_factor=1.0, epsilon=16./255, steps=40,
               step_size=2./255, random_start=False):
    """
    The Momentum Iterative Fast Gradient Sign Method (Dong et al. 2017).
    This method won the first places in NIPS 2017 Non-targeted Adversarial
    Attacks and Targeted Adversarial Attacks. The original paper used
    hard labels for this attack; no label smoothing. inf norm.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    self._epsilon = epsilon
    self._steps = steps
    self._step_size = step_size
    self._rand = random_start
    self._model = copy.deepcopy(model)
    self._loss_fn = torch.nn.CrossEntropyLoss().cuda()
    self._decay_factor = decay_factor

  def __call__(self, X_nat: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    """
    Given examples (X_nat, y), returns adversarial
    examples within epsilon of X_nat in l_infinity norm.
    """
    X_nat_np = X_nat.detach().cpu().numpy()
    for p in self._model.parameters():
      p.requires_grad = False
    
    self._model.eval()
    if self._rand:
      X = X_nat_np + np.random.uniform(
          -self._epsilon, self._epsilon, X_nat_np.shape).astype('float32')
    else:
      X = np.copy(X_nat_np)
    
    momentum = 0
    for _ in range(self._steps):
      X_var = Variable(
          torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
      y_var = y.cuda()
      _, scores = self._model(X_var)
      
      loss = self._loss_fn(scores, y_var)
      self._model.zero_grad()
      loss.backward()
      grad = X_var.grad.data.cpu().numpy()
      X_var.grad.zero_()
      velocity = grad / np.sum(np.absolute(grad))
      momentum = self._decay_factor * momentum + velocity

      X += self._step_size * np.sign(momentum)
      X = np.clip(X, X_nat_np - self._epsilon, X_nat_np + self._epsilon)
      X = np.clip(X, 0, 1)
    return torch.from_numpy(X)