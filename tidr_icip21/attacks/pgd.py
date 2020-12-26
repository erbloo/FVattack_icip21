import copy
import numpy as np
from torch.autograd import Variable
import torch

import pdb

class PGD(object):
  def __init__(self, model, epsilon=16/255., k=40, a=2/255., 
               random_start=False, attack_config=None):
    """
    Attack parameter initialization. The attack performs k steps of
    size a, while always staying within epsilon from the initial
    point.
    https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.model = copy.deepcopy(model)
    self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

  def __call__(self, X_nat: torch.Tensor,
        y: torch.Tensor) -> torch.Tensor:
    """
    Given examples (X_nat, y), returns adversarial
    examples within epsilon of X_nat in l_infinity norm.
    """
    X_nat_np = X_nat.detach().cpu().numpy()
    for p in self.model.parameters():
      p.requires_grad = False
    
    self.model.eval()
    if self.rand:
      X = X_nat_np + np.random.uniform(-self.epsilon, self.epsilon,
          X_nat_np.shape).astype('float32')
    else:
      X = np.copy(X_nat_np)
    
    for _ in range(self.k):
      X_var = Variable(torch.from_numpy(X).cuda(),
                       requires_grad=True, volatile=False)
      y_var = y.cuda()
      _, scores = self.model(X_var)
      loss = self.loss_fn(scores, y_var)
      self.model.zero_grad()
      loss.backward()
      grad = X_var.grad.data.cpu().numpy()
      X_var.grad.zero_()

      X += self.a * np.sign(grad)
      X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
      X = np.clip(X, 0, 1) # ensure valid pixel range
    return torch.from_numpy(X)