""" Script for diagnostic figures generation.
"""
import cv2
import numpy as np
import os
import shutil
import torch

from tidr_icip21.attacks.pgd import PGD
from tidr_icip21.attacks.tidr import TIDR
from tidr_icip21.models.vgg16 import VGG16
from tidr_icip21.models.resnet152 import RESNET152
from tidr_icip21.utils import img_utils, imagenet_utils

import pdb


def task1():
  output_folder = "temp_diagnostic"
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  os.mkdir(output_folder)

  imgnet_label = imagenet_utils.load_imagenet_label_dict()
  # Use vgg16 as surrogate
  vgg16 = VGG16()
  pgd_attack = PGD(vgg16)
  tidr_attack = TIDR(vgg16, dr_weight=10, steps=200)
  img_ori_var = img_utils.load_img("sample_images/orange_cat.png").cuda()
  pred_ori = torch.argmax(vgg16(img_ori_var)[1], axis=1)
  print("Generating AEs. attack mtds: PGD, TIDR, surrogate: vgg16")
  img_adv_tidr_var = tidr_attack(img_ori_var, pred_ori, internal=[12])
  img_adv_pgd_var = pgd_attack(img_ori_var, pred_ori)
  # Use resnet152 as victim model
  resnet152 = RESNET152()
  internals = [4]
  print("Evaluating AEs. victim model: resnet152")
  layers_pgd, pred_pgd = resnet152(img_adv_pgd_var.cuda(), internals)
  layers_tidr, pred_tidr = resnet152(img_adv_tidr_var.cuda(), internals)
  # Show predictions
  logits_pgd_np = pred_pgd[0].detach().cpu().numpy()
  show_predictions(logits_pgd_np, imgnet_label, "pgd")
  logits_tidr_np = pred_tidr[0].detach().cpu().numpy()
  show_predictions(logits_tidr_np, imgnet_label, "tidr")
  print("Saving figures.")
  save_features(layers_pgd, output_folder, "pgd")
  save_features(layers_tidr, output_folder, "tidr")


def show_predictions(logits: np.ndarray,
                     label_dict: dict, suffix: str) -> None:
  softmax_pgd = np.exp(logits)/sum(np.exp(logits))
  top_5_idx = softmax_pgd.argsort()[-5:][::-1]
  print("{}_prediction: ".format(suffix))
  for idx in top_5_idx:
    print("{0}: {1:.04f}".format(label_dict[idx], softmax_pgd[idx]))
  return


def save_features(layers: list, output_folder: str, suffix: str) -> None:
  for layer_idx, layer_var in enumerate(layers):
    layer = layer_var.detach().cpu().numpy()[0]
    layer_max = layer.max()
    for channel_idx, channel in enumerate(layer):
      channel_norm = ((channel / channel.max()) * 255.).astype(np.uint8)
      channel_img = cv2.applyColorMap(channel_norm, cv2.COLORMAP_JET)
      channel_img = cv2.resize(channel_img, (224, 224))
      filepath = os.path.join(
          output_folder,
          "l{0:02d}_c{1:03d}_{2}.png".format(layer_idx, channel_idx, suffix))
      cv2.imwrite(filepath, channel_img)


if __name__ == "__main__":
  task1()