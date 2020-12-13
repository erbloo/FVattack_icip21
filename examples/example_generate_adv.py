""" Script for generating AE examples.
"""
import argparse
import importlib
import os
import sys
import torch
from tqdm import tqdm

from tidr_icip21.utils import img_utils, imagenet_utils

import pdb


def generate_adv_example(args):
  imgnet_label = imagenet_utils.load_imagenet_label_dict()

  source_lib = importlib.import_module(
      "tidr_icip21.models." + args.source_model)
  source_model_class = getattr(source_lib, args.source_model.upper())
  source_model = source_model_class(is_normalize=True)

  attack_lib = importlib.import_module(
      os.path.join("tidr_icip21.attacks." + args.attack_method))
  attacker_class = getattr(attack_lib, args.attack_method.upper())
  attacker = attacker_class(source_model)

  img_names = os.listdir(args.input_dir)
  for img_name in tqdm(img_names):
    img_name_noext = os.path.splitext(img_name)[0]
    img_path = os.path.join(args.input_dir, img_name)
    img_ori_var = img_utils.load_img(img_path).cuda()
    pred_ori = torch.argmax(source_model(img_ori_var)[1], axis=1)
    img_adv_var = attacker(img_ori_var, pred_ori, internal=[12,14])
    pred_adv = torch.argmax(source_model(img_adv_var.cuda())[1], axis=1)
    
    # Visualization for debuging. 
    print("Ori: ", img_name, " , ", pred_ori, ":", imgnet_label[pred_ori.cpu().numpy()[0]])
    print("Adv: ", img_name, " , ", pred_adv, ":", imgnet_label[pred_adv.cpu().numpy()[0]])
    output_img = img_utils.save_img(img_adv_var, os.path.join(args.output_dir, img_name_noext + ".png"))
  return


def parse_args(args):
  parser = argparse.ArgumentParser(description="PyTorch AE generation.")
  parser.add_argument('--source_model', choices=["vgg16"], default="vgg16", type=str)
  parser.add_argument('--attack_method', choices=["tidr"], default="tidr", type=str)
  parser.add_argument('--input_dir', default="sample_images/", type=str)
  parser.add_argument('--output_dir', default="outputs/", type=str)
  return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    generate_adv_example(args)

if __name__ == "__main__":
   main()