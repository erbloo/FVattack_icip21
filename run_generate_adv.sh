#!/bin/bash

INPUT_DIR=/home/yilan/yilan/workspace/datasets/icip21_images/imagenet_5000
OUTPUT_DIR=/home/yilan/yilan/workspace/datasets/icip21_images
ATTACK_METHOD=tidr

for SOURCE_MODEL in vgg16 resnet152 inception_v3
do
  python3 examples/example_generate_adv.py --input_dir  $INPUT_DIR --output_dir $OUTPUT_DIR --source_model $SOURCE_MODEL --attack_method $ATTACK_METHOD
done
