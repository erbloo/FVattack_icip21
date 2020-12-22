#!/bin/bash

BENIGN_DIR=/home/yilan/yilan/workspace/datasets/icip21_images/imagenet_5000
ADV_DIR=
TARGET_MODEL=vgg16

for ADV_NAME in inception_v3_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    resnet152_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    vgg16_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    vgg16_tidr_decay_factor_1.0000_dr_weight_0.1000_epsilon_0.0627_epsilon_0.0627_prob_0.5000_prob_0.5000_step_size_0.0078_step_size_0.0078 \
    resnet152_tidr_decay_factor_1.0000_dr_weight_0.1000_epsilon_0.0627_epsilon_0.0627_prob_0.5000_prob_0.5000_step_size_0.0078_step_size_0.0078 \
    inception_v3_tidr_decay_factor_1.0000_dr_weight_0.1000_epsilon_0.0627_epsilon_0.0627_prob_0.5000_prob_0.5000_step_size_0.0078_step_size_0.0078 \
    dim_inception_v3_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    dim_resnet152_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    dim_vgg16_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    tidim_inception_v3_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    tidim_resnet152_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    tidim_vgg16_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_
do
  ADV_DIR="/home/yilan/yilan/workspace/datasets/icip21_images/${ADV_NAME}"
  python3 examples/example_evaluate.py --benign_dir  $BENIGN_DIR --adv_dir $ADV_DIR --target_model $TARGET_MODEL
done
