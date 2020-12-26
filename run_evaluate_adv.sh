#!/bin/bash

BENIGN_DIR=/home/yilan/yilan/workspace/datasets/icip21_images/imagenet_5000
TARGET_MODEL=densenet201

for ADV_NAME in inception_v3_tidr_decay_factor_1p0000_dr_weight_0p1000_epsilon_0p0627_image_resize_330_prob_0p5000_random_start_False_step_size_0p0078_steps_100 \
    resnet152_tidr_decay_factor_1p0000_dr_weight_0p1000_epsilon_0p0627_image_resize_330_prob_0p5000_random_start_False_step_size_0p0078_steps_100 \
    vgg16_tidr_decay_factor_1p0000_dr_weight_0p1000_epsilon_0p0627_image_resize_330_prob_0p5000_random_start_False_step_size_0p0078_steps_100 \
    inception_v3_mifgsm_decay_factor_1.0000_epsilon_0.0627_step_size_0.0078_step_size_0.0078 \
    resnet152_mifgsm_decay_factor_1.0000_epsilon_0.0627_step_size_0.0078_step_size_0.0078 \
    vgg16_mifgsm_decay_factor_1.0000_epsilon_0.0627_step_size_0.0078_step_size_0.0078 \
    inception_v3_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    resnet152_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    vgg16_pgd_a_0.0078_epsilon_0.0627_epsilon_0.0627 \
    tidim_inception_v3_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    tidim_resnet152_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_ \
    tidim_vgg16_layerAt_0_eps_16_stepsize_4.0_steps_100_lossmtd_
do
  ADV_DIR="/home/yilan/yilan/workspace/datasets/icip21_images/${ADV_NAME}"
  python3 examples/example_evaluate.py --benign_dir  $BENIGN_DIR --adv_dir $ADV_DIR --target_model $TARGET_MODEL
done
