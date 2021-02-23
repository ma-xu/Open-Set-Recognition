#!/bin/bash

python3 cifar10.py --train_class_num 5 --test_class_num 10 --temperature 1 --alpha 10.0 --stage1_resume ./checkpoints/cifar10/5_10_ResNet18_dim128_T1_alpha10.0_p2/stage_1_last_model.pth

sudo poweroff




#python3 cifar10.py --train_class_num 7 --test_class_num 10
#python3 cifar10.py --train_class_num 5 --test_class_num 10
#python3 cifar100.py --train_class_num 50 --test_class_num 100
#python3 cifar100.py --train_class_num 80 --test_class_num 100
#
#python3 cifar10.py --train_class_num 7 --test_class_num 10 --temperature 16
#python3 cifar10.py --train_class_num 5 --test_class_num 10 --temperature 16
#python3 cifar100.py --train_class_num 50 --test_class_num 100 --temperature 16
#python3 cifar100.py --train_class_num 80 --test_class_num 100 --temperature 16
#
#python3 cifar10.py --train_class_num 7 --test_class_num 10 --temperature 32
#python3 cifar10.py --train_class_num 5 --test_class_num 10 --temperature 32
#python3 cifar100.py --train_class_num 50 --test_class_num 100 --temperature 32
#python3 cifar100.py --train_class_num 80 --test_class_num 100 --temperature 32
#
#
