#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 0
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 0.01
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 0.1
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 1
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 3
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 5
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 10
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 50
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py --gamma 100


