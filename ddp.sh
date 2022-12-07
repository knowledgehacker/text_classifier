#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 -H localhost:2 python3 ddp_train.py
