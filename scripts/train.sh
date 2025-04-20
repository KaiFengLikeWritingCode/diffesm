#!/bin/sh
accelerate launch --num_processes=4 src/train.py
# accelerate launch --num_processes=4 --mixed_precision fp16 src/train.py