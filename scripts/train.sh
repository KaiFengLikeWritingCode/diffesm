#!/bin/sh
accelerate launch --num_processes=3 src/train.py
