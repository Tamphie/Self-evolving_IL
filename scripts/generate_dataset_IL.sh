#!/bin/bash

python3 rlbench/dataset_generator.py --save_path ./data --tasks open_door --processes 1 --episodes_per_task 1 --variations 1
