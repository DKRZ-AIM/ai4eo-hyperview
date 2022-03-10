#!/bin/bash -x

module load Stages/2020
module load TensorFlow/2.5.0-Python-3.8.5
srun --exclusive --nodes=1 --gres=gpu:1  --time=00:45:00  --cpu-bind=map_cpu:0 --account=hai_cons_ee python main.py -m 0 -l 0.000010 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/'

