#!/bin/bash -x

module load Stages/2020
module load TensorFlow/2.5.0-Python-3.8.5
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 0 -l 0.0100 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 0 -l 0.0010 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 0 -l 0.0001 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 1 -l 0.0100 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 1 -l 0.0010 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 1 -l 0.0001 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 2 -l 0.0100 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 2 -l 0.0010 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun --exclusive --nodes=1 --gres=gpu:4  --time=11:45:00 --account=hai_cons_ee python main.py -m 2 -l 0.0001 -b 32 --num-epochs 90 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &

