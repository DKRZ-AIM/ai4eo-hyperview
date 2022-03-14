#!/bin/bash -x
module use $OTHERSTAGES
module load Stages/2020
module load TensorFlow/2.5.0-Python-3.8.5


srun -N1 -p booster --account=hai_cons_ee --gres gpu:4 --pty singularity exec --bind "${PWD}:/mnt" --nv /hyperview_latest.sif python main.py -m 2 -c 1 -l 0.01000 -b 16 -w 64 --num-epochs 120 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_cons_ee --gres gpu:4 --pty singularity exec --bind "${PWD}:/mnt" --nv /hyperview_latest.sif python main.py -m 2 -c 1 -l 0.00100 -b 16 -w 64 --num-epochs 120 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_cons_ee --gres gpu:4 --pty singularity exec --bind "${PWD}:/mnt" --nv /hyperview_latest.sif python main.py -m 2 -c 2 -l 0.01000 -b 16 -w 64 --num-epochs 120 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_cons_ee --gres gpu:4 --pty singularity exec --bind "${PWD}:/mnt" --nv /hyperview_latest.sif python main.py -m 2 -c 2 -l 0.00100 -b 16 -w 64 --num-epochs 120 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
