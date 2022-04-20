#!/bin/bash -x

srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 1 -l 0.1000 -b 16 -w 90  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 1 -l 0.0100 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 1 -l 0.0010 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 1 -l 0.0001 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 2 -l 0.1000 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 2 -l 0.0100 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 2 -l 0.0010 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 2 -l 0.0001 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 3 -l 0.1000 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 3 -l 0.0100 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 3 -l 0.0010 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 1 -c 3 -l 0.0001 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 1 -l 0.1000 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 1 -l 0.0100 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 1 -l 0.0010 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 1 -l 0.0001 -b 16 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 2 -l 0.1000 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 2 -l 0.0100 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 2 -l 0.0010 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &
srun -N1 -p booster --account=hai_ai_4_eo --gres gpu:4 --time=5:00:00 --pty singularity exec --bind "${PWD}:/mnt" --nv ../hyperview_latest.sif python main_gan.py -m 7 -c 2 -l 0.0001 -b 8 -w 64  --num-epochs 69 --train-dir 'train_data/train_data/' --label-dir 'train_data/train_gt.csv' --eval-dir 'test_data/' --out-dir 'modeldir/' &