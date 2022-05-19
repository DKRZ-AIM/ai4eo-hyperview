#!/bin/bash
#SBATCH --job-name=xgb-0-cl
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=99:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=k20200
#SBATCH --output=rf_out.o%j
#SBATCH --error=rf_err.e%j 
#SBATCH --nodelist vader2

hostname

module load /sw/spack-amd/spack/modules/linux-centos8-zen2/singularity/3.7.0-gcc-10.2.0


# write a run script
cat > run.sh << 'EOF'
echo 'HELLO BOX'
codedir=/work/frauke/ai4eo-hyperview/hyperview/random_forest
datadir=/work/shared_data/2022-ai4eo_hyperview
conda init
source ~/.bashrc
conda activate ai4eo_hyper
echo "conda env activated"

echo $codedir
cd $codedir

PYTHONPATH=$PYTHONPATH:"$codedir"
export PYTHONPATH

python3 rf_train.py --in-data $datadir --submission-dir $codedir/submissions --n-trials 500 --n-estimators 500 1200 --max-depth 200 500 --max-depth-none --min-samples-leaf 1 5 10 --regressors XGB --col-ix 0 --folds 5 --save-eval --eval-dir $codedir/evaluation --save-model --model-dir $codedir/models --eta 0.001 0.1 0.3 0.5 --min-child_weight 1 5 10 50 70 --objectice cubic 

EOF


# execute run script with singularity
singularity exec --bind /mnt/lustre02/work/ka1176/:/work /mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/images/ai4eo-hyperview_rf.sif /bin/bash /work/frauke/ai4eo-hyperview/hyperview/random_forest/jobs/run.sh
