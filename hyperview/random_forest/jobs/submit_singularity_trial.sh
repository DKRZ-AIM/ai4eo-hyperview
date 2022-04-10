#!/bin/bash
#SBATCH --job-name=rf
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=k20200
#SBATCH --output=rf_out.o%j
#SBATCH --error=rf_err.e%j

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

python3 rf_train.py --in-data $datadir --submission-dir $codedir/submissions --n-trials 200

EOF


# execute run script with singularity
singularity exec --bind /mnt/lustre02/work/ka1176/:/work /mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/images/ai4eo-hyperview_rf.sif /bin/bash /work/frauke/ai4eo-hyperview/hyperview/random_forest/jobs/run.sh
