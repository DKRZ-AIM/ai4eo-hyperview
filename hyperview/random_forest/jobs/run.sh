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

python3 rf_train.py --in-data $datadir --submission-dir $codedir/submissions --n-trials 200 --n-estimators 800 1200 --max-depth 200 500 --max-depth-none --min-samples-leaf 1 3 5 --regressors RandomForest --folds 10 

