#/bin/bash

python NN-SA.py gridsearch-lr &
python NN-SA.py gridsearch-hyper &
python NN-GA.py gridsearch-lr &
python NN-GA.py gridsearch-hyper &
python NN-RHC.py gridsearch &
python NN-grad.py gridsearch &
