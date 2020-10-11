import pandas as pd
import sys
import six
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose

import matplotlib.pyplot as plt
import numpy as np
import itertools
import load_data,os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.neural_network import MLPClassifier
import pickle
import generate_plot
from timeit import default_timer as timer
###################################################################
OUTPUT = 'results'
creditcard_file = 'data/creditcard_undersample.csv'
lr_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5,0.8, 1]
PROBLEM='RHC'
SEED_LIST = range(20)

###################################################################

X, y=  load_data.load_creditcard_data(creditcard_file)
MAX_ITER = 1001
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

def write2file(file, data, mode='a'):
    with open(file, mode) as f:
        f.write(data)

def RHC_Tune(lr=0.1,max_iters=1000):
    test_accu_list = []
    test_mse_list = []
    run_time_list = []
    for seed in SEED_LIST:
        
        start = timer()
        model = mlrose.NeuralNetwork(hidden_nodes=[20, 5], 
                                     activation='relu',
                                     algorithm='random_hill_climb',
                                     is_classifier=True, 
                                     early_stopping=True,
                                     max_iters = max_iters,
                                     random_state=seed,
                                     learning_rate=lr,
                                     restarts=10
                                     )
        model.fit(X_train, y_train)
        
        end = timer()
        
        ##
        y_pred = model.predict(X_test)
        nn_acc = accuracy_score(y_test, y_pred)
        test_accu_list.append(nn_acc)
    
        elapsed_time = end - start
        run_time_list.append(elapsed_time)
        mse_test = mean_squared_error(y_test, y_pred)
        test_mse_list.append(mse_test)
        
    return np.mean(test_accu_list), np.mean(test_mse_list),np.mean(run_time_list)

def gridsearch():
    out_file = OUTPUT + os.sep + 'NN-{}-gridsearch.csv'.format(PROBLEM)
    write2file(out_file, 'lr,acc,mse,runtime\n','w')
    for lr in lr_list:
        print('running lr:{}'.format(lr))
        acc, mse, runtime = RHC_Tune(lr)
        write2file(out_file, '{},{},{},{}\n'.format(lr,acc,mse,runtime),'a')
    
    df = pd.read_csv(out_file,index_col=0)
    df = df[['acc','mse']]
    generate_plot.plot_NN_LR(df,title='NN-RHC Learning rate',output_pic=OUTPUT+os.sep+'NN-RHC-LR.PNG')

def RHC_Test(lr=0.5):
    OUTFILE_curve=  OUTPUT + os.sep + 'NN-{}_test_fitnesscurve.csv'.format(PROBLEM)
    OUTFILE=  OUTPUT + os.sep + 'NN-{}_test.csv'.format(PROBLEM)
    write2file(OUTFILE, 'mse_train,mse_test,nn_acc_train, nn_acc,elapsed_time,loss,\n','w')

    with open(OUTFILE_curve,'w' ) as f:
        pass
    
    for seed in SEED_LIST:        
        start = timer()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = mlrose.NeuralNetwork(hidden_nodes=[20, 5], 
                                     activation='relu',
                                     algorithm='random_hill_climb',
                                     is_classifier=True, 
                                     early_stopping=False,
                                     random_state=seed,
                                     learning_rate=lr,
                                     max_iters = 1000,
                                     restarts=10,
                                      curve=True,
                                     )
        model.fit(X_train, y_train)
        end = timer()
        
        fitness_curve  = model.fitness_curve 
        print('seed #',seed)
#        print('fitness_curve:',fitness_curve)
        
        loss  = model.loss
        
        y_pred = model.predict(X_test)
        nn_acc = accuracy_score(y_test, y_pred)
        y_pred_train = model.predict(X_train)
        nn_acc_train = accuracy_score(y_train, y_pred_train)
    
        elapsed_time = end - start
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred)
        
        write2file(OUTFILE, '{},{},{},{},{},{}\n'.format(mse_train,mse_test,nn_acc_train, nn_acc,elapsed_time,loss),'a')

        with open(OUTFILE_curve, "ab") as f:
            f.write(b"\n")
            np.savetxt(f, fitness_curve, delimiter=',', fmt='%.1f',newline=", ")
                    

if __name__ == '__main__':
    if sys.argv[1] == 'gridsearch':
        gridsearch()
    elif  sys.argv[1] == 'test':
        RHC_Test()
    