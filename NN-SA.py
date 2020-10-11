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
import pickle,generate_plot
from timeit import default_timer as timer
###################################################################
OUTPUT = 'results'
creditcard_file = 'data/creditcard_undersample.csv'
lr_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5,0.8,1,2,5]
exp_const_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
init_temp_list = [1, 10, 50, 100, 200, 300, 400]
PROBLEM='SA'
SEED_LIST = range(5)

###################################################################

X, y=  load_data.load_creditcard_data(creditcard_file)
MAX_ITER = 1001
def write2file(file, data, mode='a'):
    with open(file, mode) as f:
        f.write(data)

def SA_Tune(lr=0.1,init_temp=100, exp_const=0.05):
    test_accu_list = []
    test_mse_list = []
    run_time_list = []
    for seed in SEED_LIST:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        start = timer()
        model = mlrose.NeuralNetwork(hidden_nodes=[20, 5], 
                                     activation='relu',
                                     algorithm='simulated_annealing',
                                     is_classifier=True, 
                                     early_stopping=True,
                                     random_state=seed,
                                     max_iters = 1000,
                                      learning_rate=lr,
                                      schedule=mlrose.ExpDecay(
                                      init_temp=init_temp,
                                      min_temp=0.001,
                                      exp_const=exp_const,
                                      ))
        model.fit(X_train, y_train)
        end = timer()
        
        y_pred = model.predict(X_test)
        nn_acc = accuracy_score(y_test, y_pred)
        test_accu_list.append(nn_acc)
    
        elapsed_time = end - start
        run_time_list.append(elapsed_time)
        mse_test = mean_squared_error(y_test, y_pred)
        test_mse_list.append(mse_test)
#        print('test_accu_list:',test_accu_list)
    return np.mean(test_accu_list), np.mean(test_mse_list),np.mean(run_time_list)

def gridsearch_lr():
    out_file = OUTPUT + os.sep + 'NN-{}-gridsearch-lr.csv'.format(PROBLEM)
    write2file(out_file, 'lr,init_temp, exp_const, acc,mse,runtime\n','w')
    init_temp = 100
    exp_const =0.05
    for lr in lr_list:
        acc, mse, runtime = SA_Tune(lr,init_temp=init_temp, exp_const=exp_const)
        print('running lr:{}. init_temp:{}, exp_const:{},acc:{}, mse:{}, runtime:{}'.format(lr, init_temp, exp_const,acc, mse, runtime))
        write2file(out_file, '{},{},{},{},{},{}\n'.format(lr,init_temp,exp_const, acc,mse,runtime),'a')
        
    df = pd.read_csv(out_file,index_col=0)
    df = df[['acc','mse']]
    generate_plot.plot_NN_LR(df,title='NN-{} Learning rate'.format(PROBLEM),output_pic=OUTPUT+os.sep+'NN-{}-LR.PNG'.format(PROBLEM))
        
def gridsearch_hyper():
    out_file = OUTPUT + os.sep + 'NN-{}-gridsearch-hyper.csv'.format(PROBLEM)
    write2file(out_file, 'lr,init_temp,exp_const,acc,mse,runtime\n','w')
    lr=1
    for init_temp  in init_temp_list:
        for exp_const in exp_const_list:
            acc, mse, runtime = SA_Tune(lr,init_temp=init_temp, exp_const=exp_const)
            print('running lr:{}. init_temp:{}, exp_const:{},acc:{}, mse:{}, runtime:{}'.format(lr, init_temp, exp_const,acc, mse, runtime))
            write2file(out_file, '{},{},{},{},{},{}\n'.format(lr,init_temp,exp_const, acc,mse,runtime),'a')
    
    df = pd.read_csv(out_file,sep=',\s+', delimiter=',', encoding="utf-8", skipinitialspace=True)
    generate_plot.plot_NN_heatmap(df, x='init_temp',y='exp_const',z='acc', 
                                  title='{}-HyperParameterHeatMap'.format(PROBLEM),
                                  output_pic=OUTPUT+os.sep+'{}-hyper.PNG'.format(PROBLEM))
    
def SA_Test(lr=0.5,init_temp=50, exp_const=0.1):
    OUTFILE_curve=  OUTPUT + os.sep + 'NN-{}_test_fitnesscurve.csv'.format(PROBLEM)
    OUTFILE =  OUTPUT + os.sep + 'NN-{}_test.csv'.format(PROBLEM)
    write2file(OUTFILE, 'mse_train,mse_test,nn_acc_train, nn_acc,elapsed_time,loss\n','w')

    with open(OUTFILE_curve,'w' ) as f:
        pass
    
    for seed in range(20):        
        start = timer()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = mlrose.NeuralNetwork(hidden_nodes=[20, 5], 
                                     activation='relu',
                                     algorithm='simulated_annealing',
                                     is_classifier=True, 
                                     early_stopping=False,
                                     random_state=seed,
                                     max_iters = 1000,
                                      curve=True,
                                      schedule=mlrose.ExpDecay(
                                      init_temp=init_temp,
                                      min_temp=0.001,
                                      exp_const=exp_const,
                                      ))
        model.fit(X_train, y_train)
        end = timer()
        
        fitness_curve  = model.fitness_curve 
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
    if sys.argv[1] == 'gridsearch-lr':
        gridsearch_lr()
    elif sys.argv[1] == 'gridsearch-hyper':
        gridsearch_hyper()        
    else:
        SA_Test()
    