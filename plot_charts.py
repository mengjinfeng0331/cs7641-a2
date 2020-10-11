import generate_plot,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = 'results'

#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FourPeaks-tune_GA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'mutation','pop','best_fitness','FourPeakks-GA_Tuning',OUTPUT_DIR+os.sep+'FourPeak-GA_Tuning.png')



#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FourPeaks-tune_SA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'temp','decay','best_fitness','FourPeakks-SA_Tuning',OUTPUT_DIR+os.sep+'FourPeak-SA_Tuning.png')
#
#
##file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FourPeaks-tune_MIMIC.csv'
##df= pd.read_csv(file)
##generate_plot.plot_NN_heatmap(df,'keep_pct','pop','best_fitness','FourPeakks-MIMIC_Tuning',OUTPUT_DIR+os.sep+'FourPeak-MIMIC_Tuning.png')
##
###
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FlipFlop-tune_GA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'mutation','pop','best_fitness','FlipFlop-GA_Tuning',OUTPUT_DIR+os.sep+'FlipFlop-GA_Tuning.png')
#
#
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FlipFlop-tune_SA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'temp','decay','best_fitness','FlipFlop-SA_Tuning',OUTPUT_DIR+os.sep+'FlipFlop-SA_Tuning.png')
#
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\FlipFlop-tune_MIMIC.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'keep_pct','pop','best_fitness','FlipFlop-MIMIC_Tuning',OUTPUT_DIR+os.sep+'FlipFlop-MIMIC_Tuning.png')
#
#
###
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\tsp-tune_GA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'mutation','pop','best_fitness','tsp-GA_Tuning',OUTPUT_DIR+os.sep+'tsp-GA_Tuning.png')
#
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\tsp-tune_SA.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'temp','decay','best_fitness','tsp-SA_Tuning',OUTPUT_DIR+os.sep+'tsp-SA_Tuning.png')
#
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\tsp-tune_MIMIC.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'keep_pct','pop','best_fitness','tsp-MIMIC_Tuning',OUTPUT_DIR+os.sep+'tsp-MIMIC_Tuning.png')


#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-SA-gridsearch-lr.csv'
#df= pd.read_csv(file)
#df = df[['lr','acc','mse']]
#df.set_index('lr', inplace=True)
#generate_plot.plot_NN_LR(df,'SA-LearningRate', OUTPUT_DIR+os.sep+'NN-SA-LR.png')

#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-SA-gridsearch-hyper.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'init_temp','exp_const','acc', 'NN-SA-Hyper', OUTPUT_DIR+os.sep+'NN-SA-Hyper.png')
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-RHC-gridsearch.csv'
#df= pd.read_csv(file)
#df = df[['lr','acc','mse']]
#df.set_index('lr', inplace=True)
#generate_plot.plot_NN_LR(df,'RHC-LearningRate', OUTPUT_DIR+os.sep+'NN-RHC-LR.png')

#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-GA-gridsearch-hyper.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'mutation_prob','pop_size','acc','NN-GA_gridsearch',OUTPUT_DIR+os.sep+'NN-GA_gridsearch.png')
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-GA-gridsearch-lr.csv'
#df= pd.read_csv(file)
#df = df[['lr','acc','mse']]
#df.set_index('lr', inplace=True)
#generate_plot.plot_NN_LR(df,'GA-LearningRate', OUTPUT_DIR+os.sep+'NN-GA-LR.png')

#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-grad-gridsearch-lr.csv'
#df= pd.read_csv(file,sep=',\s+', delimiter=',', encoding="utf-8", skipinitialspace=True)
#df = df[['lr','acc','mse']]
#df.set_index('lr', inplace=True)
#generate_plot.plot_NN_LR(df,'NN-Grad-LearningRate', OUTPUT_DIR+os.sep+'NN-grad-LR.png')
#
#
#file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-grad-gridsearch-hyper.csv'
#df= pd.read_csv(file)
#generate_plot.plot_NN_heatmap(df,'init_temp','exp_const','acc','NN-Grad_gridsearch',OUTPUT_DIR+os.sep+'NN-Grad_gridsearch.png')

file = r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-GA_test_fitnesscurve.csv'
file_dict = {'GA':r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-GA_test_fitnesscurve.csv',
             'SA':r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-SA_test_fitnesscurve.csv',
             'RHC':r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-RHC_test_fitnesscurve.csv',
             'Grad':r'C:\Users\victor\Desktop\OMSCS\CS7641-ML\Assignment2\results\NN-GD_test_fitnesscurve.csv'
        }
def read_file(file):
    array=[]
    with open(file,'r') as f:
        lines = f.read().strip().splitlines()
    
    for line in lines:
#        print(line)
        array.append([float(i) for i in line.split(',') if i.strip() !=''])
    return array

def process_fitnessCurve(file):
    array = read_file(file)
    array  = generate_plot.fill_array(array)
    array_mean = np.mean(array, axis=0)
    return array_mean

plt.figure(figsize=(10,10))
for alg, file in file_dict.items():
    array_mean = process_fitnessCurve(file)
    if alg !='Grad':
        plt.plot([-i for i in array_mean],label=alg)
    else:
        plt.plot(array_mean,label=alg)
        
plt.legend()
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.ylim(-3,0)
plt.title('NN-Fitness')
plt.savefig(OUTPUT_DIR+os.sep+'NN-fitness_curve.png')
plt.show()