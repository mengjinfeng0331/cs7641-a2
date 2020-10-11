import sys
import six
sys.modules['sklearn.externals.six'] = six
import csv
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import os,random
import numpy as np
import algorithm_hiive as algorithm
import pandas as pd
import generate_plot
import pickle
###################################################
PROBLEM = 'tsp'
SEED_LIST = range(10)
OUTPUT_DIR='results'
RHC_MAX_ATTEMPTS = 3000
RHC_MAX_ITER = 3000

MAX_ATTEMPTS = 200
MAX_ITER = 3000

SA_temp= 100
SA_decay = 0.1

GA_pop_size = 600
GA_mutation = 0.3

MIMIC_keep_pct = 0.1
MIMIC_pop_size = 600

######################################################3
TSP_num_points = 20

dist_list = []
## create distance function
for x in range(TSP_num_points):
    for y in range(x, TSP_num_points):
        dist_list.append((x, y, random.uniform(0.0, 1.0)))
## define fitness and problem
fitness_dists = mlrose.TravellingSales(distances=dist_list)
problem = mlrose.TSPOpt(length=TSP_num_points, fitness_fn=fitness_dists, maximize=False)
problem.set_mimic_fast_mode(True)

## initialize dataframe
df = pd.DataFrame()

## RHC
RHC_curve_list = []
best = 0
best_RHC_curve= []
duration_list = []

for seed in SEED_LIST:
    best_state, best_fitness, duration, curve = algorithm.run_RHC(problem=problem, 
                                                                  max_attempts=RHC_MAX_ITER, 
                                                                  max_iters=RHC_MAX_ITER, 
                                                                  restarts=10,
                                                                  random_seed= seed
                                                                  )
    curve= [-i for i in curve]  
    RHC_curve_list.append(curve)
    duration_list.append(duration/len(curve))
    
RHC_curve_list = generate_plot.fill_array(RHC_curve_list)
rhc_mean, rhc_std =  generate_plot.get_mean_std(RHC_curve_list)
df.loc['RHC','best_fitness'] = np.max(rhc_mean)
df.loc['RHC','duration'] = round(np.mean(duration_list),4)

### SA
SA_curves_list = []
duration_list = []
for seed in SEED_LIST:
    _, best_fitness, duration, curve = algorithm.run_SA(problem=problem, 
                                                        max_attempts=MAX_ATTEMPTS, 
                                                        max_iters=MAX_ITER,
                                                        init_temp=SA_temp,
                                                        decay= SA_decay,
                                                        decay_method = 'Exp')
    curve= [-i for i in curve]  
    SA_curves_list.append(curve)
    duration_list.append(duration/len(curve))
    
SA_curves_list = generate_plot.fill_array(SA_curves_list)
sa_mean, sa_std =  generate_plot.get_mean_std(SA_curves_list)
df.loc['SA','best_fitness'] = np.max(sa_mean)
df.loc['SA','duration'] = round(np.mean(duration_list),4) 

### GA
GA_curves_list = []
duration_list = []
for seed in SEED_LIST:
    _, best_fitness, duration, curve = algorithm.run_GA(problem=problem, 
                                                        max_attempts=MAX_ATTEMPTS, 
                                                        max_iters=MAX_ITER,
                                                        mutation_prob=GA_mutation, 
                                                        pop_size=GA_pop_size,
                                                        random_seed=seed)
    curve= [-i for i in curve]  
    GA_curves_list.append(curve)
    duration_list.append(duration/len(curve))
    
GA_curves_list = generate_plot.fill_array(GA_curves_list)
ga_mean, ga_std =  generate_plot.get_mean_std(GA_curves_list)
df.loc['GA','best_fitness'] = np.max(ga_mean)
df.loc['GA','duration'] = round(np.mean(duration_list),4) 

### MIMIC
MIMIC_curves_list = []
duration_list = []
for seed in SEED_LIST:
    _, best_fitness, duration, curve = algorithm.run_MIMIC(problem=problem, 
                                                           max_attempts=MAX_ATTEMPTS, 
                                                           max_iters=MAX_ITER,
                                                           keep_pct=MIMIC_keep_pct, 
                                                           pop_size=MIMIC_pop_size,
                                                           random_seed=seed)
    curve= [-i for i in curve]  
    MIMIC_curves_list.append(curve)
    duration_list.append(duration/len(curve))
    
MIMIC_curves_list = generate_plot.fill_array(MIMIC_curves_list)
mimic_mean, mimic_std =  generate_plot.get_mean_std(MIMIC_curves_list)
df.loc['MIMIC','best_fitness'] = np.max(mimic_mean)
df.loc['MIMIC','duration'] = round(np.mean(duration_list),4) 

df.to_csv(OUTPUT_DIR + os.sep+ '{}_metrics.csv'.format(PROBLEM))

## plot runtime and best fitness
generate_plot.plot_runTime(df, '{}-RunTime'.format(PROBLEM), OUTPUT_DIR+os.sep+'{}-RunTime.png'.format(PROBLEM))
generate_plot.plot_bestFitness(df, '{}-BestFitness'.format(PROBLEM), OUTPUT_DIR+os.sep+'{}-BestFitness.png'.format(PROBLEM))

best_curves_dict = {'RHC_mean': rhc_mean, 
               'RHC_std': rhc_std, 
               'SA_mean': sa_mean, 
               'SA_std': sa_std, 
               'GA_mean': ga_mean, 
               'GA_std': ga_std, 
               'MIMIC_mean': mimic_mean, 
               'MIMIC_std': mimic_std, 
               }
Best_curve_outfile = OUTPUT_DIR + os.sep+'{}-best_curves.pkl'.format(PROBLEM)
with open(Best_curve_outfile,'wb') as f:
    pickle.dump(best_curves_dict,f)
    
generate_plot.plot_best_curves(best_curves_dict, 
                               '{}-comparison'.format(PROBLEM),
                               OUTPUT_DIR+os.sep+'{}-bestCurves.png'.format(PROBLEM))
