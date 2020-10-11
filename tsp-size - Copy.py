import sys
import six
sys.modules['sklearn.externals.six'] = six
import csv
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import os
import numpy as np
import algorithm_hiive as algorithm
import pandas as pd
import generate_plot
import random

###################################################
RHC_MAX_ATTEMPTS = 3000
RHC_MAX_ITER = 3000
PROBLEM = 'TSP'
MAX_ATTEMPTS = 50
MAX_ITER = 3000
SEED_LIST = range(5)
TSP_num_points_list = [10, 20, 30, 50]

######################################################3
## define fitness and problem        
OUTPUT_DIR = 'results'

##########################################
# run test with different problem size.
df_size=pd.DataFrame()
for TSP_num_points in TSP_num_points_list:
    ## create distance function
    dist_list =[]
    for x in range(TSP_num_points):
        for y in range(x, TSP_num_points):
            dist_list.append((x, y, random.uniform(0.0, 1.0)))
            
    ## define fitness and problem        
    fitness_dists = mlrose.TravellingSales(distances=dist_list)    
    problem = mlrose.TSPOpt(length=TSP_num_points, fitness_fn=fitness_dists, maximize=False)
    problem.set_mimic_fast_mode(True)
    ##
    rhc_best_fitness = 0
    for seed in SEED_LIST:
        best_state, best_fitness, duration, curve = algorithm.run_RHC(problem=problem, 
                                                                      max_attempts=RHC_MAX_ITER, 
                                                                      max_iters=RHC_MAX_ITER, 
                                                                      restarts=0,
                                                                      random_seed= seed
                                                                      )
        best_fitness = -best_fitness
        index = 'RHC-'+str(TSP_num_points)
        df_size.loc[index,'size']= TSP_num_points
        df_size.loc[index,'type']= 'RHC'
        if best_fitness > rhc_best_fitness:
            rhc_best_fitness  = best_fitness
            df_size.loc[index,'best_fitness']= rhc_best_fitness

    sa_best_fitness = 0
    for seed in SEED_LIST:
        _, best_fitness, duration, curve = algorithm.run_SA(problem=problem, 
                                                    max_attempts=MAX_ATTEMPTS, 
                                                    max_iters=MAX_ITER,
                                                    init_temp=200,
                                                    decay= 0.2,
                                                    decay_method = 'Exp')
        best_fitness = -best_fitness
        index = 'SA-'+str(TSP_num_points)
        df_size.loc[index,'size']= TSP_num_points
        df_size.loc[index,'type']= 'SA'
        if best_fitness > sa_best_fitness:
            sa_best_fitness = best_fitness
            df_size.loc[index,'best_fitness']= sa_best_fitness        

    ga_best_fitness = 0
    for seed in SEED_LIST:
        _, best_fitness, duration, curve = algorithm.run_GA(problem=problem, 
                                                            max_attempts=MAX_ATTEMPTS, 
                                                            max_iters=MAX_ITER,
                                                            mutation_prob=0.02, 
                                                            pop_size=200,
                                                            random_seed=seed)
        best_fitness = -best_fitness
        index = 'GA-'+str(TSP_num_points)
        df_size.loc[index,'size']= TSP_num_points
        df_size.loc[index,'type']= 'GA'
        if best_fitness > ga_best_fitness:
            ga_best_fitness =best_fitness
            df_size.loc[index,'best_fitness']= ga_best_fitness  

    mimic_best_fitness = 0
    for seed in SEED_LIST:            
        _, best_fitness, duration, curve = algorithm.run_MIMIC(problem=problem, 
                                                               max_attempts=MAX_ATTEMPTS, 
                                                               max_iters=MAX_ITER,
                                                               keep_pct=0.2, 
                                                               pop_size=200,
                                                               random_seed=seed)
        best_fitness = -best_fitness
        index = 'MIMIC-'+str(TSP_num_points)
        df_size.loc[index,'size']= TSP_num_points
        df_size.loc[index,'type']= 'MIMIC'
        if best_fitness > mimic_best_fitness:
            mimic_best_fitness  = best_fitness
            df_size.loc[index,'best_fitness']= mimic_best_fitness  

df_size.to_csv(OUTPUT_DIR+os.sep+'{}_size.csv'.format(PROBLEM))
generate_plot.plot_sizeChart(df_size, '{}: Fitness vs Size of Problem'.format(PROBLEM), OUTPUT_DIR+os.sep+'{}-size.png'.format(PROBLEM))