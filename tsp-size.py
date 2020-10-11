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
import generate_plot,random

###################################################
PROBLEM = 'tsp'
MAX_ATTEMPTS = 50
MAX_ITER = 3000
SEED_LIST = range(5)
SIZE_LIST = [10, 20, 30, 40, 50]
OUTPUT_DIR = 'results'

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

TSP_num_points_list = [10, 20, 30, 50]

######################################################3
def write2file(file, data, mode='a'):
    with open(file, mode) as f:
        f.write(data)

# Initialize fitness function object using pre-defined class
fitness = mlrose.FlipFlop()
# Define optimization problem object
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
output_file = OUTPUT_DIR+os.sep+'{}_size.csv'.format(PROBLEM)
write2file(output_file, 'index,size,type,best_fitness\n','w')

##########################################
# run test with different problem size.
df_size=pd.DataFrame()
for size in TSP_num_points_list:
    ## create distance function
    dist_list =[]
    for x in range(size):
        for y in range(x, size):
            dist_list.append((x, y, random.uniform(0.0, 1.0)))
            
    ## define fitness and problem        
    fitness_dists = mlrose.TravellingSales(distances=dist_list)    
    problem = mlrose.TSPOpt(length=size, fitness_fn=fitness_dists, maximize=False)
    problem.set_mimic_fast_mode(True)
    
    ##
    rhc_best_fitness = -np.inf
    for seed in SEED_LIST:
        best_state, best_fitness, duration, curve = algorithm.run_RHC(problem=problem, 
                                                                      max_attempts=RHC_MAX_ITER, 
                                                                      max_iters=RHC_MAX_ITER, 
                                                                      restarts=10,
                                                                      random_seed= seed
                                                                      )
        best_fitness = -best_fitness
        index = 'RHC-'+str(size)
        df_size.loc[index,'size']= size
        df_size.loc[index,'type']= 'RHC'
        if best_fitness > rhc_best_fitness:
            rhc_best_fitness = best_fitness
            df_size.loc[index,'best_fitness']= rhc_best_fitness
    
    write2file(output_file, '{},{},{},{}\n'.format(index,size,'RHC',rhc_best_fitness),'w')
    
    sa_best_fitness = -np.inf
    for seed in SEED_LIST:
        _, best_fitness, duration, curve = algorithm.run_SA(problem=problem, 
                                                    max_attempts=MAX_ATTEMPTS, 
                                                    max_iters=MAX_ITER,
                                                    init_temp=SA_temp,
                                                    decay= SA_decay,
                                                    decay_method = 'Exp')
        best_fitness = -best_fitness
        index = 'SA-'+str(size)
        df_size.loc[index,'size']= size
        df_size.loc[index,'type']= 'SA'
        if best_fitness > sa_best_fitness:
            sa_best_fitness = best_fitness
            df_size.loc[index,'best_fitness']= sa_best_fitness        

    write2file(output_file, '{},{},{},{}\n'.format(index,size,'SA',sa_best_fitness),'w')

    ga_best_fitness = -np.inf
    for seed in SEED_LIST:
        _, best_fitness, duration, curve = algorithm.run_GA(problem=problem, 
                                                            max_attempts=MAX_ATTEMPTS, 
                                                            max_iters=MAX_ITER,
                                                            mutation_prob=GA_mutation, 
                                                            pop_size=GA_pop_size,
                                                            random_seed=seed)
        best_fitness = -best_fitness
        index = 'GA-'+str(size)
        df_size.loc[index,'size']= size
        df_size.loc[index,'type']= 'GA'
        if best_fitness > ga_best_fitness:
            ga_best_fitness = best_fitness
            df_size.loc[index,'best_fitness']= ga_best_fitness  
    write2file(output_file, '{},{},{},{}\n'.format(index,size,'GA',ga_best_fitness),'w')

    mimic_best_fitness = -np.inf
    for seed in SEED_LIST:            
        _, best_fitness, duration, curve = algorithm.run_MIMIC(problem=problem, 
                                                               max_attempts=MAX_ATTEMPTS, 
                                                               max_iters=MAX_ITER,
                                                               keep_pct=MIMIC_keep_pct, 
                                                               pop_size=MIMIC_pop_size,
                                                               random_seed=seed)
        best_fitness = -best_fitness
        index = 'MIMIC-'+str(size)
        df_size.loc[index,'size']= size
        df_size.loc[index,'type']= 'MIMIC'
        if best_fitness > mimic_best_fitness:
            mimic_best_fitness = best_fitness
            df_size.loc[index,'best_fitness']= mimic_best_fitness  
    write2file(output_file, '{},{},{},{}\n'.format(index,size,'MiMIC',mimic_best_fitness),'w')

#df_size.to_csv(OUTPUT_DIR+os.sep+'{}_size.csv'.format(PROBLEM))
#generate_plot.plot_sizeChart(df_size, '{}: Fitness vs Size of Problem'.format(PROBLEM), OUTPUT_DIR+os.sep+'{}-size.png'.format(PROBLEM))