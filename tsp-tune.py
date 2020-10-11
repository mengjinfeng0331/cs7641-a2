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

###################################################
RHC_MAX_ATTEMPTS = 3000
RHC_MAX_ITER = 3000
PROBLEM = 'tsp'
MAX_ATTEMPTS = 200
MAX_ITER = 3000
SEED_LIST = range(5)
OUTPUT_DIR = 'results'
######################################################
def write2file(file, data, mode='a'):
    with open(file, mode) as f:
        f.write(data)
# Initialize fitness function object using pre-defined class
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

OUTPUT_DIR = 'results'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

## RHC
def RHC():
    fitness_list = []
    for seed in SEED_LIST:
        best_state, best_fitness, duration, curve = algorithm.run_RHC(problem=problem,
                                                                      max_attempts=RHC_MAX_ITER, 
                                                                      max_iters=RHC_MAX_ITER, 
                                                                      restarts=10,
                                                                      random_seed= seed
                                                                      )
        best_fitness = -best_fitness
        fitness_list.append(best_fitness)

### SA
def SA():
    init_temp=[1, 10, 100, 200, 300]
    #init_temp=[200]
    decay_list=[0.001,0.005, 0.05, 0.1, 0.2]
    out_file = OUTPUT_DIR + os.sep+ '{}-tune_SA.csv'.format(PROBLEM)
    header=  'temp,decay,best_fitness\n'
    write2file(out_file,header,'w')
    #decay_list=[ 0.2]
    for temp in init_temp:
        for decay in decay_list:
            fitness_list = []
            for seed in SEED_LIST:
                _, best_fitness, duration, curve = algorithm.run_SA(problem=problem,
                                                                    max_attempts=MAX_ATTEMPTS, 
                                                                    max_iters=MAX_ITER,
                                                                    init_temp=temp,
                                                                    decay= decay,
                                                                    decay_method = 'Exp')
                best_fitness = -best_fitness
                fitness_list.append(best_fitness)
            write2file(out_file,'{},{},{}\n'.format(temp,decay,np.mean(fitness_list)),'a')

### GA
def GA():            
    pop_size = [100,200, 400, 500, 600]
    #pop_size = [200]
    #mutation_prob = [0.2]
    mutation_prob = [0.01, 0.05, 0.1, 0.2, 0.3,0.4,0.5]
    out_file = OUTPUT_DIR + os.sep+ '{}-tune_GA.csv'.format(PROBLEM)
    header=  'mutation,pop,best_fitness\n'
    write2file(out_file,header,'w')
    for mutation in mutation_prob:
        for pop in pop_size:
            fitness_list = []
            for seed in SEED_LIST:
                print('GA: keep_pct :{}, pop_size:{}'.format(mutation, pop))
                
                _, best_fitness, duration, curve = algorithm.run_GA(problem=problem,
                                                                    max_attempts=MAX_ATTEMPTS, 
                                                                    max_iters=MAX_ITER,
                                                                    mutation_prob=mutation, 
                                                                    pop_size=pop,
                                                                    random_seed=seed)
                best_fitness = -best_fitness
                fitness_list.append(best_fitness)
            write2file(out_file,'{},{},{}\n'.format(mutation,pop,np.mean(fitness_list)),'a')
        
### MIMIC
def MIMIC():
    keep_pct_list = [0.1, 0.2, 0.3, 0.4, 0.5,0.6]
    #keep_pct_list = [0.4]
    pop_size = [100,200, 400, 500, 600]
    out_file = OUTPUT_DIR + os.sep+ '{}-tune_MIMIC.csv'.format(PROBLEM)
    header=  'keep_pct,pop,best_fitness\n'
    write2file(out_file,header,'w')
    #pop_size = [200]
    for keep_pct in keep_pct_list:
        for pop in pop_size:
            fitness_list = []
            for seed in SEED_LIST:
    #            print('MIMIC: keep_pct :{}, pop_size:{}'.format(keep_pct, pop))
                
                _, best_fitness, duration, curve = algorithm.run_MIMIC(problem=problem,
                                                                       max_attempts=MAX_ATTEMPTS, 
                                                                       max_iters=MAX_ITER,
                                                                       keep_pct=keep_pct, 
                                                                       pop_size=pop,
                                                                       random_seed=seed)
                best_fitness = -best_fitness
                fitness_list.append(best_fitness)
    
            write2file(out_file,'{},{},{}\n'.format(keep_pct,pop,np.mean(fitness_list)),'a')

if '__main__' == __name__:
    if sys.argv[1] == 'RHC':
        RHC()
    if sys.argv[1] == 'SA':
        SA()        
    if sys.argv[1] == 'GA':
        GA()
    if sys.argv[1] == 'MIMIC':
        MIMIC()