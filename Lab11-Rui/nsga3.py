from math import factorial
import random

import matplotlib.pyplot as plt
import numpy
import pymop.factory

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

# Problem definition
PROBLEM = "dtlz2"
NOBJ = 1
NDIM = 2
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = -1.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
##

import math

def eval(individual):
    print(individual)
    x1, x2 = individual
    z1 = math.sqrt(x1 * x1 + x2 * x2)
    z2 = math.sqrt((x1 - 1) * (x1 - 1) + (x2 + 1) * (x2 + 1))
    if z1 == 0:
        t1 = 4
    else:
        t1 = math.sin(4 * z1) / z1
    if z2 == 0:
        t2 = 2.5
    else:
        t2 = math.sin(2.5 * z2) / z2
    
    f1 = t1 + t2
    
    if z1 == 0:
        f2 = 1 - 5
    else:
        f2 = 1 - math.sin(5 * z1) / z1
    return f1,f2

# Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 100
CXPB = 1.0
MUTPB = 1.0
##

# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# Create classes
creator.create("FitnessMaxMin", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMaxMin)
##


# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##


def main(seed=None):
    pareto = tools.ParetoFront()
    
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    # Update the Pareto front
    pareto.update(pop)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # Update the Pareto front
        pareto.update(pop)
        
        if gen == 99: 
            # Plot the population in the objective space and the Pareto front
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            p = numpy.array([ind.fitness.values for ind in pop])
            ax.scatter(p[:, 0], p[:, 1], marker="o", s=24)
            ax.view_init(elev=11, azim=-25)
            ax.autoscale(tight=True)
            plt.tight_layout()
            plt.savefig(f"Lab11-Rui/nsga3_gen_{gen}.png")

    return pop, logbook, pareto

import numpy as np

if __name__ == "__main__":
    pop, stats, pareto = main()
    pop_fit = numpy.array([ind.fitness.values for ind in pop])

    pf  = problem.pareto_front(ref_points)
    #print(igd(pop_fit, pf))

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as Axes3d

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")

    # p = numpy.array([ind.fitness.values for ind in pop])
    # ax.scatter(p[:, 0], p[:, 1], marker="o", s=24, label="Final Population")

    # # ax.scatter(pf[:, 0], pf[:, 1], marker="x", c="k", s=32, label="Ideal Pareto Front")

    # # ref_points = tools.uniform_reference_points(NOBJ, P)

    # # ax.scatter(ref_points[:, 0], ref_points[:, 1], marker="o", s=24, label="Reference Points")

    # ax.view_init(elev=11, azim=-25)
    # ax.autoscale(tight=True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("nsga3.png")
    
    front = np.array([ind.fitness.values for ind in pareto])

    print(front[:,0], front[:,1])
    plt.scatter(front[:,0], front[:,1], c="b", s=10)
    plt.axis("tight")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.savefig("Lab11-Rui/nsga3.png")