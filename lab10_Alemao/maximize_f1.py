#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

def z1_fun(X1, X2):
    return math.sqrt((X1**2) + (X2**2))
def z2_fun(X1, X2):
    return math.sqrt(((X1-1)**2) + ((X2+1)**2))

def f1(part):
    X1, X2 = part
    # print(X1,X2)
    # print(((math.sin(4*z1_fun(X1, X2))/z1_fun(X1, X2)) + (math.sin(2.5*z2_fun(X1, X2))/z2_fun(X1, X2))))
    return ((math.sin(4*z1_fun(X1, X2))/z1_fun(X1, X2)) + (math.sin(2.5*z2_fun(X1, X2))/z2_fun(X1, X2))),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, c1, c2, w, phi1, phi2):
    c1 = c1
    c2 = c2
    w = w
    u1 = (random.uniform(0, phi1)*c1 for _ in range(len(part)))
    u2 = (random.uniform(0, phi2)*c2 for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part_speed_old = [w*element for element in part.speed]
    part.speed = list(map(operator.add, part_speed_old, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-3, pmax=3, smin=-2, smax=2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, c1 = 1.0, c2 = 1.0, w = 1.0, phi1=1.0, phi2=1.0)
toolbox.register("evaluate", f1)

def main():
    pop = toolbox.population(n=5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pop, logbook, best

if __name__ == "__main__":
    main()

