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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


# The general formula to update the speed of a particle is:
# Vi(t+1) = w Vi(t) + c1 r1 (pbesti - Xi(t)) + c2 r2(gbesti - Xi(t))
# where r1 and r2 are random numbers between 0 and 1, constants w (inertia weight) , c1,
# and c2 are parameters to the PSO algorithm, and pbesti is the position that gives the best
# f(x) value ever explored by particle i and gbesti is the best position explored by all the
# particles in the swarm.
def updateParticle(part, best, r1, r2, w, c1, c2):
    # v_u1 = c1 r1 (pbesti - Xi(t))
    c1u1 = (random.uniform(0, r1)*c1 for _ in range(len(part)))
    v_u1 = map(operator.mul, c1u1, map(operator.sub, part.best, part))
    
    # v_u2 = c2 r2(gbesti - Xi(t))
    c2u2 = (random.uniform(0, r2)*c2 for _ in range(len(part)))
    v_u2 = map(operator.mul, c2u2, map(operator.sub, best, part))
    
    # v_p = w Vi(t)
    v_p = [w * s for s in part.speed]
    
    # Vi(t+1) = w Vi(t) + v_u1 + v_u2
    part.speed = list(map(operator.add, v_p, map(operator.add, v_u1, v_u2)))
    
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-10, pmax=10, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, r1=1.0, r2=1.0, w=1.0, c1=1.5, c2=2.5) 
# slides: Some basic settings for the scalars are w = 1, and c1+c2=4
toolbox.register("evaluate", benchmarks.h1)

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
        
    print('Best: ', best, ', Fitness: ', best.fitness.values)
    
    return pop, logbook, best

if __name__ == "__main__":
    main()

