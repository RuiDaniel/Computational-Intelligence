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

p = [[0.10694922, 0.14502253],
 [0.27558179, 0.79057973],
 [0.40075937, 0.92828911],
 [0.57121678, 1.17224258]]

for element in p:
    print(eval(element))