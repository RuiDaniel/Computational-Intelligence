from simpful import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS = FuzzySystem()

CT1 = TriangleFuzzySet(0, 0, 50, term="Cold")
CT2 = TriangleFuzzySet(30, 50, 70, term="Warm")
CT3 = TriangleFuzzySet(50, 100, 100, term="Hot")
FS.add_linguistic_variable("core_temp", LinguisticVariable([CT1, CT2, CT3], universe_of_discourse=[0,100]))

CS1 = TriangleFuzzySet(0, 0, 1.5, term="Low")
CS2 = TriangleFuzzySet(0.5, 2, 3.5, term="Normal")
CS3 = TriangleFuzzySet(2.5, 4, 4, term="Turbo")
FS.add_linguistic_variable("CPU_speed", LinguisticVariable([CS1, CS2, CS3], universe_of_discourse=[0,4]))

FS1 = TriangleFuzzySet(0, 0, 3500, term="Slow")
FS2 = TriangleFuzzySet(2500, 6000, 6000, term="Fast")
FS.add_linguistic_variable("fan_speed", LinguisticVariable([FS1, FS2], universe_of_discourse=[0,6000]))

FS.add_rules([
    "IF (core_temp IS Cold) AND (CPU_speed IS Low) THEN (fan_speed IS Slow)",
    "IF (core_temp IS Cold) AND (CPU_speed IS Normal) THEN (fan_speed IS Slow)",
    "IF (core_temp IS Cold) AND (CPU_speed IS Turbo) THEN (fan_speed IS Fast)",
    "IF (core_temp IS Warm) AND (CPU_speed IS Low) THEN (fan_speed IS Slow)",
    "IF (core_temp IS Warm) AND (CPU_speed IS Normal) THEN (fan_speed IS Slow)",
    "IF (core_temp IS Warm) AND (CPU_speed IS Turbo) THEN (fan_speed IS Fast)",
    "IF (core_temp IS Hot) AND (CPU_speed IS Low) THEN (fan_speed IS Fast)",
    "IF (core_temp IS Hot) AND (CPU_speed IS Normal) THEN (fan_speed IS Fast)",
    "IF (core_temp IS Hot) AND (CPU_speed IS Turbo) THEN (fan_speed IS Fast)"
	])

FS.set_variable("CPU_speed", 3.5) 
FS.set_variable("core_temp", 90) 

tip = FS.inference()
# print(tip)

cs_range = list(np.arange(0, 4, 0.3))
ct_range = list(range(0, 100, 2))

results = []

for core_t in ct_range:
    for cpu_speed in cs_range:
        FS.set_variable("CPU_speed", cpu_speed) 
        FS.set_variable("core_temp", core_t)
        results.append(FS.inference()['fan_speed'])
      
results = pd.DataFrame(results)
  
# print(results)

def print3d_contour(x, y, Z):
    X, Y = np.meshgrid(x, y)

    # print(Z.shape)
    Z = np.array(Z).reshape(X.shape)
    # print(Z.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Core Temperature')
    ax.set_ylabel('Clock Speed')
    ax.set_zlabel('Fan Speed')

    plt.show()
    
print3d_contour(ct_range, cs_range, results)


# Afonso Alemão

temp_range = list(range(0,100,2))
clk_range = list(np.arange(0,4,0.3))

print(clk_range)

test_output = []

for temp_element in temp_range:
    for clk_element in clk_range:
        FS.set_variable("core_temp", temp_element) 
        FS.set_variable("CPU_speed", clk_element)
        test_output.append(FS.inference()['fan_speed'])

print('test_output: {}'.format(test_output))
print(type(test_output[0]))


# Create a grid of x and y values
temp_grid, clk_grid = np.meshgrid(temp_range, clk_range)

# Reshape the test_output to match the grid dimensions
test_output = np.array(test_output).reshape(temp_grid.shape)
print(test_output.shape)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(temp_grid, clk_grid, test_output, cmap='viridis')

ax.set_xlabel('Core Temperature')
ax.set_ylabel('Clock Speed')
ax.set_zlabel('Fan Speed')

plt.show()
