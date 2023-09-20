import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

# The aim is to control the speed of a CPU fan based on the:
# – Core temperature (in degrees Celsius)
# – Clock speed (in GHz)

FS = sf.FuzzySystem()

# Determine Fuzzy Sets

T1 = sf.TriangleFuzzySet(0,0,50,   term="cold")
T2 = sf.TriangleFuzzySet(30,50,70,  term="warm")
T3 = sf.TriangleFuzzySet(50,100,100, term="hot")
FS.add_linguistic_variable("core_temperature", sf.LinguisticVariable([T1, T2, T3], universe_of_discourse=[0,100]))

C1 = sf.TriangleFuzzySet(0,0,1.5,   term="low")
C2 = sf.TriangleFuzzySet(0.5,2,3.5,  term="normal")
C3 = sf.TriangleFuzzySet(2.5,4,4, term="turbo")
FS.add_linguistic_variable("clock_speed", sf.LinguisticVariable([C1, C2, C3], universe_of_discourse=[0,4]))



F1 = sf.TriangleFuzzySet(0,0,3500,   term="slow")
F2 = sf.TriangleFuzzySet(2500,6000,6000,  term="fast")
FS.add_linguistic_variable("fan_speed", sf.LinguisticVariable([F1, F2], universe_of_discourse=[0,6000]))



FS.add_rules([
    "IF (core_temperature IS cold) AND (clock_speed IS low) THEN (fan_speed IS slow)",
    "IF (core_temperature IS cold) AND (clock_speed IS normal) THEN (fan_speed IS slow)",
    "IF (core_temperature IS cold) AND (clock_speed IS turbo) THEN (fan_speed IS fast)",
    "IF (core_temperature IS warm) AND (clock_speed IS low) THEN (fan_speed IS slow)",
    "IF (core_temperature IS warm) AND (clock_speed IS normal) THEN (fan_speed IS slow)",
    "IF (core_temperature IS warm) AND (clock_speed IS turbo) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed IS low) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed IS normal) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed IS turbo) THEN (fan_speed IS fast)"
	])


temp_range = list(range(0,100,2))
clk_range = list(np.arange(0,4,0.3))

print(clk_range)

test_output = []

for temp_element in temp_range:
    for clk_element in clk_range:
        FS.set_variable("core_temperature", temp_element) 
        FS.set_variable("clock_speed", clk_element)
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

FS.set_variable("core_temperature", 50) 
FS.set_variable("clock_speed", 2) 
fan_speed1 = FS.inference()
print(fan_speed1)