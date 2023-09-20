import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3) # 3 terms para a variável quality -> ex: (poor, average, good) -> default terms 
service.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# You can see how these look with .view()
quality['average'].view()
plt.show()

service.view()
plt.show()

tip.view()
plt.show()



# rule1 = ctrl.Rule(quality['poor'] & service['poor'], tip['low'])
# rule1 = ctrl.Rule(quality['poor'] ~ service['poor'], tip['low'])

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

rule1.view()
plt.show()


tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# In order to simulate this control system, we will create a ControlSystemSimulation
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

# Crunch the numbers
tipping.compute()

print(tipping.output['tip'])
tip.view(sim=tipping)
plt.show()


upsampled = np.arange(0, 11, 1)
x, y = np.meshgrid(upsampled, upsampled)
tips = np.zeros_like(x)

# Loop through the system 21*21 times to collect the control surface
for i in range(11):
    for j in range(11):
        tipping.input['quality'] = x[i, j]
        tipping.input['service'] = y[i, j]
        tipping.compute()
        tips[i, j] = tipping.output['tip']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, tips, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)

cset = ax.contourf(x, y, tips, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, tips, zdir='x', offset=3, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, tips, zdir='y', offset=3, cmap='viridis', alpha=0.5)

ax.view_init(30, 200)