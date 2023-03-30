import numpy as np

# Constant Variable
number_steps = int(360/24)
actions = list(np.around(np.linspace(0.01,0.1,100),decimals=4))
products = list(np.around(np.linspace(0,0.01,11),3)) + list(np.around(np.linspace(0.02,0.1,9),3)) + list(np.around(np.linspace(0.2,1,9),3)) + list(np.around(np.linspace(1.5,10,18),3)) + list(np.around(np.linspace(11,50,40),3))
impuritys = list(np.around(np.linspace(0,0.01,11),3)) + list(np.around(np.linspace(0.02,0.1,9),3)) + list(np.around(np.linspace(0.2,1,9),3)) + list(np.around(np.linspace(1.5,10,18),3)) + list(np.around(np.linspace(11,40,30),3))
cell_densitys = list(np.linspace(0,80,81))
actions_downstream = list(np.around(np.linspace(1,4,4),decimals=0))
vector_cartesian = products+impuritys+cell_densitys+actions+actions_downstream
product_r = 50