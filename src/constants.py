import numpy as np
import copy
import math

# Constant Variable
number_steps = int(360/24)
actions = list(np.around(np.linspace(0.001,0.05,100), decimals=4))
products = list(np.around(np.linspace(0,0.01,11),3)) + list(np.around(np.linspace(0.02,0.1,9),3)) + list(np.around(np.linspace(0.2,1,9),3)) + list(np.around(np.linspace(1.5,10,18),3)) + list(np.around(np.linspace(11,50,40),3))
impuritys = list(np.around(np.linspace(0,0.01,11),3)) + list(np.around(np.linspace(0.02,0.1,9),3)) + list(np.around(np.linspace(0.2,1,9),3)) + list(np.around(np.linspace(1.5,10,18),3)) + list(np.around(np.linspace(11,40,30),3))
cell_densitys = list(np.linspace(0,80,81))
actions_downstream = list(np.around(np.linspace(1,4,4), decimals=0))
actions_downstream = list(range(1, 6))
vector_cartesian = products+impuritys+cell_densitys+actions+actions_downstream
# UCB parameter
c = 1

# set for Break current planning with probability
boolvalue = [True, False]

limitation_P = 1000
limitation_I = 1000

# boundary
cf = 2

# purity
aita = 2

pdd = 100000

# the unit price of bioProduct
UPB = 30

# yield penalty cost parameter
cl = 50

# 20-60
product_r = 50

# 0.85 - 0.93
purity_r = 0.85

maxmium_value = 1000000
d = len(vector_cartesian) - 1 - 1 - 1
H = number_steps + 3
K = 30
beta = 1*d*H*math.sqrt(2*d*K*H/0.2)
unit_price =50
cl = 0.5
aita_d = 1

#matrix parameter
lambuda = 1

# [] store action value
action_set = copy.deepcopy(actions)
action_dic = {}
action_set_downstream = copy.deepcopy(actions_downstream)
action_dic_downstream = {}
for i in range(len(actions)):
    # action value map action lable
    action_dic[actions[i]] = i
for i in range(len(action_set_downstream)):
    # action value map action lable
    action_dic_downstream[actions_downstream[i]] = i

inverse_action_dic = {}
for j in range(len(actions)):
    inverse_action_dic[j] = actions[j]

inverse_action_dic_downstream = {}
for j in range(len(action_set_downstream)):
    inverse_action_dic_downstream[j] = actions_downstream[j]