class TreeNode:

    def __init__(self,p:float,i:float,x:float,action_set:list,dicsubnode:dict,state:list,time:int,reward):
        self.V = 0
        self.N = 1
        self.action = None
        self.p = p
        self.i = i
        self.x = x
        self.state = state
        self.time = time
        self.parent = None
        self.reward = reward
        self.dicsubnode = dicsubnode
        self.action_set = action_set

    def addsubnode(self,node,key):
        self.dicsubnode[key] = node