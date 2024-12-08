from functional import *

a2c=agent('A2C','MlpPolicy')

a2c.learn()

a2c.evaluate()
a2c.visualize()

class a2c_trading():
    def __init__(self):
        pass
