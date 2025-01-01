import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import typing

import gymnasium as gym
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class TradingEnv(gym.Env):

    def __init__(self,df:pd.DataFrame,target:float,initial_balance:float = 100) -> None:
        """
        Initializes a custom financial environment for reinforcement learning.

        This function sets up the environment with financial data, defines the observation 
        and action spaces, and initializes key attributes such as the balance, _coin _holdings, 
        and price tracking.

        Parameters:
        -----------
            df (pd.DataFrame): A DataFrame containing the financial data with at least a 'Close' column.
            window_size (int, optional): The size of the observation window for the environment. 
                Defaults to 1.
            initial_balance (float, optional): The initial USD balance for the simulation. 
                Defaults to 10,000.

        Returns:
        --------
            None
        """
        super().__init__()

        self._df=df.copy()
        self._close = self._df['Close']
        self._percentage_change = self._df['Close'].pct_change(1).fillna(0)*100

        self._initial_balance = initial_balance
        self._target = target
        
        self.observation_space=gym.spaces.Box(
                            low = np.array([self._close.min()*0.9,self._percentage_change.min(),0,-100]),
                            high = np.array([self._close.max()*1.1,self._percentage_change.max(),1,1000]),
                            shape=(4,),
                            dtype=float
                        )

        self._fg,self._ax = None,None

        self.action_space=gym.spaces.Discrete(3)

    def step(self,action:int) -> typing.Iterable:
        """
        Performs a single step in the financial trading environment based on the given action.

        This method updates the environment's state, calculates the reward, and determines 
        whether the episode should terminate or truncate. It handles the buying, selling, 
        or _holding of assets and adjusts the balance and _coin holdings accordingly.

        Parameters:
        -----------
            action (int): The action to be taken. 
                - 0: Buy
                - 1: Hold
                - 2: Sell

        Returns:
        --------
            typing.Iterable: A tuple containing:
                - new_state (np.ndarray): The updated state of the environment after the action.
                - reward (float): The reward obtained from taking the action.
                - terminate (bool): Whether the episode has ended.
                - truncate (bool): Whether the episode was truncated.
                - info (dict): Additional information (empty dictionary in this case).
        """

        reward = -1
        self._action_list.append(action)
        terminate,truncate=False,False

        if action==2:
            if self._coin>0:

                self._sell_price = self._close.iloc[self._index]
                self._buy_price = None
                
                self._usd = self._close.iloc[self._index]*self._coin
                self._coin = 0

                self._hold = 0
            else:
                reward  = -2

        elif action==0:
            if self._usd>0:

                self._buy_price = self._close.iloc[self._index]
                self._sell_price = None

                self._coin = self._usd/self._close.iloc[self._index]
                self._usd = 0

                self._hold = 1
            else:
                reward  = -2

        self._index+=1

        self.total = self._close.iloc[self._index]*self._coin + self._usd
        self.roi = (self.total - self._initial_balance)*100/self._initial_balance
        self.pnl = self.total- self._initial_balance

        #if reached the upper bound of roi
        if self.roi >= 1000 :
            self.roi = 1000
            self.goal = True
            reward = 10
            terminate,truncate=True,True

        #if reached the target at the end of episode
        if (self.total>= self._target) and (self._index==self._df.shape[0]-1):
            self.goal = True
            reward = 10
            terminate,truncate=True,True

        #if finished the episode(end of dataframe)
        if self._index==self._df.shape[0]-1:
            terminate,truncate=True,True

        new_state = np.array([self._close.iloc[self._index],self._percentage_change.iloc[self._index],self._hold,self.roi])

        return new_state,reward,terminate,truncate,{}


    def reset(self,seed=None,options=None) -> typing.Iterable:
        """
        Resets the environment to its initial state.

        This method initializes or resets all key attributes of the environment, such as 
        balance, _coin _holdings, and indices, to prepare for a new episode.

        Parameters:
        -----------
            seed (int, optional): A seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. 
                Defaults to None.

        Returns:
        --------
            typing.Iterable: A tuple containing:
                - initial_state (np.ndarray): The initial state of the environment.
                - info (dict): Additional information (empty dictionary in this case).
        """

        self._action_list = []
        self._index=0

        self._usd = self._initial_balance
        self._coin = 0

        self.total = self._close.iloc[self._index]*self._coin + self._usd
        self.roi = (self.total - self._initial_balance)*100/self._initial_balance
        self.pnl = self.total- self._initial_balance

        self._buy_price = None
        self._sell_price = None

        self._hold = 0
        self.goal = False

        new_state = np.array([self._close.iloc[self._index],self._percentage_change.iloc[self._index],self._hold,self.total])

        return new_state,{}

    def render(self):

        if (self._fg is None) or (self._ax is None):
            self._fg = plt.figure()
            self._ax = self._fg.add_subplot()

        self._ax.cla()
        self._close.plot(ax = self._ax,color = 'C0')
        for i in range(len(self._action_list)):
            if self._action_list[i]==0:
                self._ax.text(i,self._close.iloc[i],'B',color='C2')
            if self._action_list[i]==2:
                self._ax.text(i,self._close.iloc[i],'S',color='C3')
        self._ax.set_title(str(self._action_list))


        if len(plt.get_fignums()) == 0:
            raise Exception('Stop trainning')

        plt.pause(0.0001)

if __name__=='__main__':
    seed = np.linspace(0,4*np.pi,10)
    # noise = 0.2*np.random.randn(1000)

    m = pd.DataFrame(4*np.sin(0.5*seed)+10,columns = ['Close'])
    m['Diff'] = m['Close'].diff(1).fillna(0)
    m['Diff_pct'] = m['Close'].pct_change().fillna(0)*100

    env = TradingEnv(df = m,target = 229.99)
    print(env.observation_space.high)
