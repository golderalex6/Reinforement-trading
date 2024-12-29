import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_environment import TradingEnv
from time import sleep

seed = np.linspace(0,4*np.pi,10)
# noise = 0.2*np.random.randn(1000)

m = pd.DataFrame(4*np.sin(0.5*seed)+10,columns = ['Close'])
m['Diff'] = m['Close'].diff(1).fillna(0)
m['Diff_pct'] = m['Close'].pct_change().fillna(0)*100

env = TradingEnv(df = m,target = 229.99)
# observation_max = np.array([m['Close'].max(),m['Diff_pct'].max(),1,500])
# observation_min = np.array([m['Close'].min(),m['Diff_pct'].min(),0,0])

c_learning_rate = 0.1
c_discount_value = 1

v_epsilon = 0.99
v_epsilon_decay = 0.0005

q_table_size = [7,20,2,50]
q_table_segment_size = (env.observation_max - env.observation_min) / q_table_size

def convert_state(real_state):
    q_state = (real_state - env.observation_min) // (q_table_segment_size*1.00001)
    return tuple(q_state.astype(int))

q_table = np.random.uniform(low=-2, high=-1, size=(q_table_size + [env.action_space.n]))


max_total_portforlio = -(10**9)
max_ep_reward = -999999999
max_ep_action_list = []

for ep in range(10):

    next_real_state,reward,terminate,truncate,_=env.reset()
    current_state=convert_state(next_real_state)

    ep_reward = 0
    action_list = []

    while True:

        action = np.argmax(q_table[current_state])

        next_real_state, reward, terminate,truncated, _  = env.step(action=action)

        action_list.append(action)
        ep_reward += reward

        next_state = convert_state(next_real_state)
        current_q_value = q_table[current_state + (action,)]
        new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (reward + c_discount_value * np.max(q_table[next_state]))
        q_table[current_state + (action,)] = new_q_value
        current_state = next_state

        if terminate:
            if env.goal:
                print(ep,'*'*50)
            if ep_reward > max_ep_reward:
                print(ep,'-'*50)
                max_ep_reward = ep_reward
                max_ep_action_list = action_list

            max_total_portforlio = max(max_total_portforlio,env.total)
            break

        env.render()

    v_epsilon -= v_epsilon_decay


# fg.clf()
# ax_1 = fg.add_subplot(1,2,1)
# ax_2 = fg.add_subplot(1,2,2)

# m['Close'].plot(ax=ax_1)
# for i in range(len(max_ep_action_list)):
#     if max_ep_action_list[i]==0:
#         ax_1.text(i,m.iloc[i,0],'B',color='C2')
#     if max_ep_action_list[i]==2:
#         ax_1.text(i,m.iloc[i,0],'S',color='C3')
# ax_1.set_title(f'Best actions {max_total_portforlio},{max_ep_reward}')


# m['Close'].plot(ax=ax_2)
# next_real_state,reward,terminate,truncate,_=env.reset()
# current_state=convert_state(next_real_state)
# i = 0
# while True:
#     action = np.argmax(q_table[current_state])
#     if action == 0:
#         ax_2.text(i,m.iloc[i,0],'B',color='C2')
#     if action == 2:
#         ax_2.text(i,m.iloc[i,0],'S',color='C3')
#     next_real_state, reward, terminate,truncated, _  = env.step(action=action)
#     next_state = convert_state(next_real_state)
#     current_state = next_state
#     i+= 1
#     if terminate:
#         break
# ax_2.set_title(f"Final model's actions {env.total}")

# plt.tight_layout()
# plt.show()
