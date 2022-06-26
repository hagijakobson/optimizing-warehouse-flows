"""
Author: Hagi Jakobson

Optimizing Warehouse Flows with Reinforcement Learning 
"""

# Libraries
import numpy as np
np.random.seed(0) # Random global seed

# Configuração dos parameters gamma and alpha for Q-Learning
gamma = 0.75
alpha = 0.9

# Part 1. Environment Setting
location_to_state = {'A':0,
                     'B':1,
                     'C':2,
                     'D':3,
                     'E':4,
                     'F':5,
                     'G':6,
                     'H':7,
                     'I':8,
                     'J':9,
                     'K':10,
                     'L':11}

actions = list(range(12))

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# Part 2. Building an AI solution with Q-Learning
Q = np.zeros([12,12])
for i in range(1000):
    current_state = np.random.randint(0,12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    
    TD = R[current_state, next_state] + \
    gamma*Q[next_state, np.argmax(Q[next_state,:])] - \
    Q[current_state, next_state] 

    Q[current_state, next_state] += alpha*TD 
         
# Part 3.
