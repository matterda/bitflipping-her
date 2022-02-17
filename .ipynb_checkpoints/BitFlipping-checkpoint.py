import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import random
from collections import deque

from copy import deepcopy


"""
-----
TODO: implement the 'future' strategy for HER with a given K
-----
"""


class BitFlipEnv:
    def __init__(self, N):
        self.N = N
    
    def reset(self):
        state = torch.rand((1,self.N)).round()
        goal = torch.rand((1,self.N)).round()
        done = False
        return torch.cat((state, goal), 1), done
    
    def step(self, x, action):
        # x (and thus y) contains both the state and the goal!
        # therefore, the returned y still contains both state and goal
        y = x.detach().clone()
        y[0,action] = (y[0,action]+1)%2
        done, reward = self.compute_done_reward(y)
        dist = (y[0,:self.N] - y[0,self.N:]).abs().sum()
        return y, reward, done, dist
    
    def state_equal_goal(self, y):
        # splits y in state, goal and True if they are equal
        return (y[0,:self.N] == y[0,self.N:]).all()
    
    def compute_done_reward(self, y):
        # if state == goal:
        # done = True, reward = 0
        # else:
        # done = False, reward = -1
        if self.state_equal_goal(y):
            return True, 0.
        return False, -1.
    
    def goal_substitution(self, from_state, to_state):
        # goal substitution
        new_goal = to_state[0,:self.N]

        state_with_new_goal = from_state.detach().clone()
        state_with_new_goal[0,self.N:] = new_goal

        new_state_with_new_goal = to_state.detach().clone()
        new_state_with_new_goal[0,self.N:] = new_goal
        return state_with_new_goal, new_state_with_new_goal

    
class Policy(nn.Module):
    def __init__(self, M, N):
        super().__init__()
        self.M = M
        self.N = N
        self.fc1 = nn.Linear(M, 256)
        self.fc2 = nn.Linear(256, N)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Logger:
    def __init__(self):
        self.log = {}
    
    def add_log(self, name):
        self.log[name] = []
    
    def add_item(self, name, x):
        self.log[name].append(x)
    
    def get_log(self, name):
        return self.log[name]


class Memory:
    def __init__(self, buffer_size):
        # memory of all stored transitions, across episodes
        self.replay = deque(maxlen=buffer_size)
        #  K = 4: for each replay sample take 4 HER samples
    
    def reset_episode(self):
        self.episode = []
    
    def _preprocess(self, state, action, reward, new_state, done):
        return [
            state.squeeze(0).detach().clone(),
            action.detach().clone(),
            reward,
            new_state.squeeze(0).detach().clone(),
            done
        ]
    
    def add(self, state, action, reward, new_state, done):
        self.replay.append(
            self._preprocess(state, action, reward, new_state, done)
        )
    
    def last_elements(self, T):
        '''
        Returns last T elements of self.replay
        '''
        start_t = len(self.replay) - T
        return [self.replay[t+start_t] for t in range(T)]
        
    
    def sample_her(self, n_steps, k):
        events = self._last_elements(n_steps)
        
        for t in range(n_steps):
            state, action, _, next_state, _, info = deepcopy(events[t])
            
            sampled_transitions = self.sample_transitions(events, t, k)
            
            for transition in sampled_transitions:
                additional_goal = transition[0]
    
    def sample_transitions(self, events, min_t, k):
        # events is a list of transitions
        # min_t is the starting position from which to sample
        # k is the number of transitions to sample *after* min_t
        if k == 1:
            # 'final' strategy
            return [events[-1]]
        
        # 'future' strategy
        n_steps = len(events)
        k = min(k, n_steps - 1 - min_t)
        if k > 0:
            return random.sample(events, k=k)
        return []
    

class DQN:
    def __init__(self, env, gamma, buffer_size):
        self.env = env
        self.N = env.N
        
        # HER parameters
        self.use_her = True
        self.K = 4 # ratio of HER data to data coming from normal experience replay
        
        self.gamma = gamma
        self.buffer_size = buffer_size
        
        self.model = Policy(self.N*2, self.N)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        
        #self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.batch_size = 64
        self.e = 0.9
        self.e_min = 0.1
        self.e_decay = 0.95
        self.steps = 0
        
        self.logger = Logger()
        self.logger.add_log('final_dist')
        
        self.memory = Memory(self.buffer_size)
        
        self.step_counter = 0
        self.update_target_step = 1000
    
    def episode(self):
        state, done = self.env.reset()
        
        min_dist = self.N
        
        for t in range(self.N):
            self.steps += 1
            self.e = self.e_min + (self.e-self.e_min) * self.e_decay
            
            Q = self.model(state)
            
            if random.random() <= self.e:
                action = torch.randint(self.N, (1, ))[0]
            else:
                action = torch.argmax(Q)
            new_state, reward, done, dist = self.env.step(state, action.item())
            
            if dist < min_dist:
                min_dist = dist
            
            if (t+1) == self.N:
                done = True
            
            # > standard experience replay
            self.memory.add(state, action, reward, new_state, done)
            
            # > HER
            if self.use_her:
                #self.her(t+1)
                self.her(state, action, reward, new_state, done)
            
            state = new_state
        
            self.step_counter += 1
            if self.step_counter >= self.update_target_step:
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
            
            if (t+1) == self.N:
                break
        
        loss = self.update_model()
        
        self.logger.add_item('final_dist', min_dist)
    
    """
    def her(self, n_steps):
        '''
        > Hindsight Experience Replay
        Samples additional goals from the current episode
        and add them to the experience replay
        '''
        events = self.memory.last_elements(n_steps)
        
        for t in range(n_steps):
            state, action, _, next_state, _ = deepcopy(events[t])
            
            achieved_goal = next_state.detach().clone()
            achieved_goal = torch.unsqueeze(achieved_goal, 0)
            
            selected_transitions = self.memory.sample_transitions(events, t, self.K)
            
            for transition in selected_transitions:
                # new_goal is the state of the sampled transition
                additional_goal = transition[0]
                additional_goal = torch.unsqueeze(additional_goal, 0)
                state_with_new_goal, new_state_with_new_goal = self.env.goal_substitution(achieved_goal, additional_goal)
                
                _, reward = self.env.compute_done_reward(new_state_with_new_goal)
                
                self.memory.add(state_with_new_goal, action, reward, new_state_with_new_goal, True)
    """
    
    
    def her(self, state, action, reward, new_state, done):
        '''
        > Hindsight Experience Replay
        Samples additional goals from the current states
        and add them to the experience replay
        '''
        if not done:
            # goal substitution
            state_with_new_goal, new_state_with_new_goal = self.env.goal_substitution(state, new_state)
            """
            new_goal = new_state[0,:self.N]

            state_with_new_goal = state.detach().clone()
            state_with_new_goal[0,self.N:] = new_goal

            new_state_with_new_goal = new_state.detach().clone()
            new_state_with_new_goal[0,self.N:] = new_goal
            """
            self.memory.add(state_with_new_goal, action, 0., new_state_with_new_goal, True) 
    
    
    """
    def memorize(self, state, action, reward, new_state, done):
        self.memory.append([
                state.squeeze(0).detach().clone(),
                 action.detach().clone(),
                 reward,
                 new_state.squeeze(0).detach().clone(),
                 done
        ])
    """
    
    def update_model(self):
        self.optimizer.zero_grad()
        
        num = len(self.memory.replay)
        K = min([num, self.batch_size])
        
        samples = random.sample(self.memory.replay, K)
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.stack(S0)
        A0 = torch.tensor(A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor(R1, dtype=torch.float).view(K, -1)
        S1 = torch.stack(S1)
        D1 = torch.tensor(D1, dtype=torch.float)
        
        target_q = R1.squeeze() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1)
        policy_q = self.model(S0).gather(1,A0)
        
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        
        return L.detach().item()