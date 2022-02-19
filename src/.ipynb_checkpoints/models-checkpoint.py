import torch
from torch import nn
import torch.nn.functional as F

import random
from copy import deepcopy

from src.config import gamma, lr
from src.utils import Logger, Memory


class Policy(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.fc1 = nn.Linear(N*2, 256)
        self.fc2 = nn.Linear(256, N)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, env, N, use_her):
        self.env = env
        self.N = N
        
        # HER parameters
        self.use_her = use_her
        #self.K = 4 
        
        self.model = Policy(self.N)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.batch_size = 64
        self.e = 0.9
        self.e_min = 0.1
        self.e_decay = 0.95
        self.steps = 0
        
        self.logger = Logger()
        self.logger.add_log('final_dist')
        
        self.memory = Memory()
        
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
    
    
    def her(self, state, action, reward, new_state, done):
        '''
        > Hindsight Experience Replay
        Samples additional goals from the current states
        and add them to the experience replay
        '''
        if not done:
            # goal substitution
            state_with_new_goal, new_state_with_new_goal = self.env.goal_substitution(state, new_state)
            self.memory.add(state_with_new_goal, action, 0., new_state_with_new_goal, True) 
    
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
        
        target_q = R1.squeeze() + gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1)
        policy_q = self.model(S0).gather(1,A0)
        
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        
        return L.detach().item()