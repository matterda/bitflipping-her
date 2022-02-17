from copy import deepcopy
from collections import deque
import random

from src.config import buffer_size


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
    def __init__(self):
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