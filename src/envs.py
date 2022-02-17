import torch

from src.config import N


class BitFlipEnv:
    def reset(self):
        state = torch.rand((1,N)).round()
        goal = torch.rand((1,N)).round()
        done = False
        return torch.cat((state, goal), 1), done
    
    def step(self, x, action):
        # x (and thus y) contains both the state and the goal!
        # therefore, the returned y still contains both state and goal
        y = x.detach().clone()
        y[0,action] = (y[0,action]+1)%2
        done, reward = self.compute_done_reward(y)
        dist = (y[0,:N] - y[0,N:]).abs().sum()
        return y, reward, done, dist
    
    def state_equal_goal(self, y):
        # splits y in state, goal and True if they are equal
        return (y[0,:N] == y[0,N:]).all()
    
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
        new_goal = to_state[0,:N]

        state_with_new_goal = from_state.detach().clone()
        state_with_new_goal[0,N:] = new_goal

        new_state_with_new_goal = to_state.detach().clone()
        new_state_with_new_goal[0,N:] = new_goal
        return state_with_new_goal, new_state_with_new_goal