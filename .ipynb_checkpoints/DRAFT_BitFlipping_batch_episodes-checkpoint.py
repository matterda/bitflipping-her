class Env:
    def __init__(self, N, episode_batch_size):
        self.N = N
        self.n_ep = episode_batch_size
    
    def reset(self):
        states = torch.rand( (self.n_ep, self.N) ).round()
        goals = torch.rand( (self.n_ep, self.N) ).round()
        done = torch.tensor( [False] * self.n_ep )
        return torch.cat( (states, goals), 1), done
    
    def step(self, x, actions):
        # x (and thus y) contains both the state and the goal!
        # therefore, the returned y still contains both state and goal
        
        y = x.detach().clone()
        # flip the bits indicated by the actions
        y[ range(self.n_ep) , actions ] = (y[ range(self.n_ep) , actions ] + 1) % 2
        # prepare done and rewards tensors
        done = torch.tensor( [False] * self.n_ep )
        rewards = torch.tensor( [-1.] * self.n_ep )
        
        # indexes of state == goal
        idx_same = (y[ :, :self.N ] == y[ :, self.N: ]).all(axis=1)
        # change done and rewards where state == goal
        done[idx_same] = True
        rewards[idx_same] = 0.
        # distance btw states and goals
        dist = (y[ :, :self.N ] - y[ :, self.N: ]).abs().sum(axis=1)
        return y, rewards, done, dist


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
    def __init__(self, buffer_size, K):
        # memory of all stored transitions, across episodes
        self.replay = deque(maxlen=buffer_size)
        # memory of the current episode
        self.episode_batch = np.array()
        #  K = 4: for each replay sample take 4 HER samples
        self.K = K
    
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
    
    def _curr_episode_events(self, T):
        return [self.replay[t] for t in range(T)]
        
    
    def sample(self, T):
        events = _curr_episode_events(self, T)



class DQN:
    def __init__(self, env, gamma, buffer_size, episode_batch_size):
        self.env = env
        self.N = env.N
        
        # number of episodes played simultaneously
        self.n_ep = episode_batch_size
        self.env.n_ep = self.n_ep
        
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
        
        self.memory = deque(maxlen=buffer_size)
        
        self.step_counter = 0
        self.update_target_step = 1000
    
    def episode(self):
        states, done = self.env.reset()
        
        min_dist = self.N
        
        for t in range(self.N):
            self.steps += 1
            self.e = self.e_min + (self.e-self.e_min) * self.e_decay
            
            Qs = self.model(states)
            
            if random.random() <= self.e:
                actions = torch.randint(self.N,(self.n_ep,))
            else:
                actions = torch.argmax(Qs, axis=1) # might be axis=0
            new_states, rewards, done, dist = self.env.step(states, actions.item())
            
            if dist < min_dist:
                min_dist = dist
            
            if (t+1) == self.N:
                done = True
            
            # > standard experience replay
            self.memorize(state, action, reward, new_state, done)
            
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
        
        self.logger.add_item('final_dist',min_dist)
    
    def her(self, state, action, reward, new_state, done):
        '''
        > Hindsight Experience Replay
        Samples additional goals from the current states
        and add them to the experience replay
        '''
        if not done:
            # goal substitution
            new_goal = new_state[0,:self.N]

            state_with_new_goal = state.detach().clone()
            state_with_new_goal[0,self.N:] = new_goal

            new_state_with_new_goal = new_state.detach().clone()
            new_state_with_new_goal[0,self.N:] = new_goal
            
            self.memorize(state_with_new_goal, action, 0., new_state_with_new_goal, True)
    
    def memorize(self, state, action, reward, new_state, done):
        self.memory.append([
                state.squeeze(0).detach().clone(),
                 action.detach().clone(),
                 reward,
                 new_state.squeeze(0).detach().clone(),
                 done
        ])
    
    def update_model(self):
        self.optimizer.zero_grad()
        
        num = len(self.memory)
        K = min([num, self.batch_size])
        
        samples = random.sample(self.memory, K)
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