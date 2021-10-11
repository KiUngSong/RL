import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

import gym
import numpy as np
import random, collections, copy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Define Replaybuffer
Transition = collections.namedtuple("Transition", ["prev_state", "action", "state", "reward", "done"])
class Replaybuffer(object):
    def __init__(self, max_size, device):
        self._storage = collections.deque([], maxlen=int(max_size))
        self.device = device

    def push(self, *args):
        self._storage.append(Transition(*args))

    def sample(self, batch_size, state_dim):
        batch = random.sample(self._storage, batch_size)
        batch = Transition(*zip(*batch))

        prev_state = torch.Tensor(batch.prev_state).view(-1, state_dim).to(self.device)
        state = torch.Tensor(batch.state).view(-1, state_dim).to(self.device)
        action = torch.Tensor(batch.action).view(batch_size, -1).to(self.device)
        reward = torch.Tensor(batch.reward).view(batch_size, -1).to(self.device)
        done = torch.Tensor(batch.done).view(batch_size, -1).to(self.device)
        
        return prev_state, action, state, reward, done

    def __len__(self):
        return len(self._storage)


# Initialization function
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Define Actor model : Policy function μ(s)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, dims, action_max):
        super(Actor, self).__init__()
        self.log_alpha = nn.Parameter(torch.Tensor([np.log(0.2)]))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3*1e-4)
        self.action_max = action_max

        model = [nn.Linear(state_dim, dims[0]), nn.ReLU(inplace = True)]
        for i in range(len(dims)-1):
            model += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        model += [nn.Linear(dims[-1], 2 * action_dim)]
        self.model = nn.Sequential(*model)
        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3*1e-4)

    def forward(self, x, is_train=True):
        x = self.model(x)
        mu, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std)
        dist = Normal(mu, std) if is_train else Normal(mu, 1e-7)
        
        mu_sampled = dist.rsample()
        log_prob = dist.log_prob(mu_sampled)
        action = torch.tanh(mu_sampled)
        log_prob -= torch.log(self.action_max * (1 - action.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return self.action_max * action, log_prob

    def update(self, q1, q2, prev_state, target_entropy):
        a, log_prob = self.forward(prev_state)
        entropy = - self.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(prev_state, a), q2(prev_state, a)
        min_q = torch.min(q1_val, q2_val)

        loss = - (min_q + entropy) # Negation for Gradient Ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = - (self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


# Define Critic model : Action-Value function Q(s,a)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, dims):
        super(Critic, self).__init__()
        self.init_s = nn.Sequential(nn.Linear(state_dim, dims[0]), nn.ReLU(inplace=True))
        self.init_a = nn.Sequential(nn.Linear(action_dim, dims[0]), nn.ReLU(inplace=True))

        model = []
        for i in range(len(dims)-1):
            if i == 0:
                linear = nn.Linear(2 * dims[i], dims[i+1])
            else:
                linear = nn.Linear(dims[i], dims[i+1])
            model += [linear, nn.ReLU(inplace = True)]
        model += [nn.Linear(dims[-1], 1)]
        self.model = nn.Sequential(*model)
        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3*1e-4)

    def forward(self, state, action):
        state = self.init_s(state)
        action = self.init_a(action)
        x = torch.cat([state, action], dim=-1)
        x = self.model(x)
        return x

    def update(self, target, prev_state, action):
        loss = F.mse_loss(self.forward(prev_state, action) , target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, target, tau):
        for param_target, param in zip(target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


# Define class to execute total process : network update & train & test with actual play
class Trainer(object):
    def __init__(self, actor_dims, critic_dims, game_name, path, load_model, 
                    batch_size, device, buffer_size=100000, maxframe=100000, verbose=100, tau=0.001):
        self.game_name = game_name
        self.env = gym.make(game_name) # e.g. 'MountainCarContinuous-v0' "LunarLanderContinuous-v2"
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = float(self.env.action_space.high[0])
        self.path = path
        self.device = device
        self.maxframe = maxframe

        self.q1 = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.q1_target = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())

        self.q2 = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.q2_target = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor = Actor(self.state_dim, self.action_dim, actor_dims, self.action_max).to(device)
        if load_model:
            self.actor.load_state_dict(self.path)

        self.replay = Replaybuffer(buffer_size, device)
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = tau
        self.verbose = verbose
        self.max_iter_frame = 1000 # Max iteration per each episode
        self.target_entropy = - np.prod(self.env.action_space.shape)


    def update_SAC(self):
        # load transitions from batch
        prev_state, action, state, reward, done = self.replay.sample(self.batch_size, self.state_dim)
        
        # Compute the target value
        with torch.no_grad():
            next_action, log_prob= self.actor(state)
            entropy = - self.actor.log_alpha.exp() * log_prob
            q1_val, q2_val = self.q1_target(state, next_action), self.q2_target(state, next_action)
            min_q = torch.min(q1_val, q2_val)
            target = reward + self.gamma * (1-done) * (min_q + entropy)

        # Update of critic & Soft-update target critic
        self.q1.update(target, prev_state, action)
        self.q2.update(target, prev_state, action)
        self.q1.soft_update(self.q1_target, self.tau)
        self.q2.soft_update(self.q2_target, self.tau)
        # Update of actor
        self.actor.update(self.q1, self.q2, prev_state, self.target_entropy)


    def train(self):
        episode, frame = 1, 1
        score = collections.deque(maxlen=self.verbose)

        with tqdm(total = int(self.maxframe)) as pbar:
            while frame < self.maxframe:
                prev_state = self.env.reset() 
                epi_score = 0
                done = False
                self.epi_length = 1

                while not done:
                    s = torch.Tensor(prev_state).view(-1, self.state_dim).to(self.device)
                    action, _ = self.actor(s)
                    action = action.data.cpu().numpy().flatten()
                    action = action.clip(self.env.action_space.low, self.env.action_space.high)

                    # environment progress
                    state, reward, done, _ = self.env.step(action)
                    epi_score += reward

                    self.replay.push(prev_state, action, state, reward, done)
                    prev_state = state
                    
                    if frame % (self.maxframe*0.5) == 0:
                        self.save(self.path)

                    frame += 1
                    pbar.update(1)
                    self.epi_length += 1

                    # Update Actor & Critic at the during episode
                    if len(self.replay) > self.batch_size:
                        self.update_SAC()

                    if self.max_iter_frame <= self.epi_length:
                            done = True

                    if done:
                        score.append(epi_score)
                        break

                if episode % self.verbose == 0:
                    print(f"Mean score over last {self.verbose} episodes was {np.mean(score):.3f}", 
                            f"/ Alpha : {self.actor.log_alpha.exp().item():.3f}")
                episode += 1

    # Save the model weight
    def save(self, path):
        torch.save(self.actor.state_dict(), path)
        print("Policy weight was saved")


    # To test actual play with trained Actor policy
    def display_play(self, frames, i, save_path): 
        img = plt.imshow(frames[0]) 
        
        def animate(i): 
            img.set_data(frames[i])
        
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(save_path+'/'+self.game_name+f' Example_{i}.gif', fps=30, dpi=100)
        plt.plot()
        
    def play(self, path, save_path, num_episodes=10):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()
        print("Model loading completed")
        
        for j in range(num_episodes):
            state = self.env.reset()
            frames = [self.env.render(mode='rgb_array')]
            epi_score = 0
            done = False
            while not done:
                frames.append(self.env.render(mode='rgb_array'))
                
                s = torch.from_numpy(state).view(-1, self.state_dim).to(self.device)
                action, _ = self.actor(s, is_train=False)
                action = action.data.cpu().numpy().flatten()
                action = action.clip(self.env.action_space.low, self.env.action_space.high)

                state, reward, done, _ = self.env.step(action)
                epi_score += reward
                if done:
                    print(f"Episode reward : {epi_score:.3f}")
                    break        
            self.env.close()
            self.display_play(frames, j, save_path)


####################################
###  Execute Training at terminal
####################################

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if cuda else "cpu")

    SAC = Trainer(actor_dims=[128,128], critic_dims=[128,128], game_name="LunarLanderContinuous-v2", 
                    path="./SAC_LunarLanderConti.pt", load_model=False, batch_size=128, device=device, 
                    maxframe=400000, verbose=50, buffer_size=100000, tau=0.1)
    SAC.train()