import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, Categorical
import gym
import numpy as np
import random, collections, copy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange


# Define Buffer to save rollout
Transition = collections.namedtuple("Transition", 
                                    ["prev_state", "action", "reward", "state", "done", "action_logprob"])
class Buffer(object):
    def __init__(self):
        self._storage = []

    def push(self, *args):
        self._storage.append(Transition(*args))

    def get(self):  # Get elements from buffer
        temp = copy.deepcopy(self._storage)
        self._storage = []
        return temp

    def __len__(self):
        return len(self._storage)

# Compute advantage & target value and make batch for training
class RolloutLoad(Dataset):
    def __init__(self, rollout, device, critic, gamma, lamda, action_dim, is_conti):
        self.rollout = rollout
        self.device = device
        self.critic = critic
        self.gamma = gamma
        self.lamda = lamda
        if is_conti:
            self.action_dim = action_dim
        self.is_conti = is_conti

    def __getitem__(self, idx):
        prev_state = torch.Tensor(self.rollout[idx].prev_state).to(self.device)
        state = torch.Tensor(self.rollout[idx].state).to(self.device)
        if self.is_conti:
            action = torch.Tensor(self.rollout[idx].action).view(-1,self.action_dim).to(self.device)
            old_action_logprob = torch.Tensor(self.rollout[idx].action_logprob).view(-1,self.action_dim).to(self.device)
        else:
            action = torch.Tensor(self.rollout[idx].action).view(-1,1).to(self.device)
            old_action_logprob = torch.Tensor(self.rollout[idx].action_logprob).view(-1,1).to(self.device)
        reward = torch.Tensor(self.rollout[idx].reward).view(-1,1).to(self.device)
        done = torch.Tensor(self.rollout[idx].done).view(-1,1).to(self.device)

        target_value = (reward + self.gamma * self.critic(state) * (1-done)).detach()
        delta = target_value - self.critic(prev_state)
        delta = delta.detach().cpu().numpy()

        # Computing General Advantage Estimates(GME)
        advantage = []
        estimates = 0.0
        for t in delta[::-1]:
            estimates = self.gamma * self.lamda * estimates + t[0]
            advantage.append([estimates])
        advantage.reverse()
        advantage = torch.tensor(advantage, dtype=torch.float).to(self.device)
        
        return prev_state, action, old_action_logprob, target_value, advantage

    def __len__(self):
        return len(self.rollout)


# Define Actor model : Policy function μ(s)
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, actor_dims, 
                    is_conti=False, action_max=None, action_std=0.5):
        super(Actor, self).__init__()
        self.is_conti = is_conti
        if is_conti:
            self.action_max = action_max

        model = [nn.Linear(input_dim, actor_dims[0]), nn.ReLU()]

        for i in range(len(actor_dims)-1):
            model += [nn.Linear(actor_dims[i], actor_dims[i+1]), nn.ReLU()]
        self.model = nn.Sequential(*model)

        if is_conti:
            fc = [nn.Linear(actor_dims[-1], 2*action_dim)]
        else:
            fc = [nn.Linear(actor_dims[-1], action_dim), nn.Softmax(dim=-1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        if self.is_conti :
            mean, std = torch.chunk(x, chunks=2, dim=-1)
            mean, std = self.action_max * torch.tanh(mean), F.softplus(std)
            return mean, std
        else:
            return x

    def act(self, state, device, is_test=False):
        state = torch.as_tensor(state, dtype=torch.float, device=device)
        if self.is_conti:
            mean, std = self.forward(state)
            if is_test:
                dist = Normal(mean, 0.001 * std)
            else:
                dist = Normal(mean, std)
        else:
            action_prob = self.forward(state)
            dist = Categorical(action_prob)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    # Compute log probability of corresponding action
    def get_logprob(self, state, action, device, size_batch):
        if self.is_conti:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action)
        else:
            action_prob = self.forward(state)
            action_prob = rearrange(action_prob, 'b r a -> (b r) a')
            dist = Categorical(action_prob)
            
            action = rearrange(action, 'b r a -> (b r) a')
            log_prob = dist.log_prob(action)
            log_prob = rearrange(log_prob[:,0].view(-1,1), '(b r) a -> b r a', b=size_batch)

        return log_prob


# Define Critic model : Value function V(s)
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, critic_dims):
        super(Critic, self).__init__()
        model = [nn.Linear(input_dim, critic_dims[0]), nn.ReLU()]

        for i in range(len(critic_dims)-1):
            model += [nn.Linear(critic_dims[i], critic_dims[i+1]), nn.ReLU()]
        self.model = nn.Sequential(*model)

        self.fc = nn.Linear(critic_dims[-1], 1)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class Trainer(object):
    def __init__(self, actor_dims, critic_dims, game_name, is_conti, path, load_model, lr,
                    device, maxframe, verbose_freq, eps_clip=0.1, K_epoch=3, len_traj=20, 
                    size_batch=1, num_batch=1, gamma=0.98, lamda=0.95):

        self.game_name = game_name
        self.env = gym.make(game_name) # e.g. 'MountainCarContinuous-v0'
        self.state_dim = self.env.observation_space.shape[0]
        self.is_conti = is_conti
        if is_conti:        
            self.action_dim = self.env.action_space.shape[0]
            self.action_max = float(self.env.action_space.high[0])
        else:
            self.action_dim = self.env.action_space.n
            self.action_max = None
        self.path = path
        self.device = device
        self.maxframe = maxframe
        self.verbose_freq = verbose_freq

        # Define main Actor & Critic model
        self.actor = Actor(self.state_dim, self.action_dim, actor_dims, is_conti, self.action_max).to(device)
        if load_model:
            self.actor.load_state_dict(self.path)
        self.actor_old = Actor(self.state_dim, self.action_dim, actor_dims, is_conti, self.action_max).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.state_dim, self.action_dim, critic_dims).to(device)

        # Set loss & many hyperparamters for training
        self.critic_loss = nn.MSELoss()
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = Buffer()
        self.size_batch = size_batch
        self.num_batch = num_batch
        self.K_epoch = K_epoch
        self.len_traj = len_traj
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lamda = lamda
        self.max_iter_frame = 1000 # Max iteration per each episode


    def update_PPO(self):
        if len(self.rollout) == self.size_batch * self.num_batch:
            temp = copy.deepcopy(self.rollout)
            dataloader = DataLoader(RolloutLoad(temp, self.device, self.critic, self.gamma, 
                                                self.lamda, self.action_dim, self.is_conti),
                                    batch_size=self.size_batch)
            self.rollout = []
            for i in range(self.K_epoch):
                for _, (prev_state, action, old_action_logprob, target_value, advantage) in enumerate(dataloader):

                    action_logprob = self.actor.get_logprob(prev_state, action, self.device, self.size_batch)
                    ratio = torch.exp(action_logprob - old_action_logprob)

                    # Update Actor
                    self.optim_actor.zero_grad()
                    actor_loss = (-torch.min(ratio * advantage, 
                                    torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage)).mean()
                    actor_loss.backward()
                    self.optim_actor.step()

                    # Update Critic
                    self.optim_critic.zero_grad()
                    critic_loss = self.critic_loss(self.critic(prev_state), target_value.detach())
                    critic_loss.backward()
                    self.optim_critic.step()


    def train(self):
        episode, frame = 1, 1
        score = collections.deque(maxlen=self.verbose_freq)
        self.rollout = []

        with tqdm(total = int(self.maxframe)) as pbar:
            while frame < self.maxframe:
                prev_state = self.env.reset() 
                epi_score = 0
                done = False
                self.epi_length = 1
                while not done:
                    # Simulate One trajectory of length len_traj
                    for i in range(self.len_traj):
                        s = torch.Tensor(prev_state).view(-1, self.state_dim).to(self.device)
                        action, action_logprob = self.actor_old.act(s, self.device)
                        if self.is_conti:
                            action = action.detach().cpu().numpy().flatten()
                            action = action.clip(self.env.action_space.low, self.env.action_space.high)
                            action_logprob = action_logprob.detach().cpu().numpy().flatten()
                        else:
                            action = action.item()
                            action_logprob = action_logprob.item()

                        # environment progress
                        state, reward, done, _ = self.env.step(action)

                        self.buffer.push(prev_state, action, reward, state, done, action_logprob)
                        # Save each trajectory as batch
                        if len(self.buffer) == self.len_traj:
                            batch = self.buffer.get()
                            batch = Transition(*zip(*batch))
                            self.rollout.append(batch)
                                
                        prev_state = state
                        epi_score += reward

                        if frame % (self.maxframe*0.5) == 0:
                            self.save(self.path)

                        frame += 1
                        pbar.update(1)
                        self.epi_length += 1

                        if self.max_iter_frame <= self.epi_length:
                            done = True

                        if done:
                            score.append(epi_score)
                            break

                    # Update network after sufficient trajectories are stored
                    self.update_PPO()
                    self.actor_old.load_state_dict(self.actor.state_dict())

                if episode % self.verbose_freq == 0:
                    print(f"Mean score over last {self.verbose_freq} episodes was {np.mean(score):.3f}")
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
                with torch.no_grad():
                    action, _ = self.actor.act(s, self.device, is_test=True)
                if self.is_conti:
                    action = action.detach().cpu().numpy().flatten()
                else:
                    action = action.item()
                
                state, reward, done, _ = self.env.step(action)
                epi_score += reward
                
                if done:
                    print(f"Episode reward : {epi_score:.3f}")
                    break        
            self.env.close()
            self.display_play(frames, j, save_path)