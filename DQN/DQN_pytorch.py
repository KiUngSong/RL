import torch
import gym
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import random, collections, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from IPython.display import HTML
from tqdm.notebook import tqdm


# Define Replaybuffer to use experience replay for DQN
Transition = collections.namedtuple("Transition", ["prev_state", "action", "reward", "state", "done"])
class Replaybuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self._storage = collections.deque([], maxlen=max_size)

    def push(self, *args):
        self._storage.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._storage, batch_size)

    def __len__(self):
        return len(self._storage)

# Function to control epsilon
class LinearEpsilonScheduler():
    def __init__(self, initial_eps=0.99, final_eps=0.01, maxframe=1000000):
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.init_frame = maxframe*0.1  # initial_exploration_frame
        self.max_frame = maxframe*0.3  # max_exploration_frame

    def get_epsilon(self, frame):
        if frame < self.init_frame:
            return self.initial_eps
        elif frame > self.max_frame:
            return self.final_eps
        else:
            progress = (frame - self.init_frame) / (self.max_frame - self.init_frame)
            return self.initial_eps + (self.final_eps - self.initial_eps) * progress


# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, dims, device):
        super(DQN, self).__init__()
        self.device = device
        model = [ nn.Linear(input_dim, dims[0]), nn.ReLU(inplace=True) ]
        for i in range(len(dims)-1):
            model += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        x = self.model(x)
        x = self.fc(x)
        return x


# Class for train DQN
class Trainer(object):
    def __init__(self, model_dims, game_name, path, load_model, batch_size, device, buffer_size, maxframe):
        self.game_name = game_name
        self.env = gym.make(game_name) # e.g. 'CartPole-v1'
        self.input_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.path = path
        self.device = device
        self.maxframe = maxframe
        self.eps = LinearEpsilonScheduler(maxframe=self.maxframe)

        # Define main DQN model
        self.policy_net = DQN(self.input_dim, self.action_dim, model_dims, device).to(self.device)
        if load_model:
            self.policy_net.load_state_dict(self.path)
        self.target_net = DQN(self.input_dim, self.action_dim, model_dims, device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Set for training
        self.loss_ftn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay = Replaybuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = torch.tensor(0.99).to(device)

    # Define to take greedy action
    def get_greedy(self, model, s):
        with torch.no_grad():
            q = model(s)
        action = q.max(1)[1]
        # Index corresponds to action
        return action


    # Function to update DQN
    def update_DQN(self):
        self.policy_net.train()
        # load transitions from batch
        batch = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        prev_state = torch.Tensor(batch.prev_state).view(-1, self.input_dim).to(self.device)
        action = torch.LongTensor(batch.action).view(self.batch_size, -1).to(self.device)
        reward = torch.Tensor(batch.reward).view(self.batch_size, -1).to(self.device)
        state = torch.Tensor(batch.state).view(-1, self.input_dim).to(self.device)
        done = torch.Tensor(batch.done).view(self.batch_size, -1).to(self.device)

        # Compute desired target & estimates
        q_estimates = self.policy_net(prev_state).gather(1, action)
        with torch.no_grad():
            target = self.target_net(state).max(1)[0].unsqueeze(1).detach()
        target = reward + self.gamma*target*(1-done)

        # Train DQN
        self.optimizer.zero_grad()
        loss = self.loss_ftn(q_estimates, target)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def train(self, target_update_freq=10000):
        writer = SummaryWriter()
        episode, frame = 1, 1
        score = collections.deque(maxlen=1000)

        with tqdm(total = int(self.maxframe)) as pbar:
            while frame < self.maxframe:
                prev_state = self.env.reset() 
                episode_length = 0
                for i in range(300):
                    s = torch.Tensor(prev_state).view(-1, self.input_dim).to(self.device)
                    if random.random() < self.eps.get_epsilon(frame):
                        action = self.env.action_space.sample()
                    else:
                        action = self.get_greedy(self.policy_net,s).item()

                    state, reward, done, _ = self.env.step(action) # environment progress
                    if done:  
                        reward = -1

                    self.replay.push(prev_state, action, reward, state, done)
                    prev_state = state

                    # Train policy network & Update target network
                    if len(self.replay) > self.batch_size:
                        if frame % 5 == 0:  
                            self.update_DQN()
                    if frame % target_update_freq == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        self.target_net.eval()
                    
                    if frame % (self.maxframe*0.5) == 0:
                        self.save(self.path)

                    episode_length += 1
                    frame += 1
                    pbar.update(1)

                    if done:
                        score.append(episode_length)
                        break

                writer.add_scalar('Reward', episode_length, frame)
                if episode % 1000 == 0:
                    print("Mean score over last 1000 episodes was {}".format(np.mean(score)))
                episode += 1

            # self.save(self.path)

    # Save the model weight
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print("Model weight was saved")
            
    # To test actual play with trained DQN
    def display_play(self, frames, i, save_path): 
        img = plt.imshow(frames[0]) 
        
        def animate(i): 
            img.set_data(frames[i])
        
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(save_path+'/'+self.game_name+f' Example_{i}.gif', fps=30, dpi=100)
        plt.plot()
        

    def play(self, path, save_path, num_episodes=10):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        print("Model loading completed")
        for j in range(num_episodes):
            # state = self.env.reset()
            # for i in range(1, 300):
            #     self.env.render()
            #     s = torch.from_numpy(state).view(-1, self.input_dim).to(self.device)
            #     action = self.get_greedy(self.policy_net,s).item()
            #     state, reward, done, _ = self.env.step(action)
            #     if done:
            #         print("Episode reward : {}".format(i))
            #         break
            # self.env.close()

            state = self.env.reset()
            frames = [self.env.render(mode='rgb_array')]
            for i in range(1, 300):
                frames.append(self.env.render(mode='rgb_array'))
                s = torch.from_numpy(state).view(-1, self.input_dim).to(self.device)
                action = self.get_greedy(self.policy_net,s).item()
                state, reward, done, _ = self.env.step(action)
                if done:
                    print("Episode reward : {}".format(i))
                    break        
            self.env.close()
            self.display_play(frames, j, save_path)
