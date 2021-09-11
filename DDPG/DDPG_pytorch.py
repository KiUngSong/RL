import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
import random, collections
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Define Replaybuffer
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


# Defined to initialize each layer
def fan_in_init(tensor, fan_in=None):
    fan_in = fan_in or tensor.size(-1)
    w = 1 / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


# Define Actor model : Policy function μ(a)
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, actor_dims, action_max):
        super(Actor, self).__init__()
        linear = nn.Linear(input_dim, actor_dims[0])
        fan_in_init(linear.weight)
        fan_in_init(linear.bias)
        model = [linear, nn.ReLU()]

        for i in range(len(actor_dims)-1):
            linear = nn.Linear(actor_dims[i], actor_dims[i+1])
            fan_in_init(linear.weight)
            fan_in_init(linear.bias)
            model += [linear, nn.ReLU()]
        self.model = nn.Sequential(*model)

        fc = nn.Linear(actor_dims[-1], action_dim)
        nn.init.uniform_(fc.weight, -3e-3, 3e-3)
        nn.init.uniform_(fc.bias, -3e-4, 3e-4)
        self.fc = nn.Sequential(fc, nn.Tanh())
        self.action_max = action_max

    def forward(self, x):
        x = self.model(x)
        x = self.action_max * self.fc(x)
        return x


# Define Critic model : Action-Value function Q(s,a)
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, critic_dims):
        super(Critic, self).__init__()
        linear = nn.Linear(input_dim, critic_dims[0])
        fan_in_init(linear.weight)
        fan_in_init(linear.bias)
        self.init = nn.Sequential(linear, nn.ReLU())

        linear = nn.Linear(critic_dims[0] + action_dim, critic_dims[1])
        fan_in_init(linear.weight)
        fan_in_init(linear.bias)
        model = [linear, nn.ReLU()]

        for i in range(1, len(critic_dims)-1):
            linear = nn.Linear(critic_dims[i], critic_dims[i+1])
            fan_in_init(linear.weight)
            fan_in_init(linear.bias)
            model += [linear, nn.ReLU()]
        self.model = nn.Sequential(*model)
        
        self.fc = nn.Linear(critic_dims[-1], 1)
        nn.init.uniform_(self.fc.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc.bias, -3e-4, 3e-4)

    def forward(self, state, action):
        x = self.init(state)
        x = torch.cat([x, action], dim=1)
        x = self.model(x)
        x = self.fc(x)
        return x


# Define class to Compute Ornstein-Uhlenbeck(OU) Noise which is added to action value
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu

    def get_noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


# Define class to execute total process : network update & train & test with actual play
class Trainer(object):
    def __init__(self, actor_dims, critic_dims, game_name, path, load_model, 
                    batch_size, device, buffer_size, maxframe, verbose_freq, 
                    tau=0.001, tau_ratio=5, action_ratio=0.2):
        self.game_name = game_name
        self.env = gym.make(game_name) # e.g. 'MountainCarContinuous-v0'
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = float(self.env.action_space.high[0])
        self.path = path
        self.device = device
        self.maxframe = maxframe
        self.verbose_freq = verbose_freq

        # Define main Actor & Critic model
        self.actor = Actor(self.state_dim, self.action_dim, actor_dims, self.action_dim).to(device)
        if load_model:
            self.actor.load_state_dict(self.path)
        self.actor_target = Actor(self.state_dim, self.action_dim, actor_dims, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim, critic_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set loss & many hyperparamters for training
        self.critic_loss = nn.MSELoss()
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)
        self.replay = Replaybuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = tau
        self.tau_critic = tau_ratio * tau # Use different tau for actor & critic network respectively
        self.exploration_noise = 0.1
        self.OU = OUNoise(action_dim=self.action_dim)
        self.max_iter_frame = 500 # Max iteration per each episode
        self.action_ratio = action_ratio # Multiplied action_ration to action value to encourage more exploration

    def update_DDPG(self):
        # load transitions from batch
        batch = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        prev_state = torch.Tensor(batch.prev_state).view(-1, self.state_dim).to(self.device)
        state = torch.Tensor(batch.state).view(-1, self.state_dim).to(self.device)
        action = torch.LongTensor(batch.action).view(self.batch_size, -1).to(self.device)
        reward = torch.Tensor(batch.reward).view(self.batch_size, -1).to(self.device)
        done = torch.Tensor(batch.done).view(self.batch_size, -1).to(self.device)

        # Compute the target & estimates Q value
        target_q = self.critic_target(state, self.actor_target(state))
        target_q = reward + (1-done) * self.gamma * target_q
        estimates_q = self.critic(prev_state, action)

        # Update Critic
        self.optim_critic.zero_grad()
        critic_loss = self.critic_loss(estimates_q, target_q.detach())
        critic_loss.backward()
        self.optim_critic.step()

        # Compute actor loss & update Actor
        self.optim_actor.zero_grad()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_loss.backward()
        self.optim_actor.step()

        # Update target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau_critic * param.data + (1-self.tau_critic) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
    
    def actor_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float, device=self.device)
        return self.actor(state).data.cpu().numpy().flatten()

    def train(self):
        writer = SummaryWriter()
        episode, frame = 1, 1
        score = collections.deque(maxlen=1000)
        # decay_rate = 1
        # temp = 1 / (0.5 * self.maxframe)

        with tqdm(total = int(self.maxframe)) as pbar:
            while frame < self.maxframe:
                prev_state = self.env.reset() 
                epi_score = 0
                for i in range(1000):
                    s = torch.Tensor(prev_state).view(-1, self.state_dim).to(self.device)
                    action = self.actor_action(s)
                    # Add noise to make action to encourage exploration
                    action = action * self.action_ratio + self.OU.get_noise() #* max(decay_rate, 0)
                    action = action.clip(self.env.action_space.low, self.env.action_space.high)
                    
                    # decay_rate -= temp

                    # environment progress
                    state, reward, done, _ = self.env.step(action)
                    epi_score += reward

                    self.replay.push(prev_state, action, reward, state, done)
                    prev_state = state
                    
                    if frame % (self.maxframe*0.5) == 0:
                        self.save(self.path)

                    frame += 1
                    pbar.update(1)

                    # Update Actor & Critic at the during episode
                    if len(self.replay) > self.batch_size:
                        self.update_DDPG()

                    if self.max_iter_frame <= (i + 1):
                        done = True

                    if done:
                        score.append(epi_score)
                        break

                # Save result of episode to tensorboard
                writer.add_scalar('Reward', epi_score, frame)

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
                action = self.actor_action(s)
                state, reward, done, _ = self.env.step(action)
                epi_score += reward
                if done:
                    print(f"Episode reward : {epi_score:.3f}")
                    break        
            self.env.close()
            self.display_play(frames, j, save_path)
