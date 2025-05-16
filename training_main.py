import os
from training_metrics_logger import init_csv_log, log_epoch_metrics
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # This disables torch._dynamo and avoids importing SymPy
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine
from tqdm import tqdm
import time
import tracemalloc

# Training settings
epochs = 500
folder_name = f"param_{epochs}"
os.makedirs(folder_name, exist_ok=True)

# Argument parsing
parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--action-repeat', type=int, default=8)
parser.add_argument('--img-stack', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--render', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--mode', type=str, default='dense32', choices=['dense32', 'dense64', 'sparse32', 'sparse64'])
args = parser.parse_args()

# Mode setup
mode = args.mode
use_sparse = 'sparse' in mode
use_float64 = '64' in mode
dtype = torch.float64 if use_float64 else torch.float32

# Device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Transition buffer format
transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)),
                       ('a_logp', np.float64), ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])

class Env():
    def __init__(self):
        self.env = gym.make('CarRacing-v2', domain_randomize=True)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb = self.env.reset()[0]
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(args.action_repeat):
            img_rgb, reward, die, _, _ = self.env.step(action)
            total_reward += reward
            if self.av_r(reward) <= -0.1 or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        return np.array(self.stack), total_reward, self.av_r(reward) <= -0.1, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        return gray / 128. - 1. if norm else gray

    @staticmethod
    def reward_memory():
        history = np.zeros(100)
        count = 0
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % 100
            return np.mean(history)
        return memory

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU()
        )
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        return (self.alpha_head(x) + 1, self.beta_head(x) + 1), v

class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().to(dtype=dtype).to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.max_moving_average_score = -np.inf

    def select_action(self, state):
        tensor = torch.from_numpy(state).to(dtype=dtype).to(device)
        if use_sparse:
            print("!!1 Sparse mode is enabled, but CNN layers do not support sparse input. Skipping sparsity.")
            with open(f"output_{mode}.log", "a") as log:
                log.write("Sparse mode is enabled, but CNN layers do not support sparse input. Skipping sparsity.\n")
            # NOTE: Do not return here! Let execution continue.
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(tensor)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        return action.squeeze().cpu().numpy(), dist.log_prob(action).sum().item()

    def save_param(self, epoch=None):
        torch.save(self.net.state_dict(), f'{folder_name}/{folder_name}_ppo_net_params.pkl')

    def save_max_moving_average_param(self, score):
        if score > self.max_moving_average_score:
            self.max_moving_average_score = score
            torch.save(self.net.state_dict(), f'{folder_name}/{folder_name}_ppo_net_params_max_mov_avergae.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False

    def update(self):
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=dtype).to(device)
        a = torch.tensor(self.buffer['a'], dtype=dtype).to(device)
        r = torch.tensor(self.buffer['r'], dtype=dtype).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=dtype).to(device)
        old_logp = torch.tensor(self.buffer['a_logp'], dtype=dtype).to(device).view(-1, 1)
        if use_sparse:
            print("!!2") #Sparse mode is enabled, but CNN layers do not support sparse input. Skipping sparsity.")
            with open(f"output_{mode}.log", "a") as log:
                log.write("Sparse mode is enabled, but CNN layers do not support sparse input. Skipping sparsity.\n")
        
        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(logp - old_logp[index])
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                loss = -torch.min(surr1, surr2).mean() + 2. * F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()

    agent = Agent()
    logfile = init_csv_log(mode)
    env = Env()
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    running_score = 0
    for i_ep in tqdm(range(epochs)):
        score = 0
        state = env.reset()
        for _ in range(1000):
            action, a_logp = agent.select_action(state)
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, next_state)):
                agent.update()
            score += reward
            state = next_state
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        log_epoch_metrics(logfile, i_ep, score, running_score)
        agent.save_max_moving_average_param(running_score)
        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            with open(f"logs_{folder_name}.txt", "a") as f:
                f.write(f"Ep {i_ep}\tLast score: {score:.2f}\tMoving average score: {running_score:.2f}\n")
            agent.save_param()
            if i_ep % 500 == 0:
                agent.save_param(i_ep)


        if running_score > env.reward_threshold:
            print(f"Solved! Running reward is now {running_score:.2f} and the last episode runs to {score:.2f}")
            agent.save_param()
        # Write final metrics summary
        """
        with open(f"metrics0_{mode}.txt", "w") as f:
            f.write(f"Training time (s): {end_time - start_time:.2f}\\n")
            f.write(f"CPU Current Memory (MB): {current / 1e6:.2f}\\n")
            f.write(f"CPU Peak Memory (MB): {peak / 1e6:.2f}\\n")
            f.write(f"CPU RSS (MB): {cpu_rss:.2f}\\n")
            f.write(f"GPU Allocated Memory (MB): {gpu_allocated:.2f}\\n")
            f.write(f"GPU Reserved Memory (MB): {gpu_reserved:.2f}\\n")
            f.write(f"GPU Max Allocated Memory (MB): {gpu_max_allocated:.2f}\\n")
            f.write(f"GPU Max Reserved Memory (MB): {gpu_max_reserved:.2f}\\n")
            f.write(f"Final moving average score: {running_score:.2f}\\n")
            break
        """

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e6
        gpu_reserved = torch.cuda.memory_reserved() / 1e6
        gpu_max_allocated = torch.cuda.max_memory_allocated() / 1e6
        gpu_max_reserved = torch.cuda.memory_reserved() / 1e6
    else:
        gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_max_reserved = 0.0
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_rss = process.memory_info().rss / 1e6
    except ImportError:
        cpu_rss = 0.0
    
    with open(f"metrics_{mode}.txt", "w") as f:
        f.write(f"Training time (s): {end_time - start_time:.2f}\n")
        f.write(f"CPU Current Memory (MB): {current / 1e6:.2f}\n")
        f.write(f"CPU Peak Memory (MB): {peak / 1e6:.2f}\n")
        f.write(f"CPU RSS (MB): {cpu_rss:.2f}\n")
        f.write(f"GPU Allocated Memory (MB): {gpu_allocated:.2f}\n")
        f.write(f"GPU Reserved Memory (MB): {gpu_reserved:.2f}\n")
        f.write(f"GPU Max Allocated Memory (MB): {gpu_max_allocated:.2f}\n")
        f.write(f"GPU Max Reserved Memory (MB): {gpu_max_reserved:.2f}\n")
        f.write(f"Final moving average score: {running_score:.2f}\n")