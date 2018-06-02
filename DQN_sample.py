import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import paperio.match_core_ as core
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

PlayerFunc = namedtuple('PlayerFunc', ('play'))

class ReplayMemory(object):
    def __init__(self, cap):
        self.capacity = cap
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Target Minimization Function:
# d = Q(s,a) - (g(s, s') + r * max_a'(Q(s', a')))
# Loss Function: Huber Loss
# L = 1/|B| Sum(L(d); d in B), where B is the sapmle from replay memory
# L(d) = 1/2 d^2 if |d| < 1 else |d| - 1/2

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(3200, 3) # ???
        # out : Left, Straight, Right

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_DECAY = 200
EPS_END = 0.05
TARGET_UPDATE = 1

policy_net = DQN().to(device)

# load parameters, if exists
try:
    with open("model.dat", "r") as f:
        policy_net.load_state_dict(torch.load(f))
except:
    pass


target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def plain_select_action(state):
    # print(state.shape)
    # print(state)
    with torch.no_grad():
        return target_net(state).argmax(dim=1).view(1, 1)

def e_select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        return plain_select_action(state)
    else:
        return torch.tensor([[random.randrange(3)]], device=device)

def simple_reward(state):
    fields = state.eq(1).sum(dtype=torch.float32).to(device)
    length = state.eq(2).sum(dtype=torch.float32).to(device)
    return fields / 3300 - length / 100

if 'comm_helper':
    def get_screen(id): # id in (1, 2): the player id
        fields = torch.tensor(core.FIELDS, dtype=torch.float32, device=device)
        fields_self = fields.eq(id) # 1 if is, otherwise not
        fields_opp = fields.eq(3 - id)
        bands = torch.tensor(core.BANDS, dtype=torch.float32, device=device)
        bands_self = bands.eq(id)
        bands_opp = bands.eq(3 - id)
        state = fields_self + fields_opp * 2 + bands_self * 3 + bands_opp * 4
        # 1 for fields of self id, 2 for bands of self id, 3 for fields of opp id, 4 for fields of opp id, 0 for none
        return state.unsqueeze(0).unsqueeze(0) # shaped (1, 1, W, H)

    def normalize(screen):
        return [[0 if i is None else i for i in l] for l in screen]

    def parse_frame(fields, bands, id): # id in (1, 2)

        players_pos = [(core.PLAYERS[i - 1].x, core.PLAYERS[i - 1].y) for i in range(2)]
        fields = normalize(fields)
        bands = normalize(bands)

        fields = torch.tensor(fields, dtype=torch.float32, device=device)
        fields_self = fields.eq(id) # 1 if is, otherwise not

        fields_opp = fields.eq(3. - id)
        bands = torch.tensor(bands, dtype=torch.float32, device=device)
        bands_self = bands.eq(id)
        bands_opp = bands.eq(3 - id)
        state = torch.tensor(fields_self + fields_opp * 2. + bands_self * 4. + bands_opp * 5.,
                             dtype=torch.float32)
        state[players_pos[id - 1][0]][players_pos[id - 1][1]] = 3. # id -> id-1 th
        state[players_pos[2 - id][0]][players_pos[2 - id][1]] = 6. # 3-id -> 2-id th
        return state.unsqueeze(0).unsqueeze(0)

    def target_net_wrapper(id): # id in (0, 1)
        def func(stat, storage):
            global parse_frame

            try:
                func.count += 1
            except:
                func.count = 1


            f, b = storage['log'][-1]['fields'], storage['log'][-1]['bands']
            state = parse_frame(f, b, id + 1) # last screen
            choice = int(e_select_action(state))
            # print(id, "has gone through", func.count, "frames, choosing", 'LXR'[choice])
            return 'LXR'[choice]

        return func

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Area')
    # plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    """
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    """
    plt.plot(episode_durations)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():

    # wait until memory is filled
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).view(-1, 1, 102, 101)
    # print(state_batch.shape)
    action_batch = torch.cat(batch.action)
    # print(action_batch.shape)

    # g(s, s')
    reward_batch = torch.cat(batch.reward)
    # print(reward_batch.shape)

    # masks off all states, that is next to final
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    # Shaped (BS_non_final)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).squeeze(1)
    # Shaped (BS_non_final, 1, W, H)
    # print(non_final_next_states.shape)

    # Q(s, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    # print(state_action_values.shape)

    # max_a'[Q(s', a')] computations
    next_state_values = torch.zeros(BATCH_SIZE, device=device)#.view(-1, 1) # shaped (BatchSize, 1)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # print(next_state_values.shape)
    # target Q = r * Q(s', a') + g(s, s')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    t = time.time()
    optimizer.zero_grad()
    loss.backward()
    t = time.time() - t
    print("This episode takes", t, "seconds")
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

if __name__ == "__main__":
    num_episodes = 1000

    for i_episode in range(num_episodes):

        # reinitialize fields
        core.init_field(51, 101, 3000, 0xFFFF)
        player_funcs = [PlayerFunc(play=target_net_wrapper(i)) for i in (0, 1)] # i in (0, 1)

        # state = get_screen(1)

        result = core.parse_match(player_funcs)
        logs = core.LOG_PUBLIC # records each frame in 0th player's view.
        history_len = len(logs)
        print(torch.cat(logs).shape, result)

        for i_frame in range(history_len):
            if i_frame % 2 == 0: # only process those of first agent.
                state, action = logs[i_frame].unsqueeze(0), core.OPS[i_frame]
                try:
                    next_state = logs[i_frame + 2].unsqueeze(0)
                    done = False
                except:
                    next_state = None
                    done = True

                if action == 'L':
                    action = torch.tensor([[0]], device=device)
                elif action == 'R':
                    action = torch.tensor([[2]], device=device)
                else:
                    action = torch.tensor([[1]], device=device)

                reward = torch.tensor([0.], dtype=torch.float32, device=device)

                if i_frame == history_len - 1: # killed opp
                    if result[1] == 1: # 如果不是对面自杀
                        reward += 3.
                elif i_frame == history_len - 2: # killed by opp
                    if result[1] == 1.5:
                        reward -= 3. # 自杀
                    reward += -3.
                else:
                    reward += simple_reward(state)

                # t = (action, reward)
                # print(t)
                memory.push(state, action, next_state, reward)

                optimize_model()

            else:
                continue

        episode_durations.append(logs[-1].eq(1).sum()) # append fields area.
        plot_durations()

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # saves model
    with open('model.dat', 'w') as f:
        torch.save(policy_net.state_dict(), f)

    print('Complete')
    # env.render()
    # env.close()
    plt.ioff()
    plt.show()























