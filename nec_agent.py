import random
import torch
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable

import einops

import numpy as np

from atari_wrapper import make_atari, wrap_deepmind

from models import DND

use_cuda = torch.cuda.is_available()


def move_to_gpu(var):
  if use_cuda:
    return var.cuda()
  else:
    return var

import scipy.signal as signal


def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def inverse_distance(h, h_i, epsilon=1e-3):
  return 1 / (torch.dist(h, h_i) + epsilon)

import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward'))
ReplayMemoryUnit = namedtuple('MemoryUnit', ('state', 'action', 'Q_N'))


class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = ReplayMemoryUnit(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class NECAgent:
  def __init__(self,
               env,
               embedding_network,
               replay_memory=ReplayMemory(100000),
               initial_epsilon=1.0,
               final_epsilon=0.01,
               epsilon_decay=0.99,
               batch_size=8,
               sgd_lr=1e-6,
               q_lr=0.01,
               gamma=0.99,
               lookahead_horizon=100,
               update_period=4,
               kernel=inverse_distance,
               num_neighbors=50,
               max_memory=500000):
    """
    Instantiate an NEC Agent

    Parameters
    ----------
    env: gym.Env
      gym environment to train on
    embedding_network: torch.nn.Module
      Model to extract the embedding from a state
    replay_memory: ReplayMemory
      Replay memory to sample from for embedding network updates
    initial_epsilon: float
      Initial epsilon for epsilon greedy search
    epsilon_decay: float
      Exponential decay factor for epsilon
    batch_size: int
      Batch size to sample from the replay memory
    sgd_lr: float
      Learning rate to use for RMSProp updates to the embedding network and DND
    q_lr: float
      Learning rate to use for Q-updates on DND updates
    gamma: float
      Discount factor
    lookahead_horizon: int
      Lookahead horizon to use for N-step Q-value estimates
    update_period: int
      Inverse of rate at which embedding network gets updated
      i.e. if 1 then update after every timestep, if 16 then update every 16 timesteps, etc.
    kernel: (torch.autograd.Variable, torch.autograd.Variable) => (torch.autograd.Variable)
      Kernel function to use for DND lookups
    num_neighbors: int
      Number of neighbors to return in K-NN lookups in DND
    max_memory: int
      Maximum number of key-value pairs to store in each DND
    """

    self.env = wrap_deepmind(make_atari(env), scale=True, frame_stack=True)
    self.embedding_network = embedding_network
    if use_cuda:
      self.embedding_network.cuda()

    self.replay_memory = replay_memory
    self.epsilon = initial_epsilon
    self.final_epsilon = final_epsilon
    self.epsilon_decay = epsilon_decay
    self.batch_size = batch_size
    self.q_lr = q_lr
    self.gamma = gamma
    self.lookahead_horizon = lookahead_horizon
    self.update_period = update_period

    self.transition_queue = []
    self.optimizer = optim.RMSprop(
        self.embedding_network.parameters(), lr=sgd_lr)
    self.dnd_list = [DND(kernel, num_neighbors, max_memory, sgd_lr)
                     for _ in range(self.env.action_space.n)]

  def choose_action(self, state_embedding):
    """
    Choose action from epsilon-greedy policy
    If not a random action, choose the action that maximizes the Q-value estimate from the DNDs
    """
    if random.uniform(0, 1) < self.epsilon:
      return random.randint(0, self.env.action_space.n - 1)
    else:
      q_estimates = [dnd.lookup(state_embedding) for dnd in self.dnd_list]
      action = torch.cat(q_estimates).max(0)[1].item()
      return action

  def Q_lookahead(self, t, warmup=False):
    """
    Return the N-step Q-value lookahead from time t in the transition queue
    """
    if warmup or len(self.transition_queue) <= t + self.lookahead_horizon:
      lookahead = discount(
          [transition.reward for transition in self.transition_queue[t:]], self.gamma)[0]
      return Variable(Tensor([lookahead]))
    else:
      lookahead = discount(
          [transition.reward for transition in self.transition_queue[t:t+self.lookahead_horizon]], self.gamma)[0]
      state = self.transition_queue[t+self.lookahead_horizon].state
      state_embedding = self.embedding_network(
          move_to_gpu(Variable(Tensor(state)).unsqueeze(0)))
      return self.gamma ** self.lookahead_horizon * torch.cat([dnd.lookup(state_embedding) for dnd in self.dnd_list]).max() + lookahead

  def Q_update(self, q_initial, q_n):
    """
    Return the Q-update for DND updates
    """
    return q_initial + self.q_lr * (q_n - q_initial)

  def update(self):
    """
    Iterate through the transition queue and make NEC updates
    """
    for t in range(len(self.transition_queue)):
      transition = self.transition_queue[t]
      state = Variable(Tensor(np.array(transition.state)))
      state = einops.rearrange(state, 'b h w c ->  b c h w')
      action = transition.action
      state_embedding = self.embedding_network(move_to_gpu(state))
      dnd = self.dnd_list[action]

      Q_N = move_to_gpu(self.Q_lookahead(t))
      embedding_index = dnd.get_index(state_embedding)
      if embedding_index is None:
        dnd.insert(state_embedding.detach(), Q_N.detach().unsqueeze(0))
      else:
        Q = self.Q_update(dnd.values[embedding_index], Q_N)
        dnd.update(Q.detach(), embedding_index)
      self.replay_memory.push(transition.state, action,
                              move_to_gpu(Q_N.detach()))

    [dnd.commit_insert() for dnd in self.dnd_list]

    for t in range(len(self.transition_queue)):
      if t % self.update_period == 0 or t == len(self.transition_queue) - 1:
        # Train on random mini-batch from self.replay_memory
        batch = self.replay_memory.sample(self.batch_size)
        actual = torch.cat([sample.Q_N for sample in batch])
        predicted = torch.cat([self.dnd_list[sample.action].lookup(self.embedding_network(move_to_gpu(
            Variable(Tensor(sample.state))).unsqueeze(0)), update_flag=True) for sample in batch])
        loss = torch.dist(actual, move_to_gpu(predicted))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        [dnd.update_params() for dnd in self.dnd_list]

    # Clear out transition queue
    self.transition_queue = []

  def episode(self):
    """
    Train an NEC agent for a single episode
    Interact with environment on-policy and append all (state, action, reward) transitions to transition queue
    Call update at the end of every episode
    """
    if self.epsilon > self.final_epsilon:
      self.epsilon = self.epsilon * self.epsilon_decay
    state = self.env.reset()
    total_reward = 0
    done = False
    while not done:
      state = Variable(Tensor(np.array(state))).unsqueeze(0)
      print(state.shape)
      state = einops.rearrange(state, 'b h w c ->  b c h w')
      state_embedding = self.embedding_network(move_to_gpu(state))
      action = self.choose_action(state_embedding)
      next_state, reward, done, _ = self.env.step(action)
      self.transition_queue.append(Transition(state, action, reward))
      total_reward += reward
      state = next_state
    self.update()
    return total_reward

  def warmup(self):
    """
    Warmup the DND with values from an episode with a random policy
    """
    state = self.env.reset()
    total_reward = 0
    done = False
    while not done:
      action = random.randint(0, self.env.action_space.n - 1)
      next_state, reward, done, _ = self.env.step(action)
      total_reward += reward
      self.transition_queue.append(Transition(state, action, reward))
      state = next_state

    for t in range(len(self.transition_queue)):
      transition = self.transition_queue[t]
      state = Variable(Tensor(np.array(transition.state))).unsqueeze(0)
      state = einops.rearrange(state, 'b h w c ->  b c h w')
      action = transition.action
      state_embedding = self.embedding_network(move_to_gpu(state))
      dnd = self.dnd_list[action]

      Q_N = move_to_gpu(self.Q_lookahead(t, True))
      if dnd.keys_to_be_inserted is None and dnd.keys is None:
        dnd.insert(state_embedding, Q_N.detach().unsqueeze(0))
      else:
        embedding_index = dnd.get_index(state_embedding)
        if embedding_index is None:
          dnd.insert(state_embedding.detach(), Q_N.detach().unsqueeze(0))
        else:
          Q = self.Q_update(dnd.values[embedding_index], Q_N)
          dnd.update(Q.detach(), embedding_index)
      self.replay_memory.push(transition.state, action, Q_N)
    [dnd.commit_insert() for dnd in self.dnd_list]
    # Clear out transition queue
    self.transition_queue = []
    return total_reward
