import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from pyflann import FLANN

class DQN(nn.Module):
  def __init__(self, embedding_size):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    self.fc = nn.Linear(2592, 256)
    self.head = nn.Linear(256, embedding_size)

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out

class DND:
  def __init__(self, kernel, num_neighbors, max_memory, lr):
    self.kernel = kernel
    self.num_neighbors = num_neighbors
    self.max_memory = max_memory
    self.lr = lr
    self.keys = None
    self.values = None
    self.kdtree = FLANN()

    self.key_cache = {}
    self.stale_index = True
    self.indexes_to_be_updated = set()
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None

    # Move recently used lookup indexes
    self.move_to_back = set()

  def get_index(self, key):
    if self.key_cache.get(tuple(key.data.cpu().numpy()[0])) is not None:
      if self.stale_index:
        self.commit_insert()
      return int(self.kdtree.nn_index(key.data.cpu().numpy(), 1)[0][0])
    else:
      return None

  def update(self, value, index):
    values = self.values.data
    values[index] = value[0].data
    self.values = Parameter(values)
    self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)

  def insert(self, key, value):
    if self.keys_to_be_inserted is None:
      # Initial insert
      self.keys_to_be_inserted = key.data
      self.values_to_be_inserted = value.data
    else:
      self.keys_to_be_inserted = torch.cat(
          [self.keys_to_be_inserted, key.data], 0)
      self.values_to_be_inserted = torch.cat(
          [self.values_to_be_inserted, value.data], 0)
    self.key_cache[tuple(key.data.cpu().numpy()[0])] = 0
    self.stale_index = True

  def commit_insert(self):
    if self.keys is None:
      self.keys = Parameter(self.keys_to_be_inserted)
      self.values = Parameter(self.values_to_be_inserted)
    elif self.keys_to_be_inserted is not None:
      self.keys = Parameter(
          torch.cat([self.keys.data, self.keys_to_be_inserted], 0))
      self.values = Parameter(
          torch.cat([self.values.data, self.values_to_be_inserted], 0))

    # Move most recently used key-value pairs to the back
    if len(self.move_to_back) != 0:
      self.keys = Parameter(torch.cat([self.keys.data[list(set(range(len(
          self.keys))) - self.move_to_back)], self.keys.data[list(self.move_to_back)]], 0))
      self.values = Parameter(torch.cat([self.values.data[list(set(range(len(
          self.values))) - self.move_to_back)], self.values.data[list(self.move_to_back)]], 0))
      self.move_to_back = set()

    if len(self.keys) > self.max_memory:
      # Expel oldest key to maintain total memory
      for key in self.keys[:-self.max_memory]:
        del self.key_cache[tuple(key.data.cpu().numpy())]
      self.keys = Parameter(self.keys[-self.max_memory:].data)
      self.values = Parameter(self.values[-self.max_memory:].data)
    self.keys_to_be_inserted = None
    self.values_to_be_inserted = None
    self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)
    self.kdtree.build_index(self.keys.data.cpu().numpy())
    self.stale_index = False

  def lookup(self, key, update=False):
    """
      If update == True, add the nearest neighbor indexes to self.indexes_to_be_updated
    """
    lookup_indexes = self.kdtree.nn_index(
        key.data.cpu().numpy(), min(self.num_neighbors, len(self.keys)))[0][0]
    output = 0
    kernel_sum = 0
    for i, index in enumerate(lookup_indexes):
      if i == 0 and self.key_cache.get(tuple(key[0].data.cpu().numpy())) is not None:
        # If a key exactly equal to key is used in the DND lookup calculation
        # then the loss becomes non-differentiable. Just skip this case to avoid the issue.
        continue
      if update:
        self.indexes_to_be_updated.add(int(index))
      else:
        self.move_to_back.add(int(index))
      kernel_val = self.kernel(self.keys[int(index)], key[0])
      output += kernel_val * self.values[int(index)]
      kernel_sum += kernel_val
    output = output / kernel_sum
    return output

  def update_params(self):
    """
    Update self.keys and self.values via backprop
    Use self.indexes_to_be_updated to update self.key_cache accordingly and rebuild the index of self.kdtree
    """
    for index in self.indexes_to_be_updated:
      del self.key_cache[tuple(self.keys[index].data.cpu().numpy())]
    self.optimizer.step()
    self.optimizer.zero_grad()
    for index in self.indexes_to_be_updated:
      self.key_cache[tuple(self.keys[index].data.cpu().numpy())] = 0
    self.indexes_to_be_updated = set()
    self.kdtree.build_index(self.keys.data.cpu().numpy())
    self.stale_index = False
