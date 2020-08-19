import argparse
import subprocess
import time
from torch.utils.tensorboard import SummaryWriter

from models import DQN
from nec_agent import NECAgent

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', default='PongNoFrameskip-v4')
  parser.add_argument('--embedding_size', default=64)
  args = parser.parse_args()
  embedding_model = DQN(args.embedding_size)
  agent = NECAgent(args.env, embedding_model)

  experiment_name = f"{args.env}__{int(time.time())}"
  writer = SummaryWriter(f"runs/{experiment_name}")
  for t in range(100):
    reward = agent.episode(t)
    print("Episode {}\nTotal Reward: {}".format(t, reward))
    agent.update(t)
    writer.add_scalar("Reward", reward, t)
