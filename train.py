import pybullet_envs
# Don't forget to install PyBullet!

import gym
import numpy as np
import torch
from torch import nn
import random

from PPO import PPO
from params import  ENV_NAME, MIN_EPISODES_PER_UPDATE, MIN_TRANSITIONS_PER_UPDATE, ITERATIONS, RANDOM_SEED

device = torch.device("cuda")

def evaluate_policy(env, agent, episodes):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, logprob = agent.act(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, logprob))
        s = ns
    return trajectory


def train():
    env = gym.make(ENV_NAME)
    env.seed(RANDOM_SEED)

    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=device)
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    best = 0
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_cnt = 0
        
        sum_reward = 0
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_cnt < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_cnt += len(traj)
            sum_reward += sum([r for _, _, r, _ in traj])
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_cnt

        print("Collected: {}".format(sum_reward / len(trajectories)))

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 20)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            val = np.mean(rewards)
            if val > best:
                best = val
                ppo.save()


def init_random_seeds(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    init_random_seeds(RANDOM_SEED)
    train()
