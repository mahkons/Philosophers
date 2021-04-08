import torch
import numpy as np
import itertools

from networks import Actor, Critic
from params import LR, GAMMA, LAMBDA, CLIP, ENTROPY_COEF, VALUE_COEFF, BATCHES_PER_UPDATE, BATCH_SIZE


P_COUNT = 5
P_DELAY = 100

P_LR = 1e-4

class PPO():
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters(), self.critic.parameters()), LR)


        self.philosophers = list()
        for i in range(P_COUNT):
            self.philosophers.append(Critic(state_dim).to(device))

        self.p_optimizers = [torch.optim.Adam(p.parameters(), lr=P_LR) for p in self.philosophers]
        self.update_cnt = 0

    def _calc_loss(self, state, action, old_log_prob, expected_values, gae):
        new_log_prob, action_distr = self.actor.compute_proba(state, action)
        state_values = self.critic.get_value(state).squeeze(1)

        critic_loss = ((expected_values - state_values) ** 2).mean()

        unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(unclipped_ratio, 1 - CLIP, 1 + CLIP)
        actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()

        entropy_loss = -action_distr.entropy().mean()

        p_loss = 0
        for p in self.philosophers:
            p_state_values = self.critic.get_value(state).squeeze(1)
            p_loss += ((p_state_values - state_values.detach()) ** 2).mean()

        return critic_loss * VALUE_COEFF + actor_loss + entropy_loss * ENTROPY_COEF + p_loss


    def update(self, trajectories):
        trajectories = map(self._compute_lambda_returns_and_gae, trajectories)
        transitions = sum(trajectories, []) # Turn a list of trajectories into list of transitions

        state, action, old_log_prob, target_value, advantage = zip(*transitions)
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        action = torch.from_numpy(np.array(action)).float().to(self.device)
        old_log_prob = torch.from_numpy(np.array(old_log_prob)).float().to(self.device)
        target_value = torch.from_numpy(np.array(target_value)).float().to(self.device)
        advantage = torch.from_numpy(np.array(advantage)).float().to(self.device)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)
            loss = self._calc_loss(state[idx], action[idx], old_log_prob[idx], target_value[idx], advantage[idx])

            self.optimizer.zero_grad()
            for p_optimizer in self.p_optimizers:
                p_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for p_optimizer in self.p_optimizers:
                p_optimizer.step()

        self.update_cnt += 1
        if self.update_cnt % P_DELAY == 0:
            self.critic = self.philosophers[0]
            self.optimizer = self.p_optimizers[0]

            self.philosophers.pop(0)
            self.philosophers.append(Critic(self.state_dim).to(self.device))
            self.p_optimizers.pop(0)
            self.p_optimizers.append(torch.optim.Adam(self.philosophers[-1].parameters(), lr=P_LR))


    def _compute_lambda_returns_and_gae(self, trajectory):
        lambda_returns = []
        gae = []
        last_lr = 0.
        last_v = 0.
        for s, _, r, _ in reversed(trajectory):
            ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
            last_lr = ret
            last_v = self.get_value(s)
            lambda_returns.append(last_lr)
            gae.append(last_lr - last_v)
        
        # Each transition contains state, action, old action probability, value estimation and advantage estimation
        return [(s, a, p, v, adv) for (s, a, _, p), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action, pure_action, log_prob = self.actor.act(state)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], log_prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


