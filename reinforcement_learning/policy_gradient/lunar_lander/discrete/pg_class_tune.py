import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse

import ray
from ray import tune


# Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int,
                 hidden_size_1: int, hidden_size_2: int):
        super(Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size_1, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size_2, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x


class PolicyGradient(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters

        self.epoch = 0

        self.env = gym.make("LunarLander-v2")

        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.hidden_size_1 = config["hidden_size_1"]
        self.hidden_size_2 = config["hidden_size_2"]
        self.beta = config["beta"]
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size_1=self.hidden_size_1,
                           hidden_size_2=self.hidden_size_2)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.learning_rate)

        self.total_rewards = deque([], maxlen=64)

        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False

        self.write = SummaryWriter(comment=f'_PG_CP_Gamma={self.gamma},'
                                            f'LR={self.learning_rate},'
                                            f'BS={self.batch_size},'
                                            f'NH1={self.hidden_size_1},'
                                            f'NH2={self.hidden_size_2},'
                                            f'BETA={self.beta}')

    def play_episode(self, episode: int):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        state = self.env.reset()
        
        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        # episode loop
        while True:

            # render the environment for the first episode in the epoch
            if not self.finished_rendering_this_epoch:
                self.env.render()

            # get the action logits from the agent - (preferences)
            action_logits = self.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state
            state, reward, done, _ = self.env.step(action=action.cpu().item())

            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            average_rewards = np.concatenate((average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)

            # the episode is over
            if done:

                # increment the episode
                episode += 1

                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.gamma)
                discounted_rewards_to_go -= average_rewards  # baseline - state specific average

                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                #mask = one_hot(episode_actions, num_classes=self.env.action_space.n)
                mask = F.one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True

                return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """
            Calculates the sequence of discounted rewards-to-go.
            Args:
                rewards: the sequence of observed rewards
                gamma: the discount factor
            Returns:
                discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.beta * entropy

        return policy_loss + entropy_bonus, entropy

    def step(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        episode = 0

        # init the epoch arrays
        # used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while episode < self.batch_size:

            # play an episode of the environment
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             episode) = self.play_episode(episode=episode)

            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

        # reset the rendering flag
        self.finished_rendering_this_epoch = False

        # increment the epoch
        self.epoch += 1

        # calculate the loss
        loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                            weighted_log_probs=epoch_weighted_log_probs)

        # zero the gradient
        self.adam.zero_grad()

        # backprop
        loss.backward()

        # update the parameters
        self.adam.step()

        print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}")

        self.writer.add_scalar(tag='Average Return over 64 episodes',
                               scalar_value=np.mean(self.total_rewards),
                               global_step=epoch)

        self.writer.add_scalar(tag='Entropy',
                               scalar_value=entropy,
                               global_step=epoch)

        self.writer.add_scalar(tag='Loss',
                               scalar_value=loss,
                               global_step=epoch)

        return {"epoch_score": np.mean(self.total_rewards)}


    def cleanup(self):
        # close the environment
        self.env.close()

        # close the writer
        self.writer.close()


def main():
    ray.init()

    analysis = tune.run(
        PolicyGradient,
        stop={"epoch_score": 200},
        verbose=1,
        config={
            #"num_epochs": 5000,
            #"batch_size": 64,
            "learning_rate": tune.loguniform(0.001, 0.1),
            "hidden_size_1": tune.uniform(2, 128),
            "hidden_size_2": tune.uniform(2, 128),
            #"gamma": 0.99,
            #"beta": 0.1
        })

    print("Best config is:",
        analysis.get_best_config(metric="epoch_score", mode="max"))


if __name__ == "__main__":
    main()
