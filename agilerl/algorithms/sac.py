import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from itertools import chain

from typing import Optional, Any

from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.networks.evolvable_cnn import EvolvableCNN


class SAC():
    """The SAC algorithm class. SAC paper: https://arxiv.org/pdf/1812.05905v2.pdf

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 1e-3
    :type tau: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param policy_freq: Frequency of target network updates compared to policy network, defaults to 2
    :type policy_freq: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        one_hot: bool,
        index: int = 0,
        net_config: dict[str, Any] = {
            'arch': 'mlp',
            'h_size': [
            64,
            64]},
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: float = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        alpha: float = 1.0,
        log_std_min: float = -20,
        log_std_max: float = 2,
        mutation: str = None,
        policy_freq: int = 2,
        device: str = 'cpu',
        auto_entropy: bool = True):
        
        
        self.algo = 'SAC'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mut = mutation
        self.policy_freq = policy_freq
        self.device = device
        self.auto_entropy = auto_entropy

        self.index = index
        
        # What type gets stored inside of these lists?
        # Where are these lists added to
        self.scores = []
        self.fitness: list[float] = []
        self.steps = [0]

        # model
        if self.net_config['arch'] == 'mlp':      # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim * 2,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)

            self.critic1 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.critic1_target = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.critic1_target.load_state_dict(self.critic.state_dict())
            
            self.critic2 = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.critic2_target = EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.critic2_target.load_state_dict(self.critic.state_dict())

        elif self.net_config['arch'] == 'cnn':    # Convolutional Neural Network
            self.actor = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                mlp_activation='tanh',
                device=self.device).to(
                self.device)

            self.critic1 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                mlp_activation='tanh',
                critic=True,
                device=self.device).to(
                self.device)
            self.critic1_target = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                mlp_activation='tanh',
                critic=True,
                device=self.device).to(
                self.device)
            self.critic1_target.load_state_dict(self.critic.state_dict())
            
            self.critic2 = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                mlp_activation='tanh',
                critic=True,
                device=self.device).to(
                self.device)
            self.critic2_target = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                mlp_activation='tanh',
                critic=True,
                device=self.device).to(
                self.device)
            self.critic2_target.load_state_dict(self.critic.state_dict())
        else:
            raise ValueError(f"{self.__class__}::__init__ - Improper architecture framework")
        
        if self.auto_entropy:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim.shape))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # Combine the tunable critic parameters so can optimize in one step
        critic_params = chain(self.critic1.parameters(), self.critic2.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=self.lr)
        self.criterion = nn.MSELoss()

    def getAction(self, state: np.ndarray, epsilon = None)->np.ndarray:
        """Returns the next action to take in the environment.

        :param state: Environment observation, or multiple observations in a batch
        :type state: float or List[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
        state = torch.from_numpy(state).float().to(self.device)

        if self.one_hot:
            state = nn.functional.one_hot(
                state.long(), num_classes=self.state_dim[0]).float().squeeze()

        if len(state.size()) < 2:
            state = state.unsqueeze(0)       

        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()

        # Concern here on logic if action_space is multi-dimensional
        # May need to test on environment with multi-dimensional action space
        action_values = action_values[:self.action_dim]

        action = action_values.cpu().data.numpy()

        return action
    
    
    def sampleAction(self, actor_output: torch.Tensor, training: bool = True, epsilon = 1e-6):
        # Concern here if action dimension is multidimensional
        mean = actor_output[:self.action_dim]
        log_std = actor_output[self.action_dim:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        action_distribution = Normal(mean, log_std)
        if training:
            action = action_distribution.rsample()
        else:
            action = mean
            
        action_prob = (action_distribution.log_prob(action) - torch.log(1-(torch.tanh(action)).pow(2) + epsilon)).sum(1, keepdim=True)
        
        action = torch.tanh(action)
        
        return action, action_prob
        
        
        

    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, rewards, next_states, dones in that order.
        :type experience: List[torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        states, actions, rewards, next_states, dones = experiences

        if self.one_hot:
            states = nn.functional.one_hot(
                states.long(), num_classes=self.state_dim[0]).float().squeeze()
            next_states = nn.functional.one_hot(
                next_states.long(), num_classes=self.state_dim[0]).float().squeeze()

        if self.net_config['arch'] == 'mlp':
            input_combined = torch.cat([states, actions], 1)
            q_value = self.critic(input_combined)
        elif self.net_config['arch'] == 'cnn':
            q_value = self.critic(states, actions)

        next_actions = self.actor(next_states)
        noise = actions.data.normal_(0, policy_noise).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_actions = (next_actions + noise)

        if self.net_config['arch'] == 'mlp':
            next_input_combined = torch.cat([next_states, next_actions], 1)
            q_value_next_state = self.critic_target(next_input_combined)
        elif self.net_config['arch'] == 'cnn':
            q_value_next_state = self.critic(next_states, next_actions)

        y_j = rewards + (self.gamma * q_value_next_state).detach()

        critic_loss = self.criterion(q_value, y_j)

        # critic loss backprop
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor and targets every policy_freq episodes
        if len(self.scores) % self.policy_freq == 0:
            if self.net_config['arch'] == 'mlp':
                input_combined = torch.cat(
                    [states, self.actor.forward(states)], 1)
                actor_loss = -self.critic(input_combined).mean()
            elif self.net_config['arch'] == 'cnn':
                actor_loss = - \
                    self.critic(states, self.actor.forward(states)).mean()

            # actor loss backprop
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.softUpdate(self.actor, self.actor_target)
            self.softUpdate(self.critic, self.critic_target)

    def softUpdate(self, net, target):
        """Soft updates target network.
        """
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def test(self, env, swap_channels=False, max_steps=500, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to 500
        :type max_steps: int, optional
        :param loop: Number of testing loops/epsiodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state = env.reset()[0]
                score = 0
                for idx_step in range(max_steps):
                    if swap_channels:
                        state = np.moveaxis(state, [3], [1])
                    action = self.getAction(state, epsilon=0)
                    state, reward, done, _, _ = env.step(action)
                    score += reward
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        if index is None:
            index = self.index

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           one_hot=self.one_hot,
                           index=index,
                           net_config=self.net_config,
                           batch_size=self.batch_size,
                           lr=self.lr,
                           learn_step=self.learn_step,
                           gamma=self.gamma,
                           tau=self.tau,
                           mutation=self.mut,
                           policy_freq=self.policy_freq,
                           device=self.device,
                           )

        clone.actor = self.actor.clone().to(self.device)
        clone.actor_target = self.actor_target.clone().to(self.device)
        clone.critic = self.critic.clone().to(self.device)
        clone.critic_target = self.critic_target.clone().to(self.device)

        clone.actor_optimizer = optim.Adam(
            clone.actor.parameters(), lr=clone.lr)
        clone.critic_optimizer = optim.Adam(
            clone.critic.parameters(), lr=clone.lr)

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save({
            'actor_init_dict': self.actor.init_dict,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_init_dict': self.actor_target.init_dict,
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_init_dict': self.critic.init_dict,
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_init_dict': self.critic_target.init_dict,
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'net_config': self.net_config,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'learn_step': self.learn_step,
            'gamma': self.gamma,
            'tau': self.tau,
            'mutation': self.mut,
            'index': self.index,
            'scores': self.scores,
            'fitness': self.fitness,
            'steps': self.steps,
        }, path)

    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path)
        self.net_config = checkpoint['net_config']
        if self.net_config['arch'] == 'mlp':
            self.actor = EvolvableMLP(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableMLP(
                **checkpoint['actor_target_init_dict'])
            self.critic = EvolvableMLP(**checkpoint['critic_init_dict'])
            self.critic_target = EvolvableMLP(
                **checkpoint['critic_target_init_dict'])
        elif self.net_config['arch'] == 'cnn':
            self.actor = EvolvableCNN(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableCNN(
                **checkpoint['actor_target_init_dict'])
            self.critic = EvolvableCNN(**checkpoint['critic_init_dict'])
            self.critic_target = EvolvableCNN(
                **checkpoint['critic_target_init_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(
            checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.lr = checkpoint['lr']
        self.learn_step = checkpoint['learn_step']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']
