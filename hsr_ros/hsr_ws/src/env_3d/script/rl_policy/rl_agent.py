#!/usr/bin/env python3
import gym
import copy
import torch
import numpy as np
from rl_games.algos_torch import model_builder, torch_ext


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class ResidualRL(object):
    def __init__(self, params):
        builder = model_builder.ModelBuilder()
        self.network = builder.load(params)

        self.states = None
        self.batch_size = 1

        self.config = params['config']
        self.clip_actions = self.config['clip_actions']
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config['normalize_value']

        self.device = 'cuda'
        self.num_actions = self.config['num_actions']
        self.num_observations = self.config['num_observations']

        self.action_space = gym.spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.observation_space = gym.spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        obs_shape = self.observation_space.shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.config['num_actors'],
            'value_size': self.config.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_determenistic=True):
        obs = self.unsqueeze_obs(obs)
        obs = self.preprocess_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)

        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def unsqueeze_obs(self, obs):
        if type(obs) is dict:
            for k, v in obs.items():
                obs[k] = self.unsqueeze_obs(v)
        else:
            if len(obs.size()) > 1 or obs.size()[0] > 1:
                obs = obs.unsqueeze(0)
        return obs

    def preprocess_obs(self, obs):
        if type(obs) is dict:
            obs = copy.copy(obs)
            for k, v in obs.items():
                if v.dtype == torch.uint8:
                    obs[k] = v.float() / 255.0
                else:
                    obs[k] = v.float()
        else:
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0
        return obs

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)

        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size()[2]),
                            dtype=torch.float32).to(self.device) for s in rnn_states]

    def reset(self):
        self.init_rnn()