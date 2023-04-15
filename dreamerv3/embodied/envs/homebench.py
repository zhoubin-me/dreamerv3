from homebench import HomeBench
from homebench.action_modes.action_mode import DeltaJointPosition

import functools
import os

import embodied
import numpy as np


class HomeBench(embodied.Env):
    def __init__(self, task, repeat=1, render=True, size=(64, 64), camera=-1):
        assert type(task) == str
        hb_env = HomeBench(task, num_envs=1, steps=2000)
        self._hbenv = hb_env
        self._hbenv.obs_space = hb_env.observation_space
        self._hbenv.act_space = hb_env.act_space
        from . import from_dm

        self._env = from_dm.FromDM(self._hbenv)
        self._env = embodied.wrappers.ExpandScalars(self._env)
        self._render = render
        self._size = size
        self._camera = camera

    @functools.cached_property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        return spaces

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        return obs

    def render(self):
        pass
