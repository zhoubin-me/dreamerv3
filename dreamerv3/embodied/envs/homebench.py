from homebench import HomeBench
from homebench.action_modes.action_mode import DeltaJointPosition

import functools
import os

import embodied
import numpy as np


class HomeBenchEmb(embodied.Env):
    def __init__(self, task, repeat=1, render=True, size=(64, 64), camera=-1):
        assert type(task) == str
        hb_env = HomeBench("HomeBenchExample.ReachTarget", DeltaJointPosition())
        self._env = hb_env

    @functools.cached_property
    def obs_space(self):
        spec = self._env.environment_spec
        if "reward" in spec:
            spec["obs_reward"] = spec.pop("reward")
        
        obs_spec = {}
        for k, v in spec.items():
            
        
        return {
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            **{k or self._obs_key: self._convert(v) for k, v in spec.items()},
        }

    @functools.cached_property
    def act_space(self):
        pass

    def step(self, action):
        pass

    def _obs(self, time_step):
        pass

    def _convert(self, space):
        pass