from homebench import HomeBench
from homebench.action_modes.action_mode import DeltaJointPosition

import functools
import os
import cv2

import embodied
import numpy as np

class HomeBenchEmb(embodied.Env):
    def __init__(self, task, repeat=1, render=True, size=(64, 64), camera=-1):
        assert type(task) == str
        hb_env = HomeBench("HomeBenchExample.ReachTarget", DeltaJointPosition(), episode_steps=200)
        self._env = hb_env

    @functools.cached_property
    def obs_space(self):
        spec = self._env.environment_spec.observations

        obs_spec = {}
        for k, v in spec.items():
            if len(v.shape) == 3:
                v_shape = (64, 64, 3)
            else:
                v_shape = v.shape

            v_ = embodied.Space(str(v.dtype), v_shape)
            obs_spec[k] = v_

        return {
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            **obs_spec,
        }

    @functools.cached_property
    def act_space(self):
        act_spec = self._env.environment_spec.actions
        act_space = embodied.Space(str(act_spec.dtype), act_spec.shape,
                                   low=act_spec.low, high=act_spec.high)
        return {
            "reset": embodied.Space(bool),
            "action": act_space
        }

    def step(self, action):
        action = action.copy()
        reset = action.pop("reset")
        if reset or self._done:
            time_step = self._env.reset()
        else:
            action_ = action['action']
            time_step = self._env.step([action_])
        time_step = time_step[0]
        self._done = time_step.last()
        obs = self._obs(time_step)
        return obs

    def _obs(self, time_step):
        if not time_step.first():
            assert time_step.discount in (0, 1), time_step.discount
        obs = time_step.observation
        obs = dict(obs)
        for k, v in obs.items():
            if len(v.shape) == 3:
                v = v.transpose(1, 2, 0)
                v = cv2.resize(v, (64, 64))
                obs[k] = v
        return dict(
            reward=np.float32(0.0 if time_step.first() else time_step.reward),
            is_first=time_step.first(),
            is_last=time_step.last(),
            is_terminal=False if time_step.first() else time_step.discount == 0,
            **obs,
        )
