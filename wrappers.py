from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent
    gym = None
    Box = None


def _require_gymnasium() -> None:
    if gym is None or Box is None:
        raise ImportError(
            "gymnasium is required to build the Breakout environment. "
            "Install gymnasium[atari,accept-rom-license] in the runtime."
        )


if gym is not None:

    class NumpyFrameStack(gym.Wrapper):
        def __init__(self, env: gym.Env, num_stack: int = 4):
            super().__init__(env)
            if num_stack <= 0:
                raise ValueError("num_stack must be positive")

            self.num_stack = num_stack
            self.frames: Deque[np.ndarray] = deque(maxlen=num_stack)

            obs_shape = env.observation_space.shape
            if obs_shape is None:
                raise ValueError("Environment must define a static observation shape")

            self.observation_space = Box(
                low=0,
                high=255,
                shape=(num_stack, *obs_shape),
                dtype=env.observation_space.dtype,
            )

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            frame = np.asarray(observation, dtype=self.observation_space.dtype)
            self.frames.clear()
            for _ in range(self.num_stack):
                self.frames.append(frame.copy())
            return self._stack_frames(), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            frame = np.asarray(observation, dtype=self.observation_space.dtype)
            self.frames.append(frame.copy())
            return self._stack_frames(), reward, terminated, truncated, info

        def _stack_frames(self) -> np.ndarray:
            return np.stack(list(self.frames), axis=0)


    class SignRewardWrapper(gym.RewardWrapper):
        def reward(self, reward):
            return float(np.sign(reward))

else:

    class NumpyFrameStack:  # pragma: no cover - placeholder only
        def __init__(self, *args, **kwargs):
            _require_gymnasium()


    class SignRewardWrapper:  # pragma: no cover - placeholder only
        def __init__(self, *args, **kwargs):
            _require_gymnasium()
