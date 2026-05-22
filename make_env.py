from __future__ import annotations

from typing import Optional

try:
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent
    gym = None
    AtariPreprocessing = None

from wrappers import NumpyFrameStack, SignRewardWrapper

DEFAULT_ENV_ID = "ALE/Breakout-v5"


def _require_runtime() -> None:
    if gym is None or AtariPreprocessing is None:
        raise ImportError(
            "gymnasium with Atari support is required. "
            "Install gymnasium[atari,accept-rom-license]."
        )

    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "ale-py is required for ALE/Breakout-v5. "
            "Install gymnasium[atari,accept-rom-license]."
        ) from exc

    register_envs = getattr(gym, "register_envs", None)
    if callable(register_envs):
        register_envs(ale_py)


def make_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: Optional[int] = None,
    clip_rewards: bool = False,
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
):
    _require_runtime()

    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = NumpyFrameStack(env, num_stack=frame_stack)

    if clip_rewards:
        env = SignRewardWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def make_train_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: Optional[int] = None,
    frame_stack: int = 4,
):
    return make_env(
        env_id=env_id,
        seed=seed,
        clip_rewards=True,
        frame_stack=frame_stack,
        render_mode=None,
    )


def make_eval_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: Optional[int] = None,
    frame_stack: int = 4,
):
    return make_env(
        env_id=env_id,
        seed=seed,
        clip_rewards=False,
        frame_stack=frame_stack,
        render_mode=None,
    )
