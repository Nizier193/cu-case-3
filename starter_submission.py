from pathlib import Path

import torch

from baseline_dqn_stub import QNetwork, _default_device
from make_env import make_eval_env

ACTION_FIRE = 1


class Agent:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(".")
        self.checkpoint_path = self.model_dir
        if self.checkpoint_path.is_dir():
            self.checkpoint_path = self.checkpoint_path / "dqn_breakout.pt"

        self.device = _default_device()

        env = make_eval_env()
        action_dim = env.action_space.n
        env.close()

        self.network = QNetwork(action_dim).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()

        self.needs_fire = True

    def reset(self, seed=None):
        self.needs_fire = True

    def act(self, observation):
        if self.needs_fire:
            self.needs_fire = False
            return ACTION_FIRE

        observation_tensor = (
            torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            / 255.0
        )

        with torch.no_grad():
            q_values = self.network(observation_tensor)

        return int(q_values.argmax(dim=1).item())