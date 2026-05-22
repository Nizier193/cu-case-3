from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple
from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


ENV_DIR = Path(__file__).resolve().parents[1] / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from make_env import make_eval_env, make_train_env


ACTION_FIRE = 1


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("Для запуска примера DQN нужен PyTorch.")


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, observation):
        features = self.encoder(observation)
        return self.head(features)


class DQNTrainer:
    def __init__(
        self,
        seed: int = 0,
        replay_size: int = 100_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        train_start: int = 10_000,
        train_freq: int = 4,
        target_sync_freq: int = 2_500,
    ):
        _require_torch()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = _default_device()
        print(f"Устройство обучения: {self.device}", flush=True)

        self.env = make_train_env(seed=seed)
        self.action_dim = self.env.action_space.n
        self.online_net = QNetwork(self.action_dim).to(self.device)
        self.target_net = QNetwork(self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        self.replay = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_start = train_start
        self.train_freq = train_freq
        self.target_sync_freq = target_sync_freq

    def _epsilon(self, step: int, total_steps: int) -> float:
        fraction = min(step / max(total_steps, 1), 1.0)
        return 0.3

    def _to_tensor(self, array: np.ndarray):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device) / 255.0

    def _select_action(self, observation: np.ndarray, epsilon: float, needs_fire: bool) -> Tuple[int, bool]:
        if needs_fire:
            return ACTION_FIRE, False

        if random.random() < epsilon:
            return random.randrange(self.action_dim), False

        observation_tensor = self._to_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(observation_tensor)
            
        return int(q_values.argmax(dim=1).item()), False

    def _optimize(self) -> Dict[str, float]:
        batch = self.replay.sample(self.batch_size)

        observations = self._to_tensor(np.stack([t.observation for t in batch], axis=0))
        next_observations = self._to_tensor(
            np.stack([t.next_observation for t in batch], axis=0)
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )

        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.online_net(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_observations).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        td_errors = targets - q_values.detach()
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_values.detach().mean().item()),
            "q_max": float(q_values.detach().max().item()),
            "target_mean": float(targets.detach().mean().item()),
            "td_error_abs_mean": float(td_errors.abs().mean().item()),
            "grad_norm": float(grad_norm.item()),
        }

    def train(
        self,
        total_steps: int,
        checkpoint_path: Path,
        log_every: int = 1_000,
    ) -> Dict[str, object]:
        checkpoint_path = Path(checkpoint_path)

        observation, info = self.env.reset()
        needs_fire = True
        episode_reward = 0.0
        episode_length = 0
        last_lives = info.get("lives") if isinstance(info, dict) else None

        history: Dict[str, object] = {
            "losses": [],
            "q_mean": [],
            "q_max": [],
            "target_mean": [],
            "td_error_abs_mean": [],
            "grad_norm": [],
            "update_steps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "reward_events": [],
            "epsilons": [],
            "action_counts": [0 for _ in range(self.action_dim)],
            "life_losses": 0,
            "train_start": self.train_start,
            "train_freq": self.train_freq,
            "target_sync_freq": self.target_sync_freq,
        }

        progress = tqdm(range(1, total_steps + 1), desc="Обучаем модельку")
        for step in progress:
            epsilon = self._epsilon(step, total_steps)
            action, needs_fire = self._select_action(observation, epsilon, needs_fire)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            action_counts = history["action_counts"]
            action_counts[action] += 1
            history["epsilons"].append(float(epsilon))
            
            if reward != 0:
                history["reward_events"].append((step, float(reward)))

            self.replay.add(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation,
                    done=done,
                )
            )
            episode_reward += reward
            episode_length += 1
            observation = next_observation

            if len(self.replay) >= self.train_start and (step % self.train_freq) == 0:
                optimize_stats = self._optimize()
                history["losses"].append(optimize_stats["loss"])
                for key in (
                    "q_mean",
                    "q_max",
                    "target_mean",
                    "td_error_abs_mean",
                    "grad_norm",
                ):
                    history[key].append(optimize_stats[key])
                history["update_steps"].append(step)

            if step % self.target_sync_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            lives = info.get("lives") if isinstance(info, dict) else None
            if (
                lives is not None
                and last_lives is not None
                and 0 < lives < last_lives
            ):
                needs_fire = True
                history["life_losses"] += 1
            if lives is not None:
                last_lives = lives

            if done:
                history["episode_rewards"].append(float(episode_reward))
                history["episode_lengths"].append(int(episode_length))

                observation, info = self.env.reset()
                needs_fire = True
                episode_reward = 0.0
                episode_length = 0
                last_lives = info.get("lives") if isinstance(info, dict) else None

            if log_every > 0 and step % log_every == 0:
                recent_losses = history["losses"][-100:]
                recent_rewards = history["episode_rewards"][-10:]
                progress.set_postfix(
                    loss=np.mean(recent_losses) if recent_losses else None,
                    return10=np.mean(recent_rewards) if recent_rewards else None,
                    updates=len(history["losses"]),
                    replay=len(self.replay),
                    eps=epsilon,
                )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), checkpoint_path)
        self.env.close()

        return history


class Agent:
    """Жадный DQN-агент, совместимый с форматом отправки.

    Ожидает файл весов `dqn_breakout.pt` внутри `model_dir`.
    """

    def __init__(self, model_dir=None):
        _require_torch()
        print("Обучаем на Kaggle")

        self.model_dir = Path(model_dir) if model_dir is not None else Path(".")
        self.checkpoint_path = self.model_dir
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {self.checkpoint_path}. Сначала обучите "
                "модель или укажите свои собственные веса."
            )

        env = make_eval_env()
        action_dim = env.action_space.n
        env.close()

        self.device = _default_device()
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Минимальный пример обучения DQN для Breakout."
    )
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/dqn_breakout.pt"),
    )
    return parser.parse_args()


# def main():
#     args = parse_args()
#     trainer = DQNTrainer(seed=args.seed)
#     trainer.train(total_steps=args.total_steps, checkpoint_path=args.checkpoint_path)
#     print(f"Файл весов сохранен в {args.checkpoint_path}", flush=True)


# if __name__ == "__main__":
#     main()
