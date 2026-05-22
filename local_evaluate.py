from __future__ import annotations

import argparse
import importlib.util
import json
import numbers
import time
from pathlib import Path
from typing import Iterable, List

import sys

ENV_DIR = Path(__file__).resolve().parents[1] / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from make_env import make_eval_env


def load_seeds(seeds_path: Path) -> List[int]:
    data = json.loads(seeds_path.read_text())
    if not isinstance(data, list) or not all(isinstance(seed, int) for seed in data):
        raise ValueError("Файл с seed'ами должен содержать JSON-список целых чисел.")
    return data


def load_agent(submission_path: Path, model_dir: Path | None):
    module_name = f"submission_{submission_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, submission_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать файл агента из {submission_path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_path.parent))
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(submission_path.parent):
            sys.path.pop(0)

    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        raise AttributeError("В файле должен быть определен класс Agent на верхнем уровне.")

    model_dir_value = str(model_dir) if model_dir is not None else None
    try:
        return agent_cls(model_dir=model_dir_value)
    except TypeError:
        return agent_cls()


def validate_action(action, action_space_n: int) -> int:
    if not isinstance(action, numbers.Integral):
        raise TypeError(
            f"Действие должно быть целым числом, получено значение типа {type(action)!r}."
        )

    action = int(action)
    if not 0 <= action < action_space_n:
        raise ValueError(f"Действие {action} выходит за диапазон [0, {action_space_n - 1}].")
    return action


def evaluate_submission(
    submission_path: Path,
    seeds: Iterable[int],
    model_dir: Path | None = None,
    max_steps: int | None = None,
):
    agent = load_agent(submission_path, model_dir=model_dir)
    episode_scores = []

    for episode_index, seed in enumerate(seeds, start=1):
        env = make_eval_env(seed=seed)
        try:
            observation, _ = env.reset(seed=seed)
            if hasattr(agent, "reset"):
                agent.reset(seed=seed)

            episode_reward = 0.0
            step_count = 0
            done = False
            started_at = time.perf_counter()

            while not done:
                action = validate_action(agent.act(observation), env.action_space.n)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                step_count += 1
                done = bool(terminated or truncated)

                if max_steps is not None and step_count >= max_steps:
                    break

            elapsed = time.perf_counter() - started_at
            episode_scores.append(episode_reward)
            print(
                f"[эпизод {episode_index:02d}] seed={seed} "
                f"счет={episode_reward:.2f} шаги={step_count} время={elapsed:.2f}с"
            )
        finally:
            env.close()

    mean_score = sum(episode_scores) / max(len(episode_scores), 1)
    print(f"средний_счет={mean_score:.4f}")
    return {"episode_scores": episode_scores, "mean_score": mean_score}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Локальная проверка отправок для RL-контеста по Breakout."
    )
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Путь к Python-файлу, в котором определен класс Agent.",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "public_seeds"
        / "local_validation_seeds.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Необязательная папка с собственными обученными весами.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Необязательное ограничение на число шагов в эпизоде для отладки.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = load_seeds(args.seeds_file)
    model_dir = args.model_dir or args.submission.parent
    evaluate_submission(
        submission_path=args.submission,
        seeds=seeds,
        model_dir=model_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
