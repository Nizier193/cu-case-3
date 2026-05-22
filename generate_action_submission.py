from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import numbers
import sys
from pathlib import Path
from typing import Iterable, List


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


def make_row_id(split_name: str, episode_index: int, step_index: int) -> str:
    return f"{split_name}_{episode_index:03d}_{step_index:05d}"


def generate_action_csv(
    agent_path: Path,
    seeds: Iterable[int],
    output_path: Path,
    model_dir: Path | None,
    split_name: str,
    max_steps_per_episode: int,
) -> None:
    agent = load_agent(agent_path, model_dir=model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "action"])
        writer.writeheader()

        for episode_index, seed in enumerate(seeds):
            env = make_eval_env(seed=seed)
            try:
                observation, _ = env.reset(seed=seed)
                if hasattr(agent, "reset"):
                    agent.reset(seed=seed)

                done = False
                for step_index in range(max_steps_per_episode):
                    if done:
                        action = 0
                    else:
                        action = validate_action(agent.act(observation), env.action_space.n)
                    writer.writerow(
                        {
                            "id": make_row_id(split_name, episode_index, step_index),
                            "action": action,
                        }
                    )

                    if not done:
                        observation, _, terminated, truncated, _ = env.step(action)
                        done = bool(terminated or truncated)
            finally:
                env.close()

    print(f"CSV с действиями сохранен в {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сгенерировать CSV с действиями агента для известных seed'ов."
    )
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "public_seeds"
        / "local_validation_seeds.json",
    )
    parser.add_argument("--output", type=Path, default=Path("action_submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--split-name", type=str, default="local")
    parser.add_argument("--max-steps-per-episode", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps_per_episode <= 0:
        raise ValueError("max-steps-per-episode должен быть положительным.")

    generate_action_csv(
        agent_path=args.agent,
        seeds=load_seeds(args.seeds_file),
        output_path=args.output,
        model_dir=args.model_dir or args.agent.parent,
        split_name=args.split_name,
        max_steps_per_episode=args.max_steps_per_episode,
    )


if __name__ == "__main__":
    main()
