"""Offline critic analysis: Q as a function of distance to the episode end.

With a terminal reward of ``r`` and a per-chunk discount ``g``, a well fitted
critic gives Q(s_k, a_k) ~= r * g**k for the transition ``k`` chunks before the
end of a successful episode, and ~0 throughout a failed one. This reads a
learner checkpoint plus the demo cache and/or the online replay buffer and
compares the critic with that target, split by outcome, for the dataset action,
the VLA reference chunk and the actor mean. It never trains anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from armnet_rlt.artifacts import load_norm_stats
from armnet_rlt.config import RLTActorLearnerConfig, RLTConfig, RLTNetworkConfig
from armnet_rlt.learner import _q_stats
from armnet_rlt.policy import RLTPolicy


def config_from_checkpoint(raw: dict[str, Any]) -> RLTConfig:
    """Rebuild the learner's RLTConfig from ``asdict(config)`` in a checkpoint."""
    network_values = dict(raw["network"])
    for name in ("hidden_dims", "delta_action_mask"):
        if name in network_values:
            network_values[name] = tuple(network_values[name])
    values: dict[str, Any] = {}
    for field in dataclasses.fields(RLTConfig):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if field.name == "network":
            value = RLTNetworkConfig(**network_values)
        elif field.name == "actor_learner":
            value = RLTActorLearnerConfig(**value)
        elif field.name == "jerk_joint_weights":
            value = tuple(value)
        values[field.name] = value
    return RLTConfig(**values)


def segment_episodes(transitions: list[dict[str, Tensor]]) -> list[list[int]]:
    """Group indices into episodes that end at ``done == 1``.

    Transitions are stored in episode order. A trailing run with no terminal
    transition is dropped, and the first episode of a ring buffer may be
    truncated at its start, which does not affect distance to the end.
    """
    episodes: list[list[int]] = []
    current: list[int] = []
    for index, transition in enumerate(transitions):
        current.append(index)
        if float(transition["done"]) == 1.0:
            episodes.append(current)
            current = []
    return episodes


def _collate(
    transitions: list[dict[str, Tensor]], indices: list[int], device: torch.device
) -> dict[str, Any]:
    def stack(key: str) -> Tensor:
        return torch.stack([transitions[i][key] for i in indices]).to(device)

    return {
        "state": {
            "rl_token": stack("rl_token"),
            "proprioception": stack("proprioception"),
            "reference_action": stack("reference_action"),
        },
        "action": stack("action"),
        "reward": stack("reward"),
    }


@torch.no_grad()
def _per_transition_q(
    policy: RLTPolicy, batch: dict[str, Any]
) -> dict[str, Tensor]:
    state = batch["state"]
    token = state["rl_token"]
    proprioception = state["proprioception"]
    reference = state["reference_action"]
    actor = policy.actor

    def min_q(action: Tensor) -> Tensor:
        return policy.critic_ensemble(token, proprioception, action).min(
            dim=0
        ).values

    data = actor.normalize_action(batch["action"], proprioception).clamp(
        -1.0, 1.0
    )
    ref = actor.normalize_action(
        reference[:, : policy.predicted_action_dim], proprioception
    ).clamp(-1.0, 1.0)
    _, actor_mean = actor(token, proprioception, reference, sample=False)
    return {
        "q_data": min_q(data),
        "q_reference": min_q(ref),
        "q_actor": min_q(actor_mean),
    }


def analyze_transitions(
    policy: RLTPolicy,
    transitions: list[dict[str, Tensor]],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    max_k: int = 30,
) -> dict[str, Any]:
    """Q versus chunks-to-end, per outcome, plus aggregate ``_q_stats``."""
    device = torch.device(device)
    discount = float(policy.chunk_discount)
    episodes = segment_episodes(transitions)
    if not episodes:
        raise ValueError("no complete episode (done == 1) in the transitions")

    order: list[int] = []
    distance: list[int] = []
    success: list[bool] = []
    ideal: list[float] = []
    for episode in episodes:
        terminal_reward = float(transitions[episode[-1]]["reward"])
        for position, index in enumerate(episode):
            k = len(episode) - 1 - position
            order.append(index)
            distance.append(k)
            success.append(terminal_reward > 0.0)
            ideal.append(terminal_reward * discount**k)

    policy.eval()
    columns: dict[str, list[Tensor]] = {}
    aggregate: dict[str, list[float]] = {}
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = _collate(transitions, indices, device)
        for name, value in _per_transition_q(policy, batch).items():
            columns.setdefault(name, []).append(value.cpu())
        if len(indices) < 2:  # _q_stats takes a std over the batch
            continue
        for name, value in _q_stats(policy, batch).items():
            aggregate.setdefault(name, []).append(value)
    q = {name: torch.cat(parts) for name, parts in columns.items()}
    k_values = torch.tensor(distance)
    succeeded = torch.tensor(success)
    ideal_values = torch.tensor(ideal)

    by_outcome: dict[str, list[dict[str, float]]] = {}
    for label, selected in (("success", succeeded), ("failure", ~succeeded)):
        rows = []
        for k in range(max_k + 1):
            mask = selected & (
                (k_values == k) if k < max_k else (k_values >= max_k)
            )
            count = int(mask.sum())
            if not count:
                continue
            rows.append(
                {
                    "k": k,
                    "n": count,
                    "ideal": float(ideal_values[mask].mean()),
                    "q_data": float(q["q_data"][mask].mean()),
                    "q_data_std": float(q["q_data"][mask].std())
                    if count > 1
                    else 0.0,
                    "q_reference": float(q["q_reference"][mask].mean()),
                    "q_actor": float(q["q_actor"][mask].mean()),
                }
            )
        by_outcome[label] = rows

    summary = {
        "episodes": len(episodes),
        "successful_episodes": int(
            sum(1 for e in episodes if float(transitions[e[-1]]["reward"]) > 0)
        ),
        "transitions": len(order),
        "chunk_discount": discount,
        "max_k_bucket": f"k>={max_k} is pooled into the last row",
        "q_data_minus_ideal_success": float(
            (q["q_data"] - ideal_values)[succeeded].mean()
        )
        if bool(succeeded.any())
        else None,
        "q_data_minus_ideal_failure": float(
            (q["q_data"] - ideal_values)[~succeeded].mean()
        )
        if bool((~succeeded).any())
        else None,
    }
    return {
        "summary": summary,
        "by_outcome": by_outcome,
        "aggregate_q_stats": {
            name: sum(values) / len(values) for name, values in aggregate.items()
        },
    }


def _format(label: str, result: dict[str, Any]) -> str:
    lines = [f"== {label} ==", json.dumps(result["summary"], indent=2)]
    for outcome, rows in result["by_outcome"].items():
        lines.append(f"-- {outcome} (k = chunks to episode end) --")
        lines.append(
            f"{'k':>3} {'n':>6} {'ideal':>8} {'Q(data)':>9} "
            f"{'+-std':>7} {'Q(ref)':>9} {'Q(actor)':>9}"
        )
        for row in rows:
            lines.append(
                f"{row['k']:>3} {row['n']:>6} {row['ideal']:>8.4f} "
                f"{row['q_data']:>9.4f} {row['q_data_std']:>7.4f} "
                f"{row['q_reference']:>9.4f} {row['q_actor']:>9.4f}"
            )
    lines.append("-- aggregate _q_stats (mean over batches) --")
    for name, value in sorted(result["aggregate_q_stats"].items()):
        lines.append(f"{name:<34} {value:>10.4f}")
    return "\n".join(lines)


def _load_replay_buffer(path: Path) -> list[dict[str, Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid online replay snapshot at {path}")
    return list(payload["transitions"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="learner checkpoint.pt (e.g. <run>/step_005000/checkpoint.pt)",
    )
    parser.add_argument(
        "--assets-dir", type=Path, required=True,
        help="directory with norm_stats.json and the per-timestep stats",
    )
    parser.add_argument("--demo-cache", type=Path)
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-k", type=int, default=30)
    parser.add_argument("--output", type=Path, help="write the results as JSON")
    args = parser.parse_args(argv)
    if args.demo_cache is None and args.replay_buffer is None:
        parser.error("give --demo-cache and/or --replay-buffer")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(payload["config"])
    policy = RLTPolicy(config)
    policy.set_norm_stats(
        load_norm_stats(
            args.assets_dir,
            use_delta_actions=config.policy_uses_delta_actions,
            network=config.network,
        ),
        use_quantiles=config.use_quantile_norm,
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(args.device)
    print(f"checkpoint step: {payload.get('step')}", flush=True)

    sources: dict[str, list[dict[str, Tensor]]] = {}
    if args.demo_cache is not None:
        from armnet_rlt.artifacts import load_transition_cache

        sources["demo_cache"], _ = load_transition_cache(
            args.demo_cache, config.network
        )
    if args.replay_buffer is not None:
        sources["online_replay_buffer"] = _load_replay_buffer(args.replay_buffer)

    results = {}
    for label, transitions in sources.items():
        results[label] = analyze_transitions(
            policy,
            transitions,
            device=args.device,
            batch_size=args.batch_size,
            max_k=args.max_k,
        )
        print(_format(label, results[label]), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
