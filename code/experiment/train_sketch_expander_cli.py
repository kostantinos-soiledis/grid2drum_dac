#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
    """Avoid the repo-local inspect.py shadowing Python's stdlib inspect."""
    original_path = list(sys.path)
    repo = str(REPO_ROOT)
    sys.path = [path for path in sys.path if path not in {"", repo}]
    try:
        import inspect  # noqa: F401
        import dataclasses  # noqa: F401
    finally:
        sys.path = original_path


_preload_stdlib_inspect()
from dataclasses import asdict

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:  # pragma: no cover
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.diffusion_cache_utils import FAMILY_STATE_ID_VOCAB_SIZES
from data.encodec_utils import resolve_device
from data.sketch_dataset import (
    DEFAULT_SKETCH_MAX_SLOTS,
    HIHAT_OPEN_CLASS_IDS,
    SKETCH_FAMILY_NAMES,
    SNARE_BACKBEAT_STEPS,
    SNARE_GHOST_VELOCITY_THRESHOLD,
    build_sketch_dataloader,
    sketch_controls_to_public_dict,
)
from data.sketch_render import DEFAULT_GRID_FRAME_RATE, build_diffusion_batch_from_events
from sketch_expander import (
    SketchExpander,
    SketchExpanderConfig,
    _batched_ornament_target_masks,
    decode_event_plan,
    sketch_expander_loss,
)


EVENT_PLAN_GROUP_NAMES: tuple[str, ...] = (
    "kick_ghost",
    "snare_ghost",
    "snare_roll_drag",
    "snare_roll_run",
    "off16_hat",
    "open_hat",
    "tom_fill",
    "crash",
    "ride",
)


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised 16-step sketch-to-grid expander.")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--out-dir", type=str, default=str(RUNS_ROOT / "sketch_expander_dac44_native_v5"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-slots", type=int, default=DEFAULT_SKETCH_MAX_SLOTS)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--preview-sample-idx", type=int, default=0)
    parser.add_argument("--preview-grid-frame-rate", type=float, default=DEFAULT_GRID_FRAME_RATE)
    parser.add_argument("--no-eval-preview", action="store_true")
    parser.add_argument("--event-plan-eval-every", type=int, default=1)
    parser.add_argument("--no-event-plan-eval", action="store_true")
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="event_plan_score",
        choices=("event_plan_score", "selection_score", "val_loss"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def _append_history(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_history_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _save_checkpoint(
    path: Path,
    *,
    model: SketchExpander,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    best_selection_score: float,
    best_event_plan_score: float,
    checkpoint_metric: str,
    best_checkpoint_score: float,
    extra: Mapping[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.to_config_dict(),
        "best_val_loss": float(best_val_loss),
        "best_selection_score": float(best_selection_score),
        "best_event_plan_score": float(best_event_plan_score),
        "checkpoint_metric": str(checkpoint_metric),
        "best_checkpoint_score": float(best_checkpoint_score),
        **dict(extra),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _mean_metrics(total: dict[str, float], count: int, prefix: str) -> dict[str, float]:
    denom = max(1, int(count))
    return {f"{prefix}_{key}": float(value) / float(denom) for key, value in sorted(total.items())}


def _musical_selection_score(row: Mapping[str, Any]) -> float:
    offset_mae = float(row.get("val_offset_mae", 0.5) or 0.5)
    offset_score = max(0.0, min(1.0, 1.0 - (float(offset_mae) / 0.5)))
    phrase_score = (
        float(row.get("val_fill_start_acc", 0.0) or 0.0)
        + float(row.get("val_fill_length_acc", 0.0) or 0.0)
        + float(row.get("val_tom_direction_acc", 0.0) or 0.0)
        + float(row.get("val_fill_accent_shape_acc", 0.0) or 0.0)
    ) / 4.0
    return float(
        (0.22 * float(row.get("val_event_recall", 0.0) or 0.0))
        + (0.17 * float(row.get("val_budget_score", 0.0) or 0.0))
        + (0.16 * float(row.get("val_tom_fill_recall", row.get("val_tom_crash_recall", 0.0)) or 0.0))
        + (0.13 * float(row.get("val_kick_ghost_recall", 0.0) or 0.0))
        + (0.08 * float(row.get("val_snare_ghost_recall", 0.0) or 0.0))
        + (0.03 * float(row.get("val_snare_roll_drag_recall", 0.0) or 0.0))
        + (0.03 * float(row.get("val_snare_roll_run_recall", 0.0) or 0.0))
        + (0.06 * float(phrase_score))
        + (0.05 * float(row.get("val_event_precision", 0.0) or 0.0))
        + (0.04 * float(row.get("val_hihat_open_class_acc", 0.0) or 0.0))
        + (0.03 * float(offset_score))
    )


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if float(denom) > 0.0 else 0.0


def _precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = _safe_div(float(tp), float(tp) + float(fp))
    recall = _safe_div(float(tp), float(tp) + float(fn))
    f1 = _safe_div(2.0 * float(precision) * float(recall), float(precision) + float(recall))
    return float(precision), float(recall), float(f1)


def _event_plan_key(event: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(event.get("family", "")),
        int(event.get("step", -1)),
        int(event.get("slot", 0)),
    )


def _event_plan_family_step_key(event: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(event.get("family", "")),
        int(event.get("step", -1)),
    )


def _target_events_from_batch(batch: Mapping[str, Any], sample_idx: int) -> list[dict[str, Any]]:
    class_names = [str(x) for x in list(batch.get("class_names") or [])]
    if not class_names:
        class_names = [f"family_{idx}" for idx in range(int(batch["target_presence"].shape[1]))]
    presence = torch.as_tensor(batch["target_presence"][int(sample_idx)], dtype=torch.float32).detach().cpu()
    velocity = torch.as_tensor(batch["target_velocity"][int(sample_idx)], dtype=torch.float32).detach().cpu()
    offset = torch.as_tensor(batch["target_offset"][int(sample_idx)], dtype=torch.float32).detach().cpu()
    class_id = torch.as_tensor(batch["target_class_id"][int(sample_idx)], dtype=torch.long).detach().cpu()
    events: list[dict[str, Any]] = []
    for family_idx, family_name in enumerate(class_names[: int(presence.shape[0])]):
        for step_idx in range(int(presence.shape[1])):
            for slot_idx in range(int(presence.shape[2])):
                if float(presence[int(family_idx), int(step_idx), int(slot_idx)].item()) <= 0.5:
                    continue
                events.append(
                    {
                        "family": str(family_name),
                        "step": int(step_idx),
                        "slot": int(slot_idx),
                        "velocity": float(velocity[int(family_idx), int(step_idx), int(slot_idx)].item()),
                        "offset": float(offset[int(family_idx), int(step_idx), int(slot_idx)].item()),
                        "class_id": int(class_id[int(family_idx), int(step_idx), int(slot_idx)].item()),
                    }
                )
    return events


def _event_plan_target_keysets_by_group(
    batch: Mapping[str, Any],
    *,
    class_names: list[str],
    budget_group_names: list[str],
) -> dict[str, list[set[tuple[str, int, int]]]]:
    presence = torch.as_tensor(batch["target_presence"], dtype=torch.float32)
    velocity = torch.as_tensor(batch["target_velocity"], dtype=torch.float32)
    class_id = torch.as_tensor(batch["target_class_id"], dtype=torch.long)
    sketch_hits_raw = batch.get("sketch_hits")
    sketch_hits = torch.as_tensor(sketch_hits_raw, dtype=torch.float32) if sketch_hits_raw is not None else None
    masks = _batched_ornament_target_masks(
        presence.gt(0.5),
        velocity,
        class_id,
        sketch_hits,
        budget_group_names=budget_group_names,
    )
    batch_size = int(presence.shape[0])
    grouped: dict[str, list[set[tuple[str, int, int]]]] = {
        group_name: [set() for _ in range(batch_size)] for group_name in EVENT_PLAN_GROUP_NAMES
    }
    for group_name in EVENT_PLAN_GROUP_NAMES:
        mask = masks.get(str(group_name))
        if mask is None:
            continue
        for batch_idx in range(batch_size):
            positions = torch.nonzero(mask[int(batch_idx)], as_tuple=False)
            for family_idx_t, step_idx_t, slot_idx_t in positions:
                family_idx = int(family_idx_t.item())
                if not (0 <= int(family_idx) < int(len(class_names))):
                    continue
                grouped[str(group_name)][int(batch_idx)].add(
                    (
                        str(class_names[int(family_idx)]),
                        int(step_idx_t.item()),
                        int(slot_idx_t.item()),
                    )
                )
    return grouped


def _sketch_anchor(batch: Mapping[str, Any], batch_idx: int, family: str, step: int) -> bool:
    if str(family) not in SKETCH_FAMILY_NAMES:
        return False
    sketch_hits_raw = batch.get("sketch_hits")
    if sketch_hits_raw is None:
        return False
    sketch_hits = torch.as_tensor(sketch_hits_raw, dtype=torch.float32)
    if int(sketch_hits.dim()) != 3:
        return False
    sketch_names = [str(x) for x in list(batch.get("sketch_family_names") or SKETCH_FAMILY_NAMES)]
    if str(family) not in sketch_names:
        return False
    sketch_idx = int(sketch_names.index(str(family)))
    if not (0 <= int(batch_idx) < int(sketch_hits.shape[0]) and 0 <= int(step) < int(sketch_hits.shape[-1])):
        return False
    return bool(float(sketch_hits[int(batch_idx), int(sketch_idx), int(step)].item()) > 0.5)


def _event_plan_predicted_groups(
    event: Mapping[str, Any],
    batch: Mapping[str, Any],
    batch_idx: int,
) -> set[str]:
    family, step_idx, slot_idx = _event_plan_key(event)
    class_id = int(event.get("class_id", 0) or 0)
    groups: set[str] = set()
    if family == "kick":
        if int(slot_idx) > 0 or not _sketch_anchor(batch, int(batch_idx), "kick", int(step_idx)):
            groups.add("kick_ghost")
    elif family == "snare":
        if int(slot_idx) > 0:
            groups.add("snare_roll_drag")
        elif (
            int(step_idx) not in set(SNARE_BACKBEAT_STEPS)
            and not _sketch_anchor(batch, int(batch_idx), "snare", int(step_idx))
        ):
            velocity = float(event.get("velocity", 0.0) or 0.0)
            if int(class_id) == 0 and velocity <= float(SNARE_GHOST_VELOCITY_THRESHOLD):
                groups.add("snare_ghost")
            else:
                groups.add("snare_roll_run")
    elif family == "hihat":
        if int(step_idx) % 2 == 1 and int(slot_idx) == 0:
            groups.add("off16_hat")
        if int(class_id) in set(HIHAT_OPEN_CLASS_IDS):
            groups.add("open_hat")
    elif family in {"tom_high", "tom_mid", "tom_floor"}:
        groups.add("tom_fill")
    elif family == "crash":
        groups.add("crash")
    elif family == "ride":
        groups.add("ride")
    return groups


def _empty_event_plan_totals() -> dict[str, float]:
    totals = {
        "examples": 0.0,
        "exact_tp": 0.0,
        "exact_fp": 0.0,
        "exact_fn": 0.0,
        "family_step_tp": 0.0,
        "family_step_fp": 0.0,
        "family_step_fn": 0.0,
        "matched_velocity_abs": 0.0,
        "matched_velocity_count": 0.0,
    }
    for group_name in EVENT_PLAN_GROUP_NAMES:
        totals[f"{group_name}_hit"] = 0.0
        totals[f"{group_name}_target"] = 0.0
    return totals


def _update_event_plan_totals(
    totals: dict[str, float],
    pred_events_batch: list[list[dict[str, Any]]],
    batch: Mapping[str, Any],
    *,
    class_names: list[str],
    budget_group_names: list[str],
) -> None:
    target_group_keysets = _event_plan_target_keysets_by_group(
        batch,
        class_names=class_names,
        budget_group_names=budget_group_names,
    )
    batch_size = int(torch.as_tensor(batch["target_presence"]).shape[0])
    for batch_idx in range(batch_size):
        pred_events = list(pred_events_batch[int(batch_idx)]) if int(batch_idx) < int(len(pred_events_batch)) else []
        target_events = _target_events_from_batch(batch, int(batch_idx))
        pred_by_key = {_event_plan_key(event): dict(event) for event in pred_events}
        target_by_key = {_event_plan_key(event): dict(event) for event in target_events}
        pred_keys = set(pred_by_key)
        target_keys = set(target_by_key)
        matched_keys = pred_keys & target_keys
        totals["exact_tp"] += float(len(matched_keys))
        totals["exact_fp"] += float(len(pred_keys - target_keys))
        totals["exact_fn"] += float(len(target_keys - pred_keys))

        pred_family_step = {_event_plan_family_step_key(event) for event in pred_events}
        target_family_step = {_event_plan_family_step_key(event) for event in target_events}
        matched_family_step = pred_family_step & target_family_step
        totals["family_step_tp"] += float(len(matched_family_step))
        totals["family_step_fp"] += float(len(pred_family_step - target_family_step))
        totals["family_step_fn"] += float(len(target_family_step - pred_family_step))

        for key in matched_keys:
            pred_velocity = float(pred_by_key[key].get("velocity", 0.0) or 0.0)
            target_velocity = float(target_by_key[key].get("velocity", 0.0) or 0.0)
            totals["matched_velocity_abs"] += abs(float(pred_velocity) - float(target_velocity))
            totals["matched_velocity_count"] += 1.0

        pred_group_keysets: dict[str, set[tuple[str, int, int]]] = {group_name: set() for group_name in EVENT_PLAN_GROUP_NAMES}
        for event in pred_events:
            key = _event_plan_key(event)
            for group_name in _event_plan_predicted_groups(event, batch, int(batch_idx)):
                if str(group_name) in pred_group_keysets:
                    pred_group_keysets[str(group_name)].add(key)
        for group_name in EVENT_PLAN_GROUP_NAMES:
            target_group_keys = target_group_keysets[str(group_name)][int(batch_idx)]
            totals[f"{group_name}_hit"] += float(len(target_group_keys & pred_group_keysets[str(group_name)]))
            totals[f"{group_name}_target"] += float(len(target_group_keys))
        totals["examples"] += 1.0


def _event_plan_score(metrics: Mapping[str, float]) -> float:
    return float(
        (0.28 * float(metrics.get("event_plan_family_step_f1", 0.0)))
        + (0.18 * float(metrics.get("event_plan_exact_f1", 0.0)))
        + (0.13 * float(metrics.get("event_plan_tom_fill_recall", 0.0)))
        + (0.11 * float(metrics.get("event_plan_kick_ghost_recall", 0.0)))
        + (0.08 * float(metrics.get("event_plan_snare_ghost_recall", 0.0)))
        + (0.05 * float(metrics.get("event_plan_snare_roll_drag_recall", 0.0)))
        + (0.05 * float(metrics.get("event_plan_snare_roll_run_recall", 0.0)))
        + (0.05 * float(metrics.get("event_plan_crash_recall", 0.0)))
        + (0.04 * float(metrics.get("event_plan_ride_recall", 0.0)))
        + (0.03 * float(metrics.get("event_plan_velocity_score", 0.0)))
    )


def _finalize_event_plan_metrics(totals: Mapping[str, float]) -> dict[str, float]:
    exact_precision, exact_recall, exact_f1 = _precision_recall_f1(
        float(totals.get("exact_tp", 0.0)),
        float(totals.get("exact_fp", 0.0)),
        float(totals.get("exact_fn", 0.0)),
    )
    family_step_precision, family_step_recall, family_step_f1 = _precision_recall_f1(
        float(totals.get("family_step_tp", 0.0)),
        float(totals.get("family_step_fp", 0.0)),
        float(totals.get("family_step_fn", 0.0)),
    )
    matched_count = float(totals.get("matched_velocity_count", 0.0))
    matched_velocity_mae = (
        float(totals.get("matched_velocity_abs", 0.0)) / float(matched_count)
        if float(matched_count) > 0.0
        else 0.5
    )
    metrics = {
        "event_plan_examples": float(totals.get("examples", 0.0)),
        "event_plan_exact_precision": float(exact_precision),
        "event_plan_exact_recall": float(exact_recall),
        "event_plan_exact_f1": float(exact_f1),
        "event_plan_family_step_precision": float(family_step_precision),
        "event_plan_family_step_recall": float(family_step_recall),
        "event_plan_family_step_f1": float(family_step_f1),
        "event_plan_matched_velocity_mae": float(matched_velocity_mae),
        "event_plan_velocity_score": _clamp01(1.0 - (float(matched_velocity_mae) / 0.5)),
    }
    for group_name in EVENT_PLAN_GROUP_NAMES:
        metrics[f"event_plan_{group_name}_recall"] = _safe_div(
            float(totals.get(f"{group_name}_hit", 0.0)),
            float(totals.get(f"{group_name}_target", 0.0)),
        )
    metrics["event_plan_score"] = _event_plan_score(metrics)
    return metrics


def _symbolic_event_plan_metrics_from_predictions(
    pred_events_batch: list[list[dict[str, Any]]],
    batch: Mapping[str, Any],
    *,
    class_names: list[str] | None = None,
    budget_group_names: list[str] | None = None,
) -> dict[str, float]:
    names = [str(x) for x in list(class_names or batch.get("class_names") or [])]
    if not names:
        names = [f"family_{idx}" for idx in range(int(torch.as_tensor(batch["target_presence"]).shape[1]))]
    groups = [str(x) for x in list(budget_group_names or batch.get("ornament_budget_group_names") or EVENT_PLAN_GROUP_NAMES)]
    totals = _empty_event_plan_totals()
    _update_event_plan_totals(
        totals,
        pred_events_batch,
        batch,
        class_names=names,
        budget_group_names=groups,
    )
    return _finalize_event_plan_metrics(totals)


def _checkpoint_metric_value(row: Mapping[str, Any], checkpoint_metric: str) -> float | None:
    if str(checkpoint_metric) == "event_plan_score":
        key = "val_event_plan_score"
    elif str(checkpoint_metric) == "selection_score":
        key = "val_selection_score"
    elif str(checkpoint_metric) == "val_loss":
        key = "val_loss"
    else:
        raise ValueError(f"unsupported checkpoint metric: {checkpoint_metric!r}")
    value = row.get(str(key))
    if value is None:
        return None
    return float(value)


def _checkpoint_metric_is_better(candidate: float, best: float, checkpoint_metric: str) -> bool:
    if str(checkpoint_metric) == "val_loss":
        return float(candidate) < float(best)
    return float(candidate) > float(best)


def _events_to_step_matrix(
    events: list[dict[str, Any]],
    *,
    class_names: list[str],
    value_key: str = "velocity",
) -> torch.Tensor:
    matrix = torch.zeros((int(len(class_names)), 16), dtype=torch.float32)
    family_to_idx = {str(name): idx for idx, name in enumerate(class_names)}
    for event in list(events):
        family = str(event.get("family") or "")
        if family not in family_to_idx:
            continue
        step = int(event.get("step", -1))
        if not (0 <= int(step) < 16):
            continue
        value = float(event.get(value_key, 1.0) or 0.0)
        matrix[int(family_to_idx[family]), int(step)] = max(
            float(matrix[int(family_to_idx[family]), int(step)].item()),
            float(value),
        )
    return matrix


def _write_eval_preview_plot(
    *,
    out_path: Path,
    preview_payload: Mapping[str, Any],
    rendered_batch: Mapping[str, Any],
    class_names: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sketch_hits = torch.as_tensor(preview_payload["sketch_hits"], dtype=torch.float32)
    pred_mat = _events_to_step_matrix(
        list(preview_payload["pred_events"]),
        class_names=class_names,
        value_key="velocity",
    )
    target_mat = _events_to_step_matrix(
        list(preview_payload["target_events"]),
        class_names=class_names,
        value_key="velocity",
    )
    rendered_grid = torch.as_tensor(rendered_batch["grid"][0], dtype=torch.float32)
    feature_names = [str(x) for x in list(rendered_batch.get("feature_row_names") or [])]
    if not feature_names:
        feature_names = [f"feat_{idx}" for idx in range(int(rendered_grid.shape[0]))]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 11),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.6, 1.6, 2.2]},
    )
    title = (
        f"epoch {int(preview_payload['epoch'])} | "
        f"source={preview_payload.get('source_id', '')} | "
        f"bpm={float(preview_payload.get('bpm', 0.0)):.1f} | "
        f"pred={int(preview_payload.get('pred_event_count', 0))} "
        f"target={int(preview_payload.get('target_event_count', 0))}"
    )
    fig.suptitle(title)

    image = axes[0].imshow(sketch_hits.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input sketch hits")
    axes[0].set_yticks(range(len(preview_payload.get("sketch_family_names") or [])))
    axes[0].set_yticklabels([str(x) for x in list(preview_payload.get("sketch_family_names") or [])])
    axes[0].set_xticks(range(16))
    axes[0].set_xlabel("16th step")
    plt.colorbar(image, ax=axes[0], fraction=0.025, pad=0.01)

    image = axes[1].imshow(pred_mat.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted events by family (velocity)")
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_yticklabels(class_names)
    axes[1].set_xticks(range(16))
    axes[1].set_xlabel("16th step")
    plt.colorbar(image, ax=axes[1], fraction=0.025, pad=0.01)

    image = axes[2].imshow(target_mat.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title("Target events by family (velocity)")
    axes[2].set_yticks(range(len(class_names)))
    axes[2].set_yticklabels(class_names)
    axes[2].set_xticks(range(16))
    axes[2].set_xlabel("16th step")
    plt.colorbar(image, ax=axes[2], fraction=0.025, pad=0.01)

    image = axes[3].imshow(rendered_grid.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[3].set_title("Rendered diffusion conditioning grid [24,T]")
    axes[3].set_yticks(range(len(feature_names)))
    axes[3].set_yticklabels(feature_names, fontsize=7)
    axes[3].set_xlabel("250 Hz grid frame")
    plt.colorbar(image, ax=axes[3], fraction=0.025, pad=0.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_eval_preview(
    *,
    out_dir: Path,
    epoch: int,
    model: SketchExpander,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    sample_idx: int,
    grid_frame_rate: float,
    seed: int,
) -> dict[str, Any]:
    batch_size = int(torch.as_tensor(raw_batch["sketch_hits"]).shape[0])
    if not (0 <= int(sample_idx) < int(batch_size)):
        raise IndexError(f"preview sample idx {sample_idx} out of range for eval batch size {batch_size}")
    batch = _move_batch(raw_batch, device)
    with torch.no_grad():
        outputs = model(batch["sketch_hits"], batch["sketch_vel"], batch["controls"])
    pred_events_batch = decode_event_plan(
        {key: value.detach().cpu() for key, value in outputs.items()},
        sketch_hits=torch.as_tensor(raw_batch["sketch_hits"], dtype=torch.float32),
        sketch_vel=torch.as_tensor(raw_batch["sketch_vel"], dtype=torch.float32),
        controls=torch.as_tensor(raw_batch["controls"], dtype=torch.float32),
        class_names=model.class_names,
        class_id_vocab_sizes=model.class_id_vocab_sizes,
        control_names=model.cfg.control_names,
        budget_group_names=model.budget_group_names,
        budget_max_counts=model.budget_max_counts,
        seed=int(seed) + int(epoch),
    )
    pred_events = list(pred_events_batch[int(sample_idx)])
    target_events = _target_events_from_batch(raw_batch, int(sample_idx))
    bpm_t = torch.as_tensor(raw_batch["bpm"], dtype=torch.float32).view(-1)
    bpm = float(bpm_t[int(sample_idx)].item()) if int(bpm_t.numel()) > int(sample_idx) else 120.0
    if not float(bpm) > 0.0:
        bpm = 120.0
    rendered_batch = build_diffusion_batch_from_events(
        [pred_events],
        bpm=float(bpm),
        grid_frame_rate=float(grid_frame_rate),
    )
    preview_dir = out_dir / "eval_examples"
    preview_dir.mkdir(parents=True, exist_ok=True)
    grid_rel = Path("eval_examples") / f"epoch_{int(epoch):03d}_grid_batch.pt"
    torch.save(rendered_batch, out_dir / grid_rel)
    source_ids = list(raw_batch.get("source_id") or [])
    split_names = list(raw_batch.get("split") or [])
    beat_indices = torch.as_tensor(raw_batch.get("beat_index", torch.zeros((batch_size,), dtype=torch.long))).view(-1)
    preview_payload = {
        "epoch": int(epoch),
        "sample_idx": int(sample_idx),
        "source_id": str(source_ids[int(sample_idx)]) if int(sample_idx) < int(len(source_ids)) else "",
        "split": str(split_names[int(sample_idx)]) if int(sample_idx) < int(len(split_names)) else "",
        "beat_index": int(beat_indices[int(sample_idx)].item()) if int(beat_indices.numel()) > int(sample_idx) else 0,
        "bpm": float(bpm),
        "control_names": list(raw_batch.get("control_names") or []),
        "controls": torch.as_tensor(raw_batch["controls"][int(sample_idx)], dtype=torch.float32).detach().cpu().tolist(),
        "public_controls": sketch_controls_to_public_dict(
            torch.as_tensor(raw_batch["controls"][int(sample_idx)], dtype=torch.float32),
            control_names=list(raw_batch.get("control_names") or model.cfg.control_names),
        ),
        "sketch_family_names": list(raw_batch.get("sketch_family_names") or []),
        "sketch_hits": torch.as_tensor(raw_batch["sketch_hits"][int(sample_idx)], dtype=torch.float32).detach().cpu().tolist(),
        "sketch_vel": torch.as_tensor(raw_batch["sketch_vel"][int(sample_idx)], dtype=torch.float32).detach().cpu().tolist(),
        "pred_events": pred_events,
        "target_events": target_events,
        "pred_event_count": int(len(pred_events)),
        "target_event_count": int(len(target_events)),
        "grid_batch_pt": str(grid_rel),
    }
    json_rel = Path("eval_examples") / f"epoch_{int(epoch):03d}_preview.json"
    plot_rel = Path("eval_examples") / f"epoch_{int(epoch):03d}_preview.png"
    class_names = [str(x) for x in list(raw_batch.get("class_names") or model.class_names)]
    _write_eval_preview_plot(
        out_path=out_dir / plot_rel,
        preview_payload=preview_payload,
        rendered_batch=rendered_batch,
        class_names=class_names,
    )
    preview_payload["plot_png"] = str(plot_rel)
    (out_dir / json_rel).write_text(json.dumps(preview_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "eval_preview_json": str(json_rel),
        "eval_preview_grid_batch": str(grid_rel),
        "eval_preview_plot": str(plot_rel),
        "eval_preview_pred_events": int(len(pred_events)),
        "eval_preview_target_events": int(len(target_events)),
    }


def main() -> None:
    args = _parse_args()
    if int(args.event_plan_eval_every) < 1:
        raise ValueError("--event-plan-eval-every must be >= 1")
    if bool(args.no_event_plan_eval) and str(args.checkpoint_metric) == "event_plan_score":
        raise ValueError("--checkpoint-metric event_plan_score requires symbolic event-plan eval")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        if bool(args.overwrite):
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise FileExistsError(f"out-dir already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    pin_memory = bool(device.type == "cuda")

    train_loader = build_sketch_dataloader(
        args.cache_root,
        split=str(args.train_split),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        max_items=int(args.max_train_items),
        max_slots=int(args.max_slots),
        pin_memory=pin_memory,
    )
    val_loader = build_sketch_dataloader(
        args.cache_root,
        split=str(args.val_split),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        max_items=int(args.max_val_items),
        max_slots=int(args.max_slots),
        pin_memory=pin_memory,
    )
    preview_raw_batch = None if bool(args.no_eval_preview) else next(iter(val_loader))
    cfg = SketchExpanderConfig(
        max_slots=int(args.max_slots),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
    )
    model = SketchExpander(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    run_config = {
        **vars(args),
        "out_dir": str(out_dir),
        "resolved_device": str(resolved_device),
        "model_cfg": asdict(cfg),
        "num_parameters": int(sum(int(param.numel()) for param in model.parameters())),
        "train_batches_per_epoch": int(len(train_loader)),
        "val_batches_per_epoch": int(len(val_loader)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    history_rows: list[dict[str, Any]] = []
    history_jsonl = out_dir / "history.jsonl"
    best_val_loss = float("inf")
    best_selection_score = float("-inf")
    best_event_plan_score = float("-inf")
    best_checkpoint_score = float("inf") if str(args.checkpoint_metric) == "val_loss" else float("-inf")
    for epoch in range(int(args.epochs)):
        model.train()
        train_total: dict[str, float] = {}
        train_batches = 0
        for raw_batch in _progress(train_loader, desc=f"train_sketch[{epoch:03d}]"):
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["sketch_hits"], batch["sketch_vel"], batch["controls"])
            loss_payload = sketch_expander_loss(
                outputs,
                batch,
                class_id_vocab_sizes=FAMILY_STATE_ID_VOCAB_SIZES,
            )
            loss = loss_payload["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for key, value in loss_payload.items():
                train_total[str(key)] = train_total.get(str(key), 0.0) + float(value.detach().item())
            train_batches += 1

        model.eval()
        val_total: dict[str, float] = {}
        val_batches = 0
        run_event_plan_eval = bool(
            not bool(args.no_event_plan_eval) and (int(epoch) % int(args.event_plan_eval_every) == 0)
        )
        event_plan_totals = _empty_event_plan_totals() if bool(run_event_plan_eval) else None
        with torch.no_grad():
            for raw_batch in _progress(val_loader, desc=f"val_sketch[{epoch:03d}]"):
                batch = _move_batch(raw_batch, device)
                outputs = model(batch["sketch_hits"], batch["sketch_vel"], batch["controls"])
                loss_payload = sketch_expander_loss(
                    outputs,
                    batch,
                    class_id_vocab_sizes=FAMILY_STATE_ID_VOCAB_SIZES,
                )
                for key, value in loss_payload.items():
                    val_total[str(key)] = val_total.get(str(key), 0.0) + float(value.detach().item())
                if event_plan_totals is not None:
                    class_names = [str(x) for x in list(raw_batch.get("class_names") or model.class_names)]
                    budget_group_names = [
                        str(x) for x in list(raw_batch.get("ornament_budget_group_names") or model.budget_group_names)
                    ]
                    pred_events_batch = decode_event_plan(
                        {key: value.detach().cpu() for key, value in outputs.items()},
                        sketch_hits=torch.as_tensor(raw_batch["sketch_hits"], dtype=torch.float32),
                        sketch_vel=torch.as_tensor(raw_batch["sketch_vel"], dtype=torch.float32),
                        controls=torch.as_tensor(raw_batch["controls"], dtype=torch.float32),
                        class_names=class_names,
                        class_id_vocab_sizes=model.class_id_vocab_sizes,
                        control_names=list(raw_batch.get("control_names") or model.cfg.control_names),
                        budget_group_names=budget_group_names,
                        budget_max_counts=list(raw_batch.get("ornament_budget_max_counts") or model.budget_max_counts),
                        seed=int(args.seed),
                    )
                    _update_event_plan_totals(
                        event_plan_totals,
                        pred_events_batch,
                        raw_batch,
                        class_names=class_names,
                        budget_group_names=budget_group_names,
                    )
                val_batches += 1

        row: dict[str, Any] = {
            "epoch": int(epoch),
            **_mean_metrics(train_total, train_batches, "train"),
            **_mean_metrics(val_total, val_batches, "val"),
        }
        row["val_selection_score"] = _musical_selection_score(row)
        if event_plan_totals is not None:
            row.update({f"val_{key}": value for key, value in _finalize_event_plan_metrics(event_plan_totals).items()})
        if preview_raw_batch is not None:
            row.update(
                _write_eval_preview(
                    out_dir=out_dir,
                    epoch=int(epoch),
                    model=model,
                    raw_batch=preview_raw_batch,
                    device=device,
                    sample_idx=int(args.preview_sample_idx),
                    grid_frame_rate=float(args.preview_grid_frame_rate),
                    seed=int(args.seed),
                )
            )
        history_rows.append(row)
        _append_history(history_jsonl, row)
        _write_history_csv(out_dir / "history.csv", history_rows)
        val_loss = float(row.get("val_loss", float("inf")))
        selection_score = float(row.get("val_selection_score", float("-inf")))
        event_plan_score = float(row.get("val_event_plan_score", float("-inf")))
        if float(val_loss) < float(best_val_loss):
            best_val_loss = float(val_loss)
        if float(selection_score) > float(best_selection_score):
            best_selection_score = float(selection_score)
        if float(event_plan_score) > float(best_event_plan_score):
            best_event_plan_score = float(event_plan_score)
        checkpoint_value = _checkpoint_metric_value(row, str(args.checkpoint_metric))
        is_best = bool(
            checkpoint_value is not None
            and _checkpoint_metric_is_better(float(checkpoint_value), float(best_checkpoint_score), str(args.checkpoint_metric))
        )
        if is_best and checkpoint_value is not None:
            best_checkpoint_score = float(checkpoint_value)
        ckpt_extra = {"run_config": run_config}
        _save_checkpoint(
            out_dir / "last_sketch_expander.pt",
            model=model,
            optimizer=optimizer,
            epoch=int(epoch),
            best_val_loss=float(best_val_loss),
            best_selection_score=float(best_selection_score),
            best_event_plan_score=float(best_event_plan_score),
            checkpoint_metric=str(args.checkpoint_metric),
            best_checkpoint_score=float(best_checkpoint_score),
            extra=ckpt_extra,
        )
        if is_best:
            _save_checkpoint(
                out_dir / "best_sketch_expander.pt",
                model=model,
                optimizer=optimizer,
                epoch=int(epoch),
                best_val_loss=float(best_val_loss),
                best_selection_score=float(best_selection_score),
                best_event_plan_score=float(best_event_plan_score),
                checkpoint_metric=str(args.checkpoint_metric),
                best_checkpoint_score=float(best_checkpoint_score),
                extra=ckpt_extra,
            )
        plan_msg = (
            f" val_plan={row.get('val_event_plan_score', 0.0):.4f}"
            if "val_event_plan_score" in row
            else ""
        )
        print(
            f"epoch={epoch} "
            f"train_loss={row.get('train_loss', 0.0):.5f} "
            f"val_loss={row.get('val_loss', 0.0):.5f} "
            f"val_f1={row.get('val_event_f1', 0.0):.4f} "
            f"val_budget={row.get('val_budget_score', 0.0):.4f} "
            f"val_select={row.get('val_selection_score', 0.0):.4f}"
            f"{plan_msg}"
        )


if __name__ == "__main__":
    main()
