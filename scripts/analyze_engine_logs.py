from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


WINDOWS_MINUTES = (15, 60, 240)
MEANINGFUL_MOVE_PCT = 0.10


@dataclass
class Snapshot:
    index: int
    ts: datetime
    timestamp: float
    snapshot_id: str
    mode: str
    engine_version: str
    price: float | None
    strategy_bias: str
    strategy_state: str
    strategy_action: str
    market_phase: str
    trade_quality: str
    entry_permission: bool
    entry_timing: str
    confidence: float | None
    execution_confidence: float | None
    entry_score: float | None
    entry_score_tier: str
    risk_state: str
    trigger_state: str
    trigger_level: float | None
    ladder_stage: str
    primary_blocker: str
    hard_fail_count: int | None
    soft_fail_count: int | None
    execution_direction: str
    entry_model: str
    entry_style: str
    reward_to_risk: float | None
    limit_enabled: bool
    limit_side: str
    limit_price: float | None
    invalidation_price: float | None
    take_profit_reference: float | None
    market_entry_allowed: bool | None
    signal_label: str
    candidate_type: str
    candidate_strength: str
    timeframe_setups: dict[str, str]
    timeframe_signals: dict[str, str]
    timeframe_traps: dict[str, str]
    blocking_reasons: tuple[str, ...]


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = f"{value[:-1]}+00:00"
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def as_bool(value: Any) -> bool:
    return bool(value)


def pick_nested(record: dict[str, Any], *path: str) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_text(value: Any, fallback: str = "unknown") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def load_snapshots(path: Path) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = parse_dt(record.get("logged_at"))
            price = as_float(pick_nested(record, "market", "price"))
            if ts is None or price is None:
                continue

            strategy = record.get("strategy") if isinstance(record.get("strategy"), dict) else {}
            trigger = record.get("trigger_context") if isinstance(record.get("trigger_context"), dict) else {}
            entry_validation = (
                record.get("entry_validation")
                if isinstance(record.get("entry_validation"), dict)
                else {}
            )
            execution_plan = (
                record.get("execution_plan") if isinstance(record.get("execution_plan"), dict) else {}
            )
            signal_meta = record.get("signal_meta") if isinstance(record.get("signal_meta"), dict) else {}
            limit_plan = record.get("limit_order_plan")
            if not isinstance(limit_plan, dict):
                limit_plan = execution_plan.get("limit_order_plan")
            if not isinstance(limit_plan, dict):
                limit_plan = {}

            timeframes = record.get("timeframes") if isinstance(record.get("timeframes"), dict) else {}
            timeframe_setups: dict[str, str] = {}
            timeframe_signals: dict[str, str] = {}
            timeframe_traps: dict[str, str] = {}
            for timeframe, payload in timeframes.items():
                if not isinstance(payload, dict):
                    continue
                timeframe_setups[timeframe] = normalize_text(payload.get("setup"))
                timeframe_signals[timeframe] = normalize_text(payload.get("signal"))
                timeframe_traps[timeframe] = normalize_text(payload.get("trap_risk"), "none")

            snapshots.append(
                Snapshot(
                    index=index,
                    ts=ts,
                    timestamp=ts.timestamp(),
                    snapshot_id=normalize_text(record.get("snapshot_id")),
                    mode=normalize_text(record.get("mode")),
                    engine_version=normalize_text(record.get("engine_version")),
                    price=price,
                    strategy_bias=normalize_text(strategy.get("bias")),
                    strategy_state=normalize_text(strategy.get("state")),
                    strategy_action=normalize_text(strategy.get("action")),
                    market_phase=normalize_text(strategy.get("market_phase")),
                    trade_quality=normalize_text(strategy.get("trade_quality")),
                    entry_permission=as_bool(strategy.get("entry_permission")),
                    entry_timing=normalize_text(strategy.get("entry_timing")),
                    confidence=as_float(strategy.get("confidence")),
                    execution_confidence=as_float(strategy.get("execution_confidence")),
                    entry_score=as_float(strategy.get("entry_score")),
                    entry_score_tier=normalize_text(strategy.get("entry_score_tier")),
                    risk_state=normalize_text(strategy.get("risk_state")),
                    trigger_state=normalize_text(trigger.get("state")),
                    trigger_level=as_float(trigger.get("confirmation_level")),
                    ladder_stage=normalize_text(trigger.get("ladder_stage")),
                    primary_blocker=normalize_text(trigger.get("primary_blocker"), "none"),
                    hard_fail_count=as_float(entry_validation.get("hard_fail_count")),
                    soft_fail_count=as_float(entry_validation.get("soft_fail_count")),
                    execution_direction=normalize_text(execution_plan.get("direction")),
                    entry_model=normalize_text(execution_plan.get("entry_model")),
                    entry_style=normalize_text(execution_plan.get("entry_style")),
                    reward_to_risk=as_float(execution_plan.get("reward_to_risk_estimate")),
                    limit_enabled=as_bool(limit_plan.get("enabled")),
                    limit_side=normalize_text(limit_plan.get("side")),
                    limit_price=as_float(limit_plan.get("limit_price")),
                    invalidation_price=as_float(limit_plan.get("invalidation_price")),
                    take_profit_reference=as_float(limit_plan.get("take_profit_reference")),
                    market_entry_allowed=(
                        None
                        if limit_plan.get("market_entry_allowed") is None
                        else as_bool(limit_plan.get("market_entry_allowed"))
                    ),
                    signal_label=normalize_text(record.get("signal_label"), normalize_text(signal_meta.get("label"))),
                    candidate_type=normalize_text(signal_meta.get("candidate_type")),
                    candidate_strength=normalize_text(signal_meta.get("candidate_strength")),
                    timeframe_setups=timeframe_setups,
                    timeframe_signals=timeframe_signals,
                    timeframe_traps=timeframe_traps,
                    blocking_reasons=tuple(
                        str(reason)
                        for reason in entry_validation.get("blocking_reasons", [])
                        if reason
                    ),
                )
            )
    return snapshots


def direction_for(snapshot: Snapshot) -> str:
    for candidate in (
        snapshot.execution_direction,
        snapshot.strategy_bias,
        snapshot.limit_side,
    ):
        if candidate in {"long", "short"}:
            return candidate
        if candidate == "buy":
            return "long"
        if candidate == "sell":
            return "short"
    return "neutral"


def directional_move_pct(direction: str, entry_price: float | None, future_price: float | None) -> float | None:
    if entry_price is None or future_price is None or entry_price <= 0:
        return None
    raw_move = ((future_price - entry_price) / entry_price) * 100.0
    if direction == "long":
        return raw_move
    if direction == "short":
        return -raw_move
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def fmt_num(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    start = int(value // 10) * 10
    end = start + 9
    return f"{start:02d}-{end:02d}"


def score_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 30:
        return "00-29"
    if value < 45:
        return "30-44"
    if value < 60:
        return "45-59"
    if value < 75:
        return "60-74"
    return "75+"


def summarize_moves(rows: list[dict[str, Any]], window: str) -> dict[str, Any]:
    moves = [row[f"move_{window}"] for row in rows if row.get(f"move_{window}") is not None]
    if not moves:
        return {
            "count": 0,
            "win_rate": None,
            "meaningful_win_rate": None,
            "meaningful_loss_rate": None,
            "avg": None,
            "median": None,
            "p25": None,
            "p75": None,
        }
    wins = [move for move in moves if move > 0]
    meaningful_wins = [move for move in moves if move >= MEANINGFUL_MOVE_PCT]
    meaningful_losses = [move for move in moves if move <= -MEANINGFUL_MOVE_PCT]
    return {
        "count": len(moves),
        "win_rate": len(wins) / len(moves) * 100.0,
        "meaningful_win_rate": len(meaningful_wins) / len(moves) * 100.0,
        "meaningful_loss_rate": len(meaningful_losses) / len(moves) * 100.0,
        "avg": sum(moves) / len(moves),
        "median": median(moves),
        "p25": percentile(moves, 0.25),
        "p75": percentile(moves, 0.75),
    }


def grouped_table(
    rows: list[dict[str, Any]],
    key: str,
    window: str,
    min_count: int,
    limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[normalize_text(row.get(key))].append(row)

    table = []
    for group_key, group_rows in groups.items():
        summary = summarize_moves(group_rows, window)
        if summary["count"] < min_count:
            continue
        table.append({"group": group_key, **summary})

    return sorted(
        table,
        key=lambda item: (item["avg"] if item["avg"] is not None else -999, item["count"]),
        reverse=True,
    )[:limit]


def write_group_csv(path: Path, table: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "count",
        "win_rate",
        "meaningful_win_rate",
        "meaningful_loss_rate",
        "avg",
        "median",
        "p25",
        "p75",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({field: row.get(field) for field in fields})


def markdown_table(table: list[dict[str, Any]], title_col: str = "Group") -> str:
    if not table:
        return "_No groups met the minimum sample threshold._"
    lines = [
        f"| {title_col} | N | Win % | >= {MEANINGFUL_MOVE_PCT:.2f}% % | <= -{MEANINGFUL_MOVE_PCT:.2f}% % | Avg % | Median % | P25 % | P75 % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["count"]),
                    fmt_num(row["win_rate"], 1),
                    fmt_num(row["meaningful_win_rate"], 1),
                    fmt_num(row["meaningful_loss_rate"], 1),
                    fmt_num(row["avg"], 3),
                    fmt_num(row["median"], 3),
                    fmt_num(row["p25"], 3),
                    fmt_num(row["p75"], 3),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_rows(snapshots: list[Snapshot]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[Snapshot]] = defaultdict(list)
    for snapshot in snapshots:
        by_mode[snapshot.mode].append(snapshot)

    timelines: dict[str, tuple[list[float], list[Snapshot]]] = {}
    for mode, mode_snapshots in by_mode.items():
        ordered = sorted(mode_snapshots, key=lambda item: item.timestamp)
        timelines[mode] = ([item.timestamp for item in ordered], ordered)

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        direction = direction_for(snapshot)
        if direction not in {"long", "short"}:
            continue

        mode_times, mode_snapshots = timelines[snapshot.mode]
        row: dict[str, Any] = {
            "snapshot_id": snapshot.snapshot_id,
            "logged_at": snapshot.ts.isoformat(),
            "mode": snapshot.mode,
            "engine_version": snapshot.engine_version,
            "direction": direction,
            "price": snapshot.price,
            "strategy_bias": snapshot.strategy_bias,
            "strategy_state": snapshot.strategy_state,
            "strategy_action": snapshot.strategy_action,
            "market_phase": snapshot.market_phase,
            "trade_quality": snapshot.trade_quality,
            "entry_permission": snapshot.entry_permission,
            "entry_timing": snapshot.entry_timing,
            "confidence_bucket": confidence_bucket(snapshot.confidence),
            "execution_confidence_bucket": confidence_bucket(snapshot.execution_confidence),
            "entry_score_bucket": score_bucket(snapshot.entry_score),
            "entry_score_tier": snapshot.entry_score_tier,
            "risk_state": snapshot.risk_state,
            "trigger_state": snapshot.trigger_state,
            "ladder_stage": snapshot.ladder_stage,
            "primary_blocker": snapshot.primary_blocker,
            "entry_model": snapshot.entry_model,
            "entry_style": snapshot.entry_style,
            "limit_enabled": snapshot.limit_enabled,
            "candidate_type": snapshot.candidate_type,
            "candidate_strength": snapshot.candidate_strength,
            "setup_5m": snapshot.timeframe_setups.get("5m", "unknown"),
            "setup_15m": snapshot.timeframe_setups.get("15m", "unknown"),
            "setup_1h": snapshot.timeframe_setups.get("1h", "unknown"),
            "signal_5m": snapshot.timeframe_signals.get("5m", "unknown"),
            "trap_5m": snapshot.timeframe_traps.get("5m", "none"),
            "trap_15m": snapshot.timeframe_traps.get("15m", "none"),
            "blocking_reasons": "; ".join(snapshot.blocking_reasons),
        }

        for minutes in WINDOWS_MINUTES:
            target_ts = snapshot.timestamp + minutes * 60
            future_index = bisect.bisect_left(mode_times, target_ts)
            if future_index < len(mode_snapshots):
                future = mode_snapshots[future_index]
                row[f"future_price_{minutes}m"] = future.price
                row[f"move_{minutes}m"] = directional_move_pct(direction, snapshot.price, future.price)
            else:
                row[f"future_price_{minutes}m"] = None
                row[f"move_{minutes}m"] = None
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze(path: Path, out_dir: Path, min_count: int) -> dict[str, Path]:
    snapshots = load_snapshots(path)
    if not snapshots:
        raise SystemExit(f"No usable snapshots found in {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(snapshots)

    first_ts = min(item.ts for item in snapshots)
    last_ts = max(item.ts for item in snapshots)
    latest_version = max(snapshots, key=lambda item: item.ts).engine_version
    versions = Counter(item.engine_version for item in snapshots)
    modes = Counter(item.mode for item in snapshots)
    permissions = Counter(item.entry_permission for item in snapshots)
    states = Counter(item.strategy_state for item in snapshots)
    actions = Counter(item.strategy_action for item in snapshots)
    blockers = Counter()
    for item in snapshots:
        blockers.update(item.blocking_reasons)

    candidate_rows = rows
    window = "60m"
    overall = summarize_moves(candidate_rows, window)
    allowed = [row for row in candidate_rows if row["entry_permission"] is True]
    blocked = [row for row in candidate_rows if row["entry_permission"] is False]
    allowed_summary = summarize_moves(allowed, window)
    blocked_summary = summarize_moves(blocked, window)
    latest_rows = [row for row in candidate_rows if row["engine_version"] == latest_version]
    latest_summaries = {
        f"{minutes}m": summarize_moves(latest_rows, f"{minutes}m")
        for minutes in WINDOWS_MINUTES
    }
    latest_by_state = grouped_table(latest_rows, "strategy_state", window, 10, 12)
    latest_by_entry_model = grouped_table(latest_rows, "entry_model", window, 10, 12)
    latest_by_permission = grouped_table(latest_rows, "entry_permission", window, 10, 12)

    group_specs = [
        ("by_engine_version", "engine_version", 20),
        ("by_mode", "mode", 20),
        ("by_strategy_state", "strategy_state", 20),
        ("by_strategy_action", "strategy_action", 20),
        ("by_entry_permission", "entry_permission", 20),
        ("by_trade_quality", "trade_quality", 20),
        ("by_entry_score_tier", "entry_score_tier", 20),
        ("by_entry_model", "entry_model", 20),
        ("by_setup_5m", "setup_5m", 20),
        ("by_setup_15m", "setup_15m", 20),
        ("by_primary_blocker", "primary_blocker", 20),
        ("by_confidence_bucket", "confidence_bucket", 20),
        ("by_entry_score_bucket", "entry_score_bucket", 20),
    ]
    tables: dict[str, list[dict[str, Any]]] = {}
    for name, key, limit in group_specs:
        tables[name] = grouped_table(candidate_rows, key, window, min_count, limit)
        write_group_csv(out_dir / f"{name}.csv", tables[name])

    write_rows_csv(out_dir / "directional_outcomes.csv", rows)

    top_blockers = blockers.most_common(12)
    report_lines = [
        "# XRP Engine Snapshot Analysis",
        "",
        f"Source: `{path}`",
        f"Generated from {len(snapshots):,} usable snapshots and {len(rows):,} directional snapshots.",
        f"Date range: {first_ts.isoformat()} to {last_ts.isoformat()}",
        "",
        "## Coverage",
        "",
        f"- Engine versions: {', '.join(f'{key} ({value:,})' for key, value in versions.most_common())}",
        f"- Modes: {', '.join(f'{key} ({value:,})' for key, value in modes.most_common())}",
        f"- Entry permission: allowed {permissions[True]:,}, blocked {permissions[False]:,}",
        f"- Top states: {', '.join(f'{key} ({value:,})' for key, value in states.most_common(8))}",
        f"- Top actions: {', '.join(f'{key} ({value:,})' for key, value in actions.most_common(8))}",
        "",
        "## 60 Minute Directional Baseline",
        "",
        (
            f"Directional snapshots had a {fmt_num(overall['win_rate'], 1)}% simple win rate, "
            f"{fmt_num(overall['meaningful_win_rate'], 1)}% moved at least "
            f"{MEANINGFUL_MOVE_PCT:.2f}% in the suggested direction, and "
            f"{fmt_num(overall['meaningful_loss_rate'], 1)}% moved at least "
            f"{MEANINGFUL_MOVE_PCT:.2f}% against it. Average 60m directional move was "
            f"{fmt_num(overall['avg'], 3)}%."
        ),
        "",
        "| Permission | N | Win % | Meaningful Win % | Meaningful Loss % | Avg % | Median % |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Allowed | {allowed_summary['count']} | {fmt_num(allowed_summary['win_rate'], 1)} | "
            f"{fmt_num(allowed_summary['meaningful_win_rate'], 1)} | "
            f"{fmt_num(allowed_summary['meaningful_loss_rate'], 1)} | "
            f"{fmt_num(allowed_summary['avg'], 3)} | {fmt_num(allowed_summary['median'], 3)} |"
        ),
        (
            f"| Blocked/watch | {blocked_summary['count']} | {fmt_num(blocked_summary['win_rate'], 1)} | "
            f"{fmt_num(blocked_summary['meaningful_win_rate'], 1)} | "
            f"{fmt_num(blocked_summary['meaningful_loss_rate'], 1)} | "
            f"{fmt_num(blocked_summary['avg'], 3)} | {fmt_num(blocked_summary['median'], 3)} |"
        ),
        "",
        f"## Latest Version: {latest_version}",
        "",
        f"Latest-version directional snapshots: {len(latest_rows):,}.",
        "",
        "| Window | N | Win % | Meaningful Win % | Meaningful Loss % | Avg % | Median % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for latest_window, latest_summary in latest_summaries.items():
        report_lines.append(
            f"| {latest_window} | {latest_summary['count']} | "
            f"{fmt_num(latest_summary['win_rate'], 1)} | "
            f"{fmt_num(latest_summary['meaningful_win_rate'], 1)} | "
            f"{fmt_num(latest_summary['meaningful_loss_rate'], 1)} | "
            f"{fmt_num(latest_summary['avg'], 3)} | {fmt_num(latest_summary['median'], 3)} |"
        )
    report_lines.extend(
        [
            "",
            "### Latest By Entry Permission",
            markdown_table(latest_by_permission, "Permission"),
            "",
            "### Latest By Strategy State",
            markdown_table(latest_by_state, "State"),
            "",
            "### Latest By Entry Model",
            markdown_table(latest_by_entry_model, "Entry Model"),
            "",
            "## Best 60 Minute Groups",
            "",
            "### By Strategy State",
            markdown_table(tables["by_strategy_state"], "State"),
            "",
            "### By Entry Model",
            markdown_table(tables["by_entry_model"], "Entry Model"),
            "",
            "### By 5m Setup",
            markdown_table(tables["by_setup_5m"], "5m Setup"),
            "",
            "### By Entry Score Tier",
            markdown_table(tables["by_entry_score_tier"], "Entry Score Tier"),
            "",
            "### By Engine Version",
            markdown_table(tables["by_engine_version"], "Version"),
            "",
            "### By Confidence Bucket",
            markdown_table(tables["by_confidence_bucket"], "Confidence"),
            "",
            "## Top Blocking Reasons",
            "",
        ]
    )
    if top_blockers:
        report_lines.extend([f"- {reason}: {count:,}" for reason, count in top_blockers])
    else:
        report_lines.append("_No blocking reasons were recorded._")

    report_lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `directional_outcomes.csv`: one row per directional snapshot with 15m, 60m, and 240m forward moves.",
            "- `by_*.csv`: grouped 60m outcome summaries used by this report.",
            "",
            "Notes: forward outcomes use the next snapshot at or after each time window, grouped within the same strategy mode. "
            "This is a signal-quality analysis, not an exchange-fill backtest; limit order fills and intraperiod touches are approximated by snapshot prices only.",
        ]
    )

    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_path = out_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": str(path),
                "snapshots": len(snapshots),
                "directional_snapshots": len(rows),
                "first_logged_at": first_ts.isoformat(),
                "last_logged_at": last_ts.isoformat(),
                "versions": versions,
                "modes": modes,
                "overall_60m": overall,
                "allowed_60m": allowed_summary,
                "blocked_60m": blocked_summary,
                "top_blockers": top_blockers,
            },
            indent=2,
            default=dict,
        ),
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "summary": summary_path,
        "outcomes": out_dir / "directional_outcomes.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze XRP engine JSONL snapshots.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("engine_logs/engine_snapshots.jsonl"),
        help="Path to engine_snapshots.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("engine_logs/analysis"),
        help="Output directory for the markdown report and CSV summaries.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=25,
        help="Minimum sample count for grouped tables.",
    )
    args = parser.parse_args()
    outputs = analyze(args.log, args.out, args.min_count)
    for name, output_path in outputs.items():
        print(f"{name}: {output_path}")


if __name__ == "__main__":
    main()
