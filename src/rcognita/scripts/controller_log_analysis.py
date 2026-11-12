#!/usr/bin/env python3
"""Compute controller metrics and generate comparison plots from CSV log files.

Examples
--------
Scan a directory recursively and auto-detect controller names from file names or controller folders::

    python3 controller_log_analysis.py /path/to/logs \
        --context "Circle — Velocity 0.25 m/s"

Pass individual files::

    python3 controller_log_analysis.py pp_log.csv stanley_log.csv pid_log.csv mpc_log.csv

Use explicit controller names when file names do not identify them::

    python3 controller_log_analysis.py \
        --log "Pure Pursuit=/path/to/a.csv" \
        --log "Stanley=/path/to/b.csv" \
        --log "PID=/path/to/c.csv" \
        --log "MPC=/path/to/d.csv"

The script writes time-series comparison plots, one bar chart per metric, and a
``metrics_summary.csv`` file into the output directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_MAX_STEERING = 0.5
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "rosdata"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "plots"

CONTROLLER_ORDER = ["Pure Pursuit", "Stanley", "PID", "MPC"]
CONTROLLER_COLORS = {
    "Pure Pursuit": "orange",
    "Stanley": "blue",
    "PID": "red",
    "MPC": "green",
}

TIME_SERIES_PLOTS = {
    "cte": "Cross-track error (m)",
    "heading_error": "Heading error (rad)",
    "velocity": "Velocity (m/s)",
    "steering_cmd": "Steering command (rad)",
}

METRIC_PLOTS = {
    "cte_rmse": ("CTE RMSE", "CTE RMSE"),
    "cte_max_abs": ("Maximum absolute CTE", "Maximum |CTE| (m)"),
    "heading_error_rmse": ("Heading-error RMSE", "Heading-error RMSE (rad)"),
    "steering_rate_rms": ("Steering-rate RMS", "Steering-rate RMS (rad/s)"),
    "total_steering_variation": (
        "Total steering variation",
        "Total steering variation (rad)",
    ),
    "steering_saturation_count": (
        "Steering saturation count",
        "Saturation count",
    ),
    "avg_control_frequency_hz": (
        "Average logger frequency",
        "Average logger frequency (Hz)",
    ),
    "controller_loop_frequency_hz": (
        "Controller loop frequency",
        "Average controller frequency (Hz)",
    ),
    "controller_cpu_usage_percent": (
        "Controller CPU usage",
        "Mean CPU usage (%)",
    ),
    "controller_execution_time_mean_ms": (
        "Controller execution time",
        "Mean execution time (ms)",
    ),
    "controller_execution_time_p95_ms": (
        "Controller execution time (95th percentile)",
        "95th percentile execution time (ms)",
    ),
    "controller_deadline_miss_ratio_percent": (
        "Deadline miss ratio",
        "Deadline miss ratio (%)",
    ),
}

REQUIRED_METRIC_COLUMNS = {"time", "cte", "heading_error", "steering_cmd"}


def rmse(values: np.ndarray) -> float:
    """Return the root-mean-square value, ignoring non-finite samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def read_log(csv_file: Path) -> pd.DataFrame:
    """Read, validate, clean, and time-normalize one CSV log."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:
        raise ValueError(f"Could not read {csv_file}: {exc}") from exc

    # Support BOTH the preserved legacy ROS logs and the new QCar CSVs.
    # Benchmark error uses the same REAR AXLE for all controllers, whereas the
    # original `cte` field describes each controller's chosen tracking axle.
    if 'time' not in df and 't' in df:
        df['time'] = df['t']
    if 'steering_cmd' not in df and 'delta' in df:
        df['steering_cmd'] = df['delta']
    if 'velocity' not in df:
        if 'v_measured' in df and pd.to_numeric(df['v_measured'], errors='coerce').notna().any():
            df['velocity'] = df['v_measured']
        elif 'v' in df:
            df['velocity'] = df['v']
    if 'benchmark_cte' in df:
        df['controller_cte'] = df.get('cte')
        df['cte'] = df['benchmark_cte']
    if 'benchmark_heading_error' in df:
        df['controller_heading_error'] = df.get('heading_error')
        df['heading_error'] = df['benchmark_heading_error']

    missing = sorted(REQUIRED_METRIC_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{csv_file} is missing required column(s): {', '.join(missing)}"
        )

    realtime_columns = {
        "controller_loop_frequency_hz",
        "controller_execution_time_ms",
        "controller_cpu_usage_percent",
        "controller_deadline_miss",
        "controller_deadline_miss_ratio",
        "controller_cycle_count",
        "controller_deadline_miss_count",
    }
    numeric_columns = (
        set(TIME_SERIES_PLOTS).intersection(df.columns)
        | realtime_columns.intersection(df.columns)
        | {"time"}
    )
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{csv_file} contains no valid time samples")

    df["time"] = df["time"] - df["time"].iloc[0]
    return df


def compute_metrics(df: pd.DataFrame, csv_file: Path, max_steering: float) -> dict:
    """Compute the metrics previously produced by compute_metrics.py."""
    time = df["time"].to_numpy(dtype=float)
    cte = df["cte"].to_numpy(dtype=float)
    heading_error = df["heading_error"].to_numpy(dtype=float)
    steering = df["steering_cmd"].to_numpy(dtype=float)

    valid_time = np.isfinite(time)
    time = time[valid_time]

    dt = np.diff(time)
    steering_diff = np.diff(steering)

    # Steering rate is valid only for finite, positive time increments.
    rate_count = min(dt.size, steering_diff.size)
    dt_for_rate = dt[:rate_count]
    diff_for_rate = steering_diff[:rate_count]
    valid_rate_mask = (
        np.isfinite(dt_for_rate)
        & (dt_for_rate > 0.0)
        & np.isfinite(diff_for_rate)
    )
    steering_rate = diff_for_rate[valid_rate_mask] / dt_for_rate[valid_rate_mask]

    valid_dt = dt[np.isfinite(dt) & (dt > 0.0)]
    valid_steering_diff = steering_diff[np.isfinite(steering_diff)]
    valid_steering = steering[np.isfinite(steering)]

    duration = float(time[-1] - time[0]) if time.size >= 2 else 0.0
    avg_frequency = (
        float(1.0 / np.mean(valid_dt)) if valid_dt.size > 0 else float("nan")
    )

    def finite_column(name):
        if name not in df.columns:
            return np.array([], dtype=float)
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    loop_frequency_samples = finite_column("controller_loop_frequency_hz")
    execution_time_samples = finite_column("controller_execution_time_ms")
    cpu_usage_samples = finite_column("controller_cpu_usage_percent")
    cycle_count_samples = finite_column("controller_cycle_count")
    deadline_count_samples = finite_column("controller_deadline_miss_count")

    controller_loop_frequency = (
        float(np.mean(loop_frequency_samples))
        if loop_frequency_samples.size else float("nan")
    )
    controller_cpu_usage = (
        float(np.mean(cpu_usage_samples))
        if cpu_usage_samples.size else float("nan")
    )
    controller_cpu_usage_max = (
        float(np.max(cpu_usage_samples))
        if cpu_usage_samples.size else float("nan")
    )
    controller_execution_time_mean = (
        float(np.mean(execution_time_samples))
        if execution_time_samples.size else float("nan")
    )
    controller_execution_time_p95 = (
        float(np.percentile(execution_time_samples, 95))
        if execution_time_samples.size else float("nan")
    )
    controller_execution_time_max = (
        float(np.max(execution_time_samples))
        if execution_time_samples.size else float("nan")
    )

    controller_cycle_count = (
        int(round(cycle_count_samples[-1]))
        if cycle_count_samples.size else 0
    )
    controller_deadline_miss_count = (
        int(round(deadline_count_samples[-1]))
        if deadline_count_samples.size else 0
    )
    controller_deadline_miss_ratio_percent = (
        100.0 * controller_deadline_miss_count / controller_cycle_count
        if controller_cycle_count > 0 else float("nan")
    )

    return {
        "file": str(csv_file),
        "duration_s": duration,
        "cte_rmse": rmse(cte),
        "cte_max_abs": (
            float(np.nanmax(np.abs(cte))) if np.isfinite(cte).any() else float("nan")
        ),
        "heading_error_rmse": rmse(heading_error),
        "steering_rate_rms": rmse(steering_rate),
        "total_steering_variation": float(np.sum(np.abs(valid_steering_diff))),
        "steering_saturation_count": int(
            np.sum(np.abs(valid_steering) >= 0.98 * max_steering)
        ),
        "avg_control_frequency_hz": avg_frequency,
        "controller_loop_frequency_hz": controller_loop_frequency,
        "controller_cpu_usage_percent": controller_cpu_usage,
        "controller_cpu_usage_max_percent": controller_cpu_usage_max,
        "controller_execution_time_mean_ms": controller_execution_time_mean,
        "controller_execution_time_p95_ms": controller_execution_time_p95,
        "controller_execution_time_max_ms": controller_execution_time_max,
        "controller_cycle_count": controller_cycle_count,
        "controller_deadline_miss_count": controller_deadline_miss_count,
        "controller_deadline_miss_ratio_percent": controller_deadline_miss_ratio_percent,
    }


def _controller_name_from_text(text: str) -> str | None:
    """Return a standard controller name when *text* identifies one."""
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    tokens = set(filter(None, normalized.split("_")))

    if (
        normalized.startswith("pp")
        or "purepursuit" in normalized
        or ("pure" in tokens and "pursuit" in tokens)
    ):
        return "Pure Pursuit"
    if normalized.startswith("stanley") or normalized == "st" or "stanley" in tokens:
        return "Stanley"
    if normalized.startswith("pid") or normalized == "pi" or "pid" in tokens:
        return "PID"
    if normalized.startswith("mpc") or normalized == "mp" or "mpc" in tokens:
        return "MPC"

    return None


def detect_controller_name(path: Path, search_root: Path | None = None) -> str:
    """Infer a controller name from the CSV filename or its parent folders.

    This supports both layouts::

        logs/
            pp_run.csv
            stanley_run.csv

    and::

        logs/
            PP/
                run.csv
            Stanley/
                run.csv
            PID/
                run.csv
            MPC/
                run.csv

    Parent folders are checked up to ``search_root`` so nested run folders are
    also supported.
    """
    # Preserve the old behavior first: try to identify the controller from
    # the CSV filename.
    controller = _controller_name_from_text(path.stem)
    if controller is not None:
        return controller

    # New behavior: if the filename itself is generic, infer the controller
    # from the folder hierarchy.
    parent = path.parent
    resolved_root = search_root.resolve() if search_root is not None else None

    while True:
        controller = _controller_name_from_text(parent.name)
        if controller is not None:
            return controller

        if resolved_root is None:
            break

        try:
            if parent.resolve() == resolved_root:
                break
        except OSError:
            if parent == search_root:
                break

        if parent == parent.parent:
            break
        parent = parent.parent

    # Unknown names are still supported; make the file stem readable.
    return re.sub(r"[_-]+", " ", path.stem).strip().title()


def parse_explicit_log(value: str) -> tuple[str, Path]:
    """Parse --log CONTROLLER=FILE."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            '--log must use the format "CONTROLLER=/path/to/file.csv"'
        )

    controller, file_name = value.split("=", 1)
    controller = controller.strip()
    file_name = file_name.strip()
    if not controller or not file_name:
        raise argparse.ArgumentTypeError(
            '--log must use the format "CONTROLLER=/path/to/file.csv"'
        )

    return controller, Path(file_name).expanduser()


def collect_logs(
    input_paths: Iterable[Path],
    explicit_logs: Iterable[tuple[str, Path]],
    filename_filter: str | None = None,
) -> dict[str, Path]:
    """Collect CSV logs from files, directories, and explicit mappings."""
    logs: dict[str, Path] = {}

    for controller, path in explicit_logs:
        if controller in logs:
            raise ValueError(f"Duplicate controller name: {controller}")
        logs[controller] = path

    for input_path in input_paths:
        input_path = input_path.expanduser()
        if input_path.is_dir():
            # Search recursively so controller-specific subfolders are supported.
            #
            # Old layout:
            #   logs/pp.csv
            #   logs/stanley.csv
            #
            # New layout:
            #   logs/PP/run.csv
            #   logs/Stanley/run.csv
            #   logs/PID/run.csv
            #   logs/MPC/run.csv
            #
            # rglob() still finds CSV files placed directly in input_path, so the
            # old layout continues to work without any command-line changes.
            csv_files = sorted(input_path.rglob("*.csv"))

            # Do not accidentally re-read summaries generated by this script.
            csv_files = [p for p in csv_files if p.name != "metrics_summary.csv"]

            # When the hard-coded log directory is used, a context such as
            # "c_0.25" can identify the experiment either in the CSV filename
            # or in one of its containing folders.
            # Only apply the filter when it actually matches at least one file.
            if filename_filter:
                filter_text = filename_filter.lower()
                matching_files = [
                    p for p in csv_files
                    if filter_text in str(p.relative_to(input_path)).lower()
                ]
                if matching_files:
                    csv_files = matching_files

            if not csv_files:
                raise ValueError(
                    f"No CSV files found in directory or its subfolders: {input_path}"
                )
        elif input_path.is_file():
            csv_files = [input_path]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        for csv_file in csv_files:
            controller = detect_controller_name(
                csv_file,
                search_root=input_path if input_path.is_dir() else None,
            )
            if controller in logs:
                raise ValueError(
                    f"More than one log resolved to '{controller}'. "
                    "Rename the files or use --log CONTROLLER=FILE explicitly."
                )
            logs[controller] = csv_file

    if not logs:
        raise ValueError("No logs were supplied")

    for controller, path in logs.items():
        if not path.is_file():
            raise ValueError(f"Log for {controller} does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Log for {controller} is not a CSV file: {path}")

    return logs


def ordered_controllers(controllers: Iterable[str]) -> list[str]:
    """Keep the standard controller order and append any custom names."""
    controllers = list(controllers)
    standard = [name for name in CONTROLLER_ORDER if name in controllers]
    custom = sorted(name for name in controllers if name not in CONTROLLER_ORDER)
    return standard + custom


def controller_color(controller: str, index: int) -> str:
    """Return a stable color for known controllers and a fallback for custom ones."""
    if controller in CONTROLLER_COLORS:
        return CONTROLLER_COLORS[controller]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return cycle[index % len(cycle)]


def add_context(base_title: str, context: str | None) -> str:
    return f"{base_title} — {context}" if context else base_title


def save_time_series_plots(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    context: str | None,
    dpi: int,
    show: bool,
) -> None:
    controllers = ordered_controllers(data)

    for column, ylabel in TIME_SERIES_PLOTS.items():
        available = [name for name in controllers if column in data[name].columns]
        if not available:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for index, controller in enumerate(available):
            df = data[controller]
            valid = np.isfinite(df["time"].to_numpy(dtype=float)) & np.isfinite(
                df[column].to_numpy(dtype=float)
            )
            ax.plot(
                df.loc[valid, "time"].to_numpy(),
                df.loc[valid, column].to_numpy(),
                label=controller,
                color=controller_color(controller, index),
                linewidth=2,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(add_context(f"{ylabel} vs Time", context))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{column}_comparison.png", dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)


def value_label(value: float, metric_name: str) -> str:
    if not np.isfinite(value):
        return "N/A"
    if metric_name == "steering_saturation_count":
        return str(int(round(value)))
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def save_metric_bar_plots(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    context: str | None,
    dpi: int,
    show: bool,
) -> None:
    controllers = ordered_controllers(metrics_df.index)

    for metric_name, (title, ylabel) in METRIC_PLOTS.items():
        if metric_name not in metrics_df.columns:
            continue

        values = pd.to_numeric(
            metrics_df.loc[controllers, metric_name], errors="coerce"
        ).to_numpy(dtype=float)
        plot_values = np.where(np.isfinite(values), values, 0.0)
        colors = [controller_color(name, i) for i, name in enumerate(controllers)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(controllers, plot_values, color=colors, width=0.62)

        ax.set_xlabel("Controller")
        ax.set_ylabel(ylabel)
        ax.set_title(add_context(f"{title} Comparison", context))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        legend_handles = [
            Patch(facecolor=color, label=controller)
            for controller, color in zip(controllers, colors)
        ]
        ax.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=8)

        finite_values = values[np.isfinite(values)]
        maximum = float(np.max(finite_values)) if finite_values.size else 1.0
        offset = 0.018 * maximum if maximum > 0 else 0.02
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                value_label(value, metric_name),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        upper = maximum * 1.20 if maximum > 0 else 1.0
        ax.set_ylim(bottom=0, top=upper)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_name}_comparison.png", dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)


def context_folder_name(context: str | None) -> str | None:
    """Convert context text into a safe output-folder name."""
    if context is None or not context.strip():
        return None

    # Keep letters, numbers, dots, underscores, and hyphens. Replace the rest
    # with underscores so contexts such as "Circle — Velocity 0.25 m/s" are
    # also valid directory names.
    folder_name = re.sub(r"[^A-Za-z0-9._-]+", "_", context.strip())
    folder_name = folder_name.strip("._-")
    return folder_name or "results"


def default_output_dir(context: str | None) -> Path:
    """Return the hard-coded plot root, optionally grouped by context."""
    folder_name = context_folder_name(context)
    return DEFAULT_OUTPUT_ROOT / folder_name if folder_name else DEFAULT_OUTPUT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read controller CSV logs, compute metrics, and produce time-series "
            "and metric-comparison plots."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "CSV log files and/or a directory containing CSV logs (subfolders are searched recursively) "
            "(default: /home/prajwal_086/ros2_ackermann/logs)"
        ),
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        type=parse_explicit_log,
        metavar="CONTROLLER=FILE",
        help="Explicit controller/file mapping; may be supplied more than once",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override. By default results are written under "
            "/home/prajwal_086/ros2_ackermann/plots/<context>"
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        help='Text appended to plot titles, e.g. "Circle — Velocity 0.25 m/s"',
    )
    parser.add_argument(
        "--max-steering",
        type=float,
        default=DEFAULT_MAX_STEERING,
        help=f"Maximum steering magnitude in radians (default: {DEFAULT_MAX_STEERING})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved image resolution (default: 300)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots interactively",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    using_default_log_dir = not args.inputs and not args.log
    if using_default_log_dir:
        args.inputs = [DEFAULT_LOG_DIR]
        print(f"No input path supplied; using hard-coded log directory: {DEFAULT_LOG_DIR}")
    if args.max_steering <= 0:
        parser.error("--max-steering must be greater than zero")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")

    try:
        logs = collect_logs(
            args.inputs,
            args.log,
            filename_filter=args.context if using_default_log_dir else None,
        )
        output_dir = (
            args.output_dir.expanduser()
            if args.output_dir is not None
            else default_output_dir(args.context)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, pd.DataFrame] = {}
        metrics_by_controller: dict[str, dict] = {}

        for controller, csv_file in logs.items():
            df = read_log(csv_file)
            data[controller] = df
            metrics_by_controller[controller] = compute_metrics(
                df, csv_file, args.max_steering
            )

        controller_sequence = ordered_controllers(data)
        metrics_df = pd.DataFrame.from_dict(metrics_by_controller, orient="index")
        metrics_df.index.name = "controller"
        metrics_df = metrics_df.loc[controller_sequence]
        metrics_df.to_csv(output_dir / "metrics_summary.csv")

        save_time_series_plots(data, output_dir, args.context, args.dpi, args.show)
        save_metric_bar_plots(
            metrics_df, output_dir, args.context, args.dpi, args.show
        )

        printable = metrics_df.copy()
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print("\n=== Metrics ===")
            print(printable.to_string(float_format=lambda x: f"{x:.6g}"))
        print(f"\nPlots and metrics saved to: {output_dir.resolve()}")
        return 0

    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
