#!/usr/bin/env python3
"""
Smooth the per-client `rate` traces with a short-term Gaussian filter followed by
an EWMA applied within active windows. The output overwrites the original pickle.
"""
from __future__ import annotations

import argparse
import math
import pickle
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


TRACE_HORIZON = 172800  # 48 hours expressed in seconds
DEFAULT_MAX_INTERVAL = 1800  # 30 minutes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth client `rate` traces in-place.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to clients pickle (OrderedDict of client entries).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Half-width of the Gaussian smoothing window in seconds (W).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Standard deviation of the Gaussian kernel in seconds.",
    )
    parser.add_argument(
        "--ewma_half_life",
        type=float,
        default=120.0,
        help="Half-life (in seconds) for the EWMA smoothing inside active sub-intervals.",
    )
    parser.add_argument(
        "--max_sub_interval",
        type=float,
        default=DEFAULT_MAX_INTERVAL,
        help="Maximum length (seconds) of each active sub-interval for EWMA.",
    )
    return parser.parse_args()


def gaussian_smooth(
    timestamps: Sequence[float], values: Sequence[float], window: float, sigma: float
) -> List[float]:
    if sigma <= 0.0:
        raise ValueError("`sigma` must be positive for Gaussian smoothing.")
    if window < 0.0:
        raise ValueError("`window` must be non-negative.")

    denom = 2.0 * sigma * sigma
    times = list(timestamps)
    series = list(values)
    if len(times) != len(series):
        raise ValueError("Mismatched timestamps and values lengths for Gaussian smoothing.")

    smoothed: List[float] = [0.0] * len(series)

    for idx, t in enumerate(times):
        left = bisect_left(times, t - window)
        right = bisect_right(times, t + window)
        if left >= right:
            smoothed[idx] = series[idx]
            continue

        weight_sum = 0.0
        weighted_total = 0.0
        for j in range(left, right):
            diff = times[j] - t
            weight = math.exp(-(diff * diff) / denom)
            weight_sum += weight
            weighted_total += weight * series[j]
        if weight_sum == 0.0:
            smoothed[idx] = series[idx]
        else:
            smoothed[idx] = weighted_total / weight_sum
    return smoothed


def build_active_intervals(
    active: Sequence[float], inactive: Sequence[float], horizon: float = TRACE_HORIZON
) -> List[Tuple[float, float]]:
    active = list(active)
    inactive = list(inactive)

    if not active and not inactive:
        return []

    active_idx = 0
    inactive_idx = 0
    intervals: List[Tuple[float, float]] = []

    if active and active[0] == 0:
        current_state = "active"
        current_time = float(active[active_idx])
        active_idx += 1
    elif inactive and inactive[0] == 0:
        current_state = "inactive"
        current_time = float(inactive[inactive_idx])
        inactive_idx += 1
    else:
        current_state = "inactive"
        current_time = 0.0

    while current_time < horizon and (
        active_idx < len(active) or inactive_idx < len(inactive)
    ):
        if current_state == "active":
            next_time = float(inactive[inactive_idx]) if inactive_idx < len(inactive) else horizon
            intervals.append((current_time, min(next_time, horizon)))
            current_state = "inactive"
            current_time = next_time
            if inactive_idx < len(inactive):
                inactive_idx += 1
        else:
            next_time = float(active[active_idx]) if active_idx < len(active) else horizon
            current_state = "active"
            current_time = next_time
            if active_idx < len(active):
                active_idx += 1

    if current_state == "active" and current_time < horizon:
        intervals.append((current_time, horizon))

    # Filter out any degenerate intervals.
    return [(start, end) for start, end in intervals if end > start]


def split_intervals(
    intervals: Iterable[Tuple[float, float]], max_length: float
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for start, end in intervals:
        length = end - start
        if length <= 0.0:
            continue
        if length <= max_length:
            result.append((start, end))
            continue

        current = start
        while current < end:
            next_boundary = min(current + max_length, end)
            result.append((current, next_boundary))
            current = next_boundary
    return result


def apply_ewma(
    timestamps: Sequence[float],
    gaussian_values: Sequence[float],
    intervals: Sequence[Tuple[float, float]],
    half_life: float,
) -> List[float]:
    if half_life <= 0.0:
        raise ValueError("`ewma_half_life` must be positive.")

    times = list(timestamps)
    values = list(gaussian_values)
    result = values[:]
    tau = half_life / math.log(2.0)

    for start, end in intervals:
        start_idx = bisect_left(times, start)
        end_idx = bisect_left(times, end)
        if end_idx <= start_idx:
            continue

        segment_length = end_idx - start_idx
        if segment_length <= 0:
            continue

        prev = values[start_idx]
        result[start_idx] = prev
        for offset in range(1, segment_length):
            current_idx = start_idx + offset
            delta_t = times[current_idx] - times[current_idx - 1]
            alpha = 1.0 - math.exp(-delta_t / tau)
            prev = (1.0 - alpha) * prev + alpha * values[current_idx]
            result[current_idx] = prev

    return result


def smooth_client_rate(
    client: dict,
    window: float,
    sigma: float,
    half_life: float,
    max_interval: float,
) -> None:
    timestamps = list(client.get("timestamps-livelab", []))
    rates = list(client.get("rate", []))

    if not timestamps or not rates:
        return
    if len(timestamps) != len(rates):
        raise ValueError("Mismatched lengths for `timestamps-livelab` and `rate`.")

    active = client.get("active", [])
    inactive = client.get("inactive", [])
    active_intervals = build_active_intervals(active, inactive)

    if not active_intervals:
        # Leave the rate unchanged when there are no active spans.
        return

    gaussian_smoothed = gaussian_smooth(timestamps, rates, window=window, sigma=sigma)
    sub_intervals = split_intervals(active_intervals, max_interval)

    if not sub_intervals:
        final_rate = gaussian_smoothed
    else:
        final_rate = apply_ewma(timestamps, gaussian_smoothed, sub_intervals, half_life)

    if len(final_rate) != len(rates):
        raise RuntimeError("Final smoothed rate length does not match original length.")

    client["rate"] = [float(value) for value in final_rate]


def load_clients(path: Path) -> OrderedDict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, OrderedDict):
        raise TypeError(f"Expected OrderedDict in pickle, got {type(data).__name__}.")
    return data


def save_clients(path: Path, clients: OrderedDict) -> None:
    with path.open("wb") as handle:
        pickle.dump(clients, handle)


def iterate_clients(clients: OrderedDict):
    if tqdm is not None:
        return tqdm(clients.items(), total=len(clients), desc="Smoothing rates")
    return clients.items()


def main() -> None:
    args = parse_args()
    clients = load_clients(args.input)

    iterator = iterate_clients(clients)
    total = len(clients)

    for idx, (client_id, client_entry) in enumerate(iterator, start=1):
        smooth_client_rate(
            client_entry,
            window=args.window,
            sigma=args.sigma,
            half_life=args.ewma_half_life,
            max_interval=args.max_sub_interval,
        )
        if tqdm is None and idx % 100 == 0:
            print(f"Processed {idx}/{total} clients", flush=True)

    save_clients(args.input, clients)

    if tqdm is None:
        print(f"Completed smoothing for {total} clients.")


if __name__ == "__main__":
    main()
