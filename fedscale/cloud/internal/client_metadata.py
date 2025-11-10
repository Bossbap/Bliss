import numpy as np
from typing import Callable, Optional
from fedscale.cloud.fllibs import *


class ClientMetadata:
    """
    Contains the server-side metadata for a single client,
    including static device capacities and dynamic time-varying traces.
    """

    def __init__(
        self,
        host_id: int,
        client_id: int,
        size: int,
        # Static compute capacities
        cpu_flops: float,
        gpu_flops: float,
        # Dynamic traces: livelab (network) 
        timestamps_livelab: list[int],
        rate: list[int],
        # Dynamic traces: carat (compute/availability)
        timestamps_carat: list[int],
        availability: list[int],
        batteryLevel: list[int],
        # activity traces
        active: list[int],
        inactive: list[int],
        peak_throughput,
        rng_seed: Optional[int] = None,
    ):
        """
        :param host_id:   ID of the executor handling this client
        :param client_id: Global client ID
        :param cpu_flops:       Static compute capacity (e.g. FLOPS baseline)
        :param gpu_flops:       Static GPU capacity (if used)
        :param timestamps_livelab: sorted timestamps (s) for network-rate changes
        :param rate:      upload/download rate trace (Mb/s) at each timestamp
        :param timestamps_carat:    sorted timestamps (s) for compute changes
        :param availability: CPU/GPU availability percentage (0–100) per timestamp
        :param batteryLevel:  battery percentage (0–100) per timestamp
        """
        # Identity
        self.host_id = host_id
        self.client_id = client_id
        self.size = size

        # Static capacities
        self.cpu_flops = cpu_flops
        self.gpu_flops = gpu_flops

        # Network traces (livelab)
        self.timestamps_livelab = np.asarray(timestamps_livelab, dtype=np.float32)
        self.rate = np.asarray(rate, dtype=np.float32)
        self.peak_throughput = peak_throughput

        # Compute traces (carat)
        self.timestamps_carat = np.asarray(timestamps_carat, dtype=np.float32)
        self.availability = np.asarray(availability, dtype=np.float32)
        self.batteryLevel = np.asarray(batteryLevel, dtype=np.float32)

        # activity intervals
        self.active = active
        self.inactive = inactive

        # For adaptive sampling (e.g. Oort)
        self.score = 0

        # Noise to perturb FLOPS in one round
        self._round_noise = 1.0
        self._rng = np.random.default_rng(rng_seed)

        # most-recent end-to-end latency (sec)
        self.last_duration = None
        # cache of the latest simulated download/compute/upload breakdown
        self._last_time_breakdown: Optional[dict] = None

    def get_score(self):
        return self.score

    def register_reward(self, reward: float):
        """Update the sampling score for this client."""
        self.score = reward

    def is_active(self, cur_time):
        """
        Determines whether the client is active at the given simulation time.

        Args:
            cur_time (int or float): Current simulation time in seconds.

        Returns:
            bool: True if client is active, False otherwise.
        """
        T = 48 * 3600
        t = cur_time % T

        # Merge the two sorted lists, tag each timestamp with the phase it *starts*
        boundaries = sorted(
            [(ts, 'a') for ts in self.active] +
            [(ts, 'i') for ts in self.inactive]
        )

        # Initial phase
        phase = 'a' if (self.active and self.active[0] == 0) else 'i'

        # Walk through boundaries and flip the phase whenever we pass one
        for ts, _ in boundaries[1:]:  # skip the initial 0 entry
            if t < ts:
                break
            phase = 'i' if phase == 'a' else 'a'

        return phase == 'a'

    def _lookup(self, timestamps: list[int], values: list[float], t: float) -> float:
        """Return the trace value at sim-time t, assuming timestamps are sorted."""
        norm_t = t % (48*3600)
        if norm_t < timestamps[0]:
            return values[0]
        idx = max(i for i, ts in enumerate(timestamps) if ts <= norm_t)
        return values[idx]


    def bandwidth(self, t):
        rate = self._lookup(self.timestamps_livelab, self.rate, t)
        return self.peak_throughput * rate/54  # Mb/s

    def compute_speed(self, t: float) -> float:
        """
        Calculate effective FLOPS/s at time t, factoring in availability,
        battery level, and log-normal noise.
        """
        # 1) Base peak FLOPS
        base_peak = self.cpu_flops + self.gpu_flops  # CPU_FLOPS + GPU_FLOPS

        # 2) Availability fraction (0–1)
        avail_pct = self._lookup(self.timestamps_carat, self.availability, t) / 100.0

        # 3) Battery reduction factor
        batt = self._lookup(self.timestamps_carat, self.batteryLevel, t)
        if batt >= 70:
            batt_factor = 1.0
        elif batt >= 50:
            batt_factor = 0.9
        elif batt >= 30:
            batt_factor = 0.8
        elif batt >= 10:
            batt_factor = 0.6
        else:
            batt_factor = 0.4

        # 5) Final effective FLOPS/s
        return base_peak * avail_pct * batt_factor * self._round_noise


    def _simulate_data_phase(
        self,
        start_time: float,
        total_work: float,
        timestamps: list[int],
        rate_fn: Callable[[float], float],
        window: float,
        scale: float,
    ) -> float:
        """
        Generic simulator for download/upload or compute:
        - Loops over each interval where the rate_fn is constant,
        - Subtracts work done until total_work <= 0, then returns the exact finish time.

        Args:
            start_time: absolute sim time when phase begins.
            total_work: total MB (or total FLOPs) to complete.
            timestamps: breakpoints (in [0, window]) for when rate_fn may change.
            rate_fn: function t->rate (MB/s or FLOPS).
            window: cycle length (48h).
            scale: multiplier on rate_fn (e.g. 1.0).

        Returns:
            float: sim time when work_remaining hits zero.
        """
        # sort the cycle breakpoints
        pts = timestamps
        # normalize into window
        t0 = start_time % window
        abs_cycle_start = start_time - t0

        # find next index in pts after t0
        idx = next((i for i, x in enumerate(pts) if x > t0), len(pts))

        curr_time = start_time
        work_rem = total_work

        while True:
            # determine end of this sub-interval
            if idx < len(pts):
                next_point = abs_cycle_start + pts[idx]
            else:
                # wrap-around to end of window
                next_point = abs_cycle_start + window

            dt = next_point - curr_time
            rate = rate_fn(curr_time) * scale
            if rate <= 0:
                raise RuntimeError(f"Zero rate at t={curr_time}")

            potential = rate * dt
            if potential >= work_rem:
                # finishes within this interval
                return curr_time + (work_rem / rate)

            # subtract what we can do in this slice
            work_rem -= potential
            # advance time
            curr_time = next_point

            # if we wrapped around, shift the cycle
            if idx >= len(pts):
                abs_cycle_start += window
                t0 = 0
                idx = 0
            else:
                idx += 1

    def get_completion_time(self, cur_time, batch_size, local_steps, model_size, model_amount_parameters,
                        augmentation_factor: float = 3.0, reduction_factor: float = 0.5, clock_factor: float = 1.0) -> float:
        # Wrap the unified kernel with dropout_p=0
        _, t_total = self.get_times_with_dropout(
            cur_time=cur_time,
            batch_size=batch_size,
            local_steps=local_steps,
            model_size=model_size,
            model_amount_parameters=model_amount_parameters,
            reduction_factor=reduction_factor,
            dropout_p=0.0,
            augmentation_factor=augmentation_factor,
            clock_factor=clock_factor,
        )
        return t_total


    def get_times_with_dropout(
        self,
        cur_time: float,
        batch_size: int,
        local_steps: int,
        model_size: int,
        model_amount_parameters: int,
        reduction_factor: float = 0.5,
        dropout_p: float = 0.0,
        augmentation_factor: float = 3.0,
        clock_factor: float = 1.0,
    ) -> tuple[float, float]:
        """
        Simulate download -> compute -> upload with sparsification (dropout_p).
        Returns (t_comp, t_total), both in seconds.
        """
        WINDOW = 48 * 3600  # 48 h

        self.sample_round_noise()

        # DOWNLOAD (unchanged by dropout)
        download_end = self._simulate_data_phase(
            start_time=cur_time,
            total_work=model_size * clock_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0,
        )

        # COMPUTE
        total_ops = augmentation_factor * model_amount_parameters * batch_size * local_steps
        compute_end = self._simulate_data_phase(
            start_time=download_end,
            total_work=total_ops * clock_factor,
            timestamps=self.timestamps_carat,
            rate_fn=self.compute_speed,
            window=WINDOW,
            scale=1.0,
        )

        # UPLOAD (reduced payload by (1 - dropout_p))
        effective_model_size = max(0.0, (1.0 - float(dropout_p))) * model_size
        upload_end = self._simulate_data_phase(
            start_time=compute_end,
            total_work=effective_model_size / reduction_factor * clock_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0,
        )

        t_dl  = download_end - cur_time
        t_comp = compute_end - download_end
        t_ul  = upload_end - compute_end
        t_total = t_dl + t_comp + t_ul

        # Cache latest breakdown (used by server-side logging)
        self.last_duration = float(t_total)
        self._last_time_breakdown = {
            "t_dl": float(t_dl),
            "t_comp": float(t_comp),
            "t_ul": float(t_ul),
            "t_total": float(t_total),
            "local_steps": int(local_steps),
            "dropout_frac": float(dropout_p),
        }

        return float(t_comp), float(t_total)

    def get_last_time_breakdown(self) -> Optional[dict]:
        """Return a copy of the latest simulated time breakdown (or None if unavailable)."""
        if self._last_time_breakdown is None:
            return None
        return dict(self._last_time_breakdown)

    def get_download_time(self, cur_time: float, model_size_mb: float,
                          clock_factor: float = 1.0) -> float:
        """Return ONLY the simulated download latency (sec)."""
        WINDOW = 48 * 3600  # 48 h
        finish = self._simulate_data_phase(
            start_time=cur_time,
            total_work=model_size_mb * clock_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0
        )
        return finish - cur_time

    def get_upload_time(self, start_time: float,
                        model_size_mb: float,
                        reduction_factor: float = 0.5,
                        clock_factor: float = 1.0) -> float:
        """Return ONLY the simulated upload latency (sec)."""
        WINDOW = 48 * 3600
        finish = self._simulate_data_phase(
            start_time=start_time,
            total_work=model_size_mb / reduction_factor * clock_factor,
            timestamps=self.timestamps_livelab,
            rate_fn=self.bandwidth,
            window=WINDOW,
            scale=1.0
        )
        return finish - start_time
    
    def sample_round_noise(self, target_mean: float = 0.9, sigma: float = 0.25):
        """One log-normal noise multiplier per round (σ from original code)."""
        # Noise
        mu = np.log(target_mean) - (sigma**2)/2
        self._round_noise = float(self._rng.lognormal(mean=mu, sigma=sigma))

    # ------------------------------------------------------------------
    #  Serialisation helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return a serialisable snapshot of this metadata record."""
        return {
            "host_id": self.host_id,
            "client_id": self.client_id,
            "size": self.size,
            "cpu_flops": float(self.cpu_flops),
            "gpu_flops": float(self.gpu_flops),
            "timestamps_livelab": self.timestamps_livelab.tolist(),
            "rate": self.rate.tolist(),
            "timestamps_carat": self.timestamps_carat.tolist(),
            "availability": self.availability.tolist(),
            "batteryLevel": self.batteryLevel.tolist(),
            "active": list(self.active),
            "inactive": list(self.inactive),
            "peak_throughput": self.peak_throughput,
            "score": self.score,
            "last_duration": self.last_duration,
            "_round_noise": self._round_noise,
            "rng_state": self._rng.bit_generator.state,
            "_last_time_breakdown": dict(self._last_time_breakdown) if self._last_time_breakdown is not None else None,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ClientMetadata":
        """Instantiate ClientMetadata from a state dict."""
        obj = cls(
            host_id=state["host_id"],
            client_id=state["client_id"],
            size=state["size"],
            cpu_flops=state["cpu_flops"],
            gpu_flops=state["gpu_flops"],
            timestamps_livelab=state["timestamps_livelab"],
            rate=state["rate"],
            timestamps_carat=state["timestamps_carat"],
            availability=state["availability"],
            batteryLevel=state["batteryLevel"],
            active=state["active"],
            inactive=state["inactive"],
            peak_throughput=state["peak_throughput"],
            rng_seed=None,
        )
        obj.score = state.get("score", obj.score)
        obj.last_duration = state.get("last_duration")
        obj._round_noise = state.get("_round_noise", obj._round_noise)
        obj._last_time_breakdown = state.get("_last_time_breakdown")

        rng_state = state.get("rng_state")
        if rng_state is not None:
            obj._rng = np.random.default_rng()
            obj._rng.bit_generator.state = rng_state
        return obj
