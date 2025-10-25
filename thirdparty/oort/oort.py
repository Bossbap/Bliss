import logging
import math
from collections import OrderedDict
from random import Random

import numpy
import numpy as np2

from .utils.lp import *


def create_training_selector(args, sample_seed=None):
    seed = sample_seed if sample_seed is not None else getattr(args, "sample_seed", 233)
    return _training_selector(args, sample_seed=seed)

class _training_selector(object):
    """Oort's training selector with explicit explore/exploit split."""
    def __init__(self, args, sample_seed=233):
        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.alpha = args.exploration_alpha

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.args = args
        # Legacy percentile-based pacer params (kept for reference)
        # self.round_threshold = args.round_threshold
        # self.round_prefer_duration = float('inf')
        # self.last_util_record = 0
        # self._last_round_threshold = self.round_threshold

        # New T-budget pacer parameters (aligned with Bliss / Oort paper)
        self.t_budget = float(getattr(args, "t_budget", 0.0))
        self.pacer_step = int(getattr(args, "pacer_step", 20))
        self.pacer_delta = float(getattr(args, "pacer_delta", 0.0))
        self.round_prefer_duration = self.t_budget

        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None
        self._last_round_prefer_duration = None

        np2.random.seed(sample_seed)

    # ---------- registration / feedback ----------
    def register_client(self, client_id, feedbacks):
        if client_id not in self.totalArms:
            self.totalArms[client_id] = {
                'reward'    : float(feedbacks.get('reward', 0.0)),
                'duration'  : float(feedbacks.get('duration', 0.0)),
                'time_stamp': self.training_round,
                'count'     : 0,
                'status'    : True,
            }
            self.unexplored.add(client_id)

    def update_duration(self, client_id, duration):
        if client_id in self.totalArms:
            self.totalArms[client_id]['duration'] = float(duration)

    def update_client_util(self, client_id, feedbacks):
        self.totalArms[client_id]['reward']     = float(feedbacks['reward'])
        self.totalArms[client_id]['duration']   = float(feedbacks['duration'])
        self.totalArms[client_id]['time_stamp'] = float(feedbacks['time_stamp'])
        self.totalArms[client_id]['count']     += 1
        self.totalArms[client_id]['status']     = bool(feedbacks['status'])
        self.unexplored.discard(client_id)
        self.successfulClients.add(client_id)

    # ---------- pacer ----------
    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0.0
        for client in clientList:
            if client in self.successfulClients:
                cnt    += 1
                cntUtil += self.totalArms[client]['reward']
        return cntUtil / cnt

    def pacer(self):
        lastExplorationUtil  = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)
        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)
        self.successfulClients = set()

        # Legacy percentile-based pacer (kept for traceability):
        # if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:
        #     utilLast = sum(self.exploitUtilHistory[-2*self.args.pacer_step:-self.args.pacer_step])
        #     utilCurr = sum(self.exploitUtilHistory[-self.args.pacer_step:])
        #     baseline = max(utilLast, 1e-12)
        #     rel_change = abs(utilCurr - utilLast) / baseline
        #
        #     if rel_change <= 0.1:
        #         old = self.round_threshold
        #         self.round_threshold = min(100., self.round_threshold + self.args.pacer_delta)
        #         self.last_util_record = self.training_round - self.args.pacer_step
        #         logging.info(
        #             "[Oort.Pacer] Plateau at %d: last=%g, curr=%g, Δ/last=%0.2f%% → relax %0.1f%%→%0.1f%%",
        #             self.training_round, utilLast, utilCurr, 100*rel_change, old, self.round_threshold
        #         )
        #     elif rel_change >= 5.0:
        #         old = self.round_threshold
        #         self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
        #         self.last_util_record = self.training_round - self.args.pacer_step
        #         logging.info(
        #             "[Oort.Pacer] Surge at %d: last=%g, curr=%g, Δ/last=%0.2f%% → tighten %0.1f%%→%0.1f%%",
        #             self.training_round, utilLast, utilCurr, 100*rel_change, old, self.round_threshold
        #         )

        if (self.training_round >= 2 * self.pacer_step and
                self.training_round % self.pacer_step == 0):
            util_last = sum(self.exploitUtilHistory[-2 * self.pacer_step:-self.pacer_step])
            util_curr = sum(self.exploitUtilHistory[-self.pacer_step:])
            baseline = max(util_last, 1e-12)
            rel_change = abs(util_curr - util_last) / baseline

            if rel_change <= 0.1:
                self.t_budget += self.pacer_delta
            elif rel_change >= 5.0:
                self.t_budget = max(self.pacer_delta, self.t_budget - self.pacer_delta)

            # Expose the new preferred duration to downstream consumers.
            self.args.t_budget = self.t_budget
            self.round_prefer_duration = self.t_budget

    # ---------- top-K with explicit explore/exploit split ----------
    def select_participant(self, num_of_clients, feasible_clients=None):
        viable_clients = feasible_clients if feasible_clients is not None \
            else set([x for x in self.totalArms.keys() if self.totalArms[x]['status']])
        return self.getTopK(num_of_clients, self.training_round + 1, viable_clients)

    def _safe_norm(self, values, clip_bound=0.95, thres=1e-4):
        """Guarded normalization stats."""
        if not values:
            # (max, min, range, avg, clip_value)
            return 1.0, 0.0, 1.0, 0.0, 1.0
        _list = sorted(values)
        clip_value = _list[min(int(len(_list)*clip_bound), len(_list)-1)]
        _max = _list[-1]
        _min = _list[0]*0.999
        _range = max(_max - _min, thres)
        _avg = sum(_list)/max(1e-4, float(len(_list)))
        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        self.training_round = cur_time
        self.blacklist = self.get_blacklist()
        self.pacer()

        # decay exploration e_t
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        e = self.exploration

        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if int(x) in feasible_clients and int(x) not in self.blacklist]

        # Split into unexplored vs explored
        unexplored_keys = [k for k in orderedKeys if self.totalArms[k]['count'] == 0]
        explored_keys   = [k for k in orderedKeys if self.totalArms[k]['count'] >  0]
        # Trimmed verbose debug logging: candidate sizes / heads omitted

        # Preferred duration sourced from pacer T-budget
        self.round_prefer_duration = float(self.t_budget) if self.t_budget > 0 else float('inf')

        prev_pd = self._last_round_prefer_duration
        if (prev_pd is None) or (abs(self.round_prefer_duration - prev_pd) > 1e-9):
            logging.info(
                "[Oort.Pacer] round=%d: preferred_duration=%s (t_budget)%s",
                self.training_round,
                ("inf" if self.round_prefer_duration == float('inf') else f"{self.round_prefer_duration:.3f}s"),
                ("" if prev_pd is None else f", was {prev_pd:.3f}s" if prev_pd != float('inf') else ", was inf")
            )
            self._last_round_prefer_duration = self.round_prefer_duration

        # How many from each pool?
        want_explore = int(e * numOfSamples)
        want_exploit = numOfSamples - want_explore
        # RNG heads intentionally not logged

        # Reallocate if one pool is short
        take_explore = min(want_explore, len(unexplored_keys))
        take_exploit = min(want_exploit, len(explored_keys))
        deficit = numOfSamples - (take_explore + take_exploit)
        if deficit > 0:
            # Fill from the larger remaining pool
            extra_from_unexplored = min(deficit, max(0, len(unexplored_keys) - take_explore))
            take_explore += extra_from_unexplored
            deficit -= extra_from_unexplored
            if deficit > 0:
                extra_from_explored = min(deficit, max(0, len(explored_keys) - take_exploit))
                take_exploit += extra_from_explored
                deficit -= extra_from_explored

        picked_explore = []
        picked_exploit = []

        # -------- UNEXPLORED: uniform (no prior) --------
        if take_explore > 0 and len(unexplored_keys) > 0:
            picked_explore = list(np2.random.choice(unexplored_keys, size=take_explore, replace=False))

        # -------- EXPLORED: Oort scoring + stochastic top-K --------
        if take_exploit > 0 and len(explored_keys) > 0:
            # Build rewards & staleness arrays from explored only
            moving_reward = [self.totalArms[k]['reward'] for k in explored_keys]
            staleness     = [cur_time - self.totalArms[k]['time_stamp'] for k in explored_keys]
            max_r, min_r, rng_r, avg_r, clip_r = self._safe_norm(moving_reward, self.args.clip_bound)
            max_s, min_s, rng_s, avg_s, _      = self._safe_norm(staleness, thres=1)
            # (debug logging removed)

            scores = {}
            for k in explored_keys:
                creward = min(self.totalArms[k]['reward'], clip_r)
                sc = (creward - min_r) / float(rng_r)
                # temporal uncertainty (staleness)
                sc += math.sqrt(0.1 * math.log(max(cur_time, 1.0)) / max(1.0, self.totalArms[k]['time_stamp']))
                # pacer penalty if longer than preferred T
                clientDuration = self.totalArms[k]['duration']
                if clientDuration > self.round_prefer_duration:
                    sc *= ((float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.args.round_penalty)
                scores[k] = abs(sc)

            if scores and max(scores.values()) - min(scores.values()) > 1e-12:
                sorted_keys = sorted(scores, key=scores.get, reverse=True)
                cut_idx = min(take_exploit, len(sorted_keys) - 1)
                cutoff = scores[sorted_keys[cut_idx]] * self.args.cut_off_util
                temp_pool = []
                for cid in sorted_keys:
                    if scores[cid] < cutoff and len(temp_pool) > 10 * take_exploit:
                        break
                    temp_pool.append(cid)
                totalSc = max(1e-4, float(sum(scores[cid] for cid in temp_pool)))
                probs = [scores[cid] / totalSc for cid in temp_pool]
                picked_exploit = list(np2.random.choice(temp_pool, size=take_exploit, replace=False, p=probs))
            else:
                picked_exploit = list(np2.random.choice(explored_keys, size=take_exploit, replace=False))

        picked = picked_explore + picked_exploit
        # Ensure uniqueness and stable order (optional)
        picked = list(dict.fromkeys(picked))

        # Record pools for pacer window accounting (used at next call)
        self.exploreClients = list(picked_explore)
        self.exploitClients = list(picked_exploit)

        # Trimmed selection diagnostics

        return picked

    # ---------- misc (unchanged helpers) ----------
    def get_blacklist(self):
        blacklist = []
        if self.args.blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True,
                                       key=lambda k: self.totalArms[k]['count'])
            for client_id in sorted_client_ids:
                if self.totalArms[client_id]['count'] > self.args.blacklist_rounds:
                    blacklist.append(client_id)
                else:
                    break
            predefined_max_len = self.args.blacklist_max_len * len(self.totalArms)
            if len(blacklist) > predefined_max_len:
                logging.warning("Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]
        return set(blacklist)

    def get_median_reward(self):
        feasible_rewards = [self.totalArms[x]['reward'] for x in list(self.totalArms.keys()) if int(x) not in self.blacklist]
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards)/float(len(feasible_rewards))
        return 0

    def get_client_reward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        # kept for backward compatibility; we now use _safe_norm internally
        aList.sort()
        clip_value = aList[min(int(len(aList)*clip_bound), len(aList)-1)]
        _max = aList[-1]
        _min = aList[0]*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))
        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)

    def get_pacer_state(self):
        return {
            "algo": "oort",
            "training_round": int(self.training_round),
            "t_budget": (None if self.t_budget in (None, float('inf')) else float(self.t_budget)),
            "pacer_step": int(self.pacer_step),
            "pacer_delta": float(self.pacer_delta),
            "preferred_duration": (None if self.round_prefer_duration == float('inf') else float(self.round_prefer_duration))
        }

    def get_state(self):
        """Return a serialisable snapshot of the selector."""
        return {
            "totalArms": [(k, v) for k, v in self.totalArms.items()],
            "training_round": self.training_round,
            "exploration": self.exploration,
            "decay_factor": self.decay_factor,
            "exploration_min": self.exploration_min,
            "alpha": self.alpha,
            "t_budget": self.t_budget,
            "pacer_step": self.pacer_step,
            "pacer_delta": self.pacer_delta,
            "round_prefer_duration": self.round_prefer_duration,
            "sample_window": self.sample_window,
            "exploitUtilHistory": list(self.exploitUtilHistory),
            "exploreUtilHistory": list(self.exploreUtilHistory),
            "exploitClients": list(self.exploitClients),
            "exploreClients": list(self.exploreClients),
            "successfulClients": list(self.successfulClients),
            "unexplored": list(self.unexplored),
            "blacklist": list(self.blacklist) if self.blacklist is not None else None,
            "last_round_prefer_duration": self._last_round_prefer_duration,
            "rng_state": self.rng.getstate(),
            "np_random_state": np2.random.get_state(),
        }

    def load_state(self, state):
        """Restore selector fields from `get_state` output."""
        if not state:
            return

        self.totalArms = OrderedDict(state.get("totalArms", []))
        self.training_round = state.get("training_round", 0)
        self.exploration = state.get("exploration", self.exploration)
        self.decay_factor = state.get("decay_factor", self.decay_factor)
        self.exploration_min = state.get("exploration_min", self.exploration_min)
        self.alpha = state.get("alpha", self.alpha)
        self.t_budget = state.get("t_budget", self.t_budget)
        self.pacer_step = state.get("pacer_step", self.pacer_step)
        self.pacer_delta = state.get("pacer_delta", self.pacer_delta)
        self.round_prefer_duration = state.get("round_prefer_duration", self.round_prefer_duration)
        self.sample_window = state.get("sample_window", self.sample_window)
        self.exploitUtilHistory = list(state.get("exploitUtilHistory", []))
        self.exploreUtilHistory = list(state.get("exploreUtilHistory", []))
        self.exploitClients = list(state.get("exploitClients", []))
        self.exploreClients = list(state.get("exploreClients", []))
        self.successfulClients = set(state.get("successfulClients", []))
        self.unexplored = set(state.get("unexplored", []))
        blacklist = state.get("blacklist", None)
        self.blacklist = set(blacklist) if blacklist is not None else None
        self._last_round_prefer_duration = state.get("last_round_prefer_duration", self._last_round_prefer_duration)

        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)
        np_state = state.get("np_random_state")
        if np_state is not None:
            np2.random.set_state(np_state)
        # ensure downstream components (aggregator/client_manager) see updated pacer budget
        setattr(self.args, "t_budget", self.t_budget)
    
def create_pyramid_training_selector(args, sample_seed=None):
    seed = sample_seed if sample_seed is not None else getattr(args, "sample_seed", 233)
    return _pyramid_training_selector(args, sample_seed=seed)


class _pyramid_training_selector(_training_selector):
    """
    PyramidFL selector:
    - Selection: same explore/exploit split and scoring as Oort
    - Maintains per-client gradient size and previous timings
    - Produces per-round per-client overrides: dropout_p, local_steps
    """
    def __init__(self, args, sample_seed=233):
        super().__init__(args, sample_seed)
        # per-round overrides
        self._overrides = {}  # cid -> {"dropout_p": float, "local_steps": int}

    # ---------- registration / feedback ----------
    def register_client(self, client_id, feedbacks):
        super().register_client(client_id, feedbacks)
        # Seed extra fields
        self.totalArms[client_id].update({
            'gsize': float(feedbacks.get('gsize', 0.0) or 0.0),
            't_comp': feedbacks.get('t_comp', None),
            't_total': feedbacks.get('t_total', None),
            # last observed iterations (fall back to base local_steps)
            'steps': int(getattr(self.args, 'local_steps', 1)),
        })

    def update_client_util(self, client_id, feedbacks):
        super().update_client_util(client_id, feedbacks)
        arm = self.totalArms[client_id]
        if 'gsize' in feedbacks and feedbacks['gsize'] is not None:
            arm['gsize'] = float(feedbacks['gsize'])
        if 't_comp' in feedbacks and feedbacks['t_comp'] is not None:
            arm['t_comp'] = float(feedbacks['t_comp'])
        if 't_total' in feedbacks and feedbacks['t_total'] is not None:
            arm['t_total'] = float(feedbacks['t_total'])
        # track actual iterations run last time if reported
        if 'steps' in feedbacks and feedbacks['steps'] is not None:
            try:
                arm['steps'] = int(feedbacks['steps'])
            except Exception:
                pass

    # ---------- selection + overrides ----------
    def select_participant(self, num_of_clients, feasible_clients=None):
        picked = self.getTopK(num_of_clients, self.training_round + 1,
                            feasible_clients if feasible_clients is not None else None)
        # Build overrides for JUST selected clients
        self._overrides = {}
        # Determine exploited (seen) vs explored (unseen) split as Oort did
        e = self.exploration  # Oort uses this as exploration fraction
        want_explore = int(e * num_of_clients)
        want_exploit = num_of_clients - want_explore

        # Preserve the internal ordering used by Oort's getTopK:
        # We recompute the explored/unexplored partition over 'picked'
        picked_seen = [cid for cid in picked if self.totalArms[cid]['count'] > 0]
        picked_new  = [cid for cid in picked if self.totalArms[cid]['count'] == 0]

        # Truncate to requested counts (if needed)
        picked_exploit = picked_seen[:want_exploit]
        # Fill deficit from seen if explore shortfall happens — consistent with Oort’s fill logic
        if len(picked_seen) < want_exploit:
            deficit = want_exploit - len(picked_seen)
        else:
            deficit = 0

        picked_explore = picked_new[:(want_explore + deficit)]

        # --- Rank exploited by G_i (descending) ---
        rankables = [cid for cid in picked_exploit
                    if self.totalArms[cid].get('gsize', 0.0) is not None]
        rankables.sort(key=lambda cid: self.totalArms[cid].get('gsize', 0.0), reverse=True)

        L = max(1, len(rankables))
        a = float(self.args.dropout_a)
        b = float(self.args.dropout_b)
        beta = float(self.args.confidence_beta)
        I_fix = int(self.args.local_steps)
        T = self.round_prefer_duration if self.round_prefer_duration != float('inf') else float('inf')

        # Defaults: explore => no dropout, base iterations
        for cid in picked_explore:
            self._overrides[cid] = {"dropout_p": 0.0, "local_steps": I_fix}

        # Exploit clients: map rank -> P_i, compute I_i using previous timings
        for rank, cid in enumerate(rankables, start=1):
            P_i = a if L == 1 else a + (rank - 1) * (b - a) / float(L - 1)
            arm = self.totalArms[cid]
            t_comp_prev = arm.get('t_comp', None)
            t_total_prev = arm.get('t_total', None)
            steps_prev = int(arm.get('steps', I_fix) or I_fix)

            # If missing stats, fall back to no dropout / base steps
            if t_comp_prev is None or t_total_prev is None or t_comp_prev <= 0.0:
                self._overrides[cid] = {"dropout_p": 0.0, "local_steps": I_fix}
                continue

            # Split comm/compute and estimate per-step compute time
            t_comm_prev = max(0.0, t_total_prev - t_comp_prev)
            t_prime = t_comp_prev + (1.0 - P_i) * t_comm_prev
            # Estimate the time it takes to run I_fix steps on this client
            # using the last-round observed per-step time (t_comp_prev / steps_prev).
            # This mirrors the intended formula that scales relative to I_fix.
            per_step = max(t_comp_prev / max(1, steps_prev), 1e-8)
            t_comp_equiv_fix = per_step * float(I_fix)

            if not (T < float('inf')):  # no pacer yet -> base steps
                I_i = I_fix
            else:
                extra = beta * max(T - t_prime, 0.0) / max(t_comp_equiv_fix, 1e-8)
                I_i = int(max(I_fix, round((1.0 + extra) * I_fix)))

            self._overrides[cid] = {"dropout_p": float(P_i), "local_steps": int(I_i)}

        # For any remaining seen client not in rankables (edge cases), use defaults
        for cid in picked_exploit:
            if cid not in self._overrides:
                self._overrides[cid] = {"dropout_p": 0.0, "local_steps": I_fix}

        return picked

    def get_overrides(self):
        return self._overrides

    def get_state(self):
        state = super().get_state()
        state.update({
            "overrides": self._overrides,
        })
        return state

    def load_state(self, state):
        super().load_state(state)
        self._overrides = state.get("overrides", {})


def create_testing_selector(data_distribution=None, client_info=None, model_size=None):
    return _testing_selector(data_distribution, client_info, model_size)

class _testing_selector:
    """Oort's testing selector
    We provide two kinds of selector:
    select_by_deviation: testing participant selection that preserves data representativeness.
    select_by_category: testing participant selection that enforce developer's requirement on
        distribution of the testing set. Note that this selector is avaliable only if the client
        info is provided.
    Attributes:
        client_info: Optional; A dictionary that stores client id to client profile(system speech and
            network bandwidth) mapping. For example, {1: [153.0, 2209.61]} indicates that client 1
            needs 153ms to run a single sample inference and their network bandwidth is 2209 Kbps.
        model_size: Optional; the size of the model(i.e., the data transfer size) in kb
        data_distribution: Optional; individual data characteristics(distribution).
    """
    def __init__(self, data_distribution=None, client_info=None, model_size=None):
        """Inits testing selector."""
        self.client_info = client_info
        self.model_size = model_size
        self.data_distribution = data_distribution
        if self.client_info:
            self.client_idx_list = list(range(len(client_info)))
    def update_client_info(self, client_ids, client_profile):
        """Update clients' profile(system speed and network bandwidth)
        Since the clients' info is dynamic, developers can use this function
        to update clients' profile. If the client id does not exist, Oort will
        create a new entry for this client.
        Args:
            client_ids: A list of client ids whose profile needs to be updated
            client_info: Updated information about client profile, formatted as
                a list of pairs(speed, bw)
        Raises:
            Raises an error if len(client_ids) != len(client_info)
        """
        return 0
    def _hoeffding_bound(self, dev_tolerance, capacity_range, total_num_clients, confidence=0.8):
        """Use hoeffding bound to cap the deviation from E[X]
        Args:
            dev_tolerance: maximum deviation from the empirical (E[X])
            capacity_range: the global max-min range of number of samples across all clients
            total_num_clients: total number of feasible clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence
        Returns:
            The estimated number of participant needed to satisfy developer's requirement
        """
        factor = (1.0 - 2*total_num_clients/math.log(1-math.pow(confidence, 1)) \
                                    * (dev_tolerance/float(capacity_range)) ** 2)
        n = (total_num_clients+1.0)/factor
        return n
    def select_by_deviation(self, dev_target, range_of_capacity, total_num_clients,
            confidence=0.8, overcommit=1.1):
        """Testing selector that preserves data representativeness.
        Given the developer-specified tolerance `dev_target`, Oort can estimate the number
        of participants needed such that the deviation from the representative categorical
        distribution is bounded.
        Args:
            dev_target: developer-specified tolerance
            range_of_capacity: the global max-min range of number of samples across all clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence
            overcommit: Optional; to handle stragglers
        Returns:
            A list of selected participants
        """
        num_of_selected = self._hoeffding_bound(dev_target, range_of_capacity, total_num_clients, confidence=0.8)
        return num_of_selected
    def select_by_category(self, request_list, max_num_clients=None, greedy_heuristic=True):
        """Testing selection based on requested number of samples per category.
        When individual data characteristics(distribution) is provided, Oort can
        enforce client's request on the number of samples per category.
        Args:
            request_list: a list that specifies the desired number of samples per category.
                i.e., [num_requested_samples_class_x for class_x in request_list].
            max_num_clients: Optional; the maximum number of participants .
            greedy_heuristic: Optional; whether to use Oort-based solver. Otherwise, Mix-Integer Linear Programming
        Returns:
            A list of selected participants ids.
        Raises:
            Raises an error if 1) no client information is provided or 2) the requirement
            cannot be satisfied(e.g., max_num_clients too small).
        """
        client_sample_matrix, test_duration, lp_duration = run_select_by_category(request_list, self.data_distribution,
            self.client_info, max_num_clients, self.model_size, greedy_heuristic)
        return client_sample_matrix, test_duration, lp_duration
