import logging
import math
from collections import OrderedDict
from random import Random

import numpy
import numpy as np2

from .utils.lp import *


def create_training_selector(args):
    return _training_selector(args)

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
        self.round_threshold = args.round_threshold
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0

        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None

        self._last_round_threshold = self.round_threshold
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

        if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:
            utilLast = sum(self.exploitUtilHistory[-2*self.args.pacer_step:-self.args.pacer_step])
            utilCurr = sum(self.exploitUtilHistory[-self.args.pacer_step:])
            baseline = max(utilLast, 1e-12)
            rel_change = abs(utilCurr - utilLast) / baseline

            if rel_change <= 0.1:
                old = self.round_threshold
                self.round_threshold = min(100., self.round_threshold + self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.info(
                    "[Oort.Pacer] Plateau at %d: last=%g, curr=%g, Δ/last=%0.2f%% → relax %0.1f%%→%0.1f%%",
                    self.training_round, utilLast, utilCurr, 100*rel_change, old, self.round_threshold
                )
            elif rel_change >= 5.0:
                old = self.round_threshold
                self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.info(
                    "[Oort.Pacer] Surge at %d: last=%g, curr=%g, Δ/last=%0.2f%% → tighten %0.1f%%→%0.1f%%",
                    self.training_round, utilLast, utilCurr, 100*rel_change, old, self.round_threshold
                )

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

        # Preferred duration percentile for pacer penalty
        if self.round_threshold < 100.:
            # We compute over known durations (including zeros). If you prefer to
            # ignore zeros, filter them here.
            sortedDuration = sorted([self.totalArms[k]['duration'] for k in client_list])
            idx = min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)
            self.round_prefer_duration = sortedDuration[idx] if sortedDuration else float('inf')
        else:
            self.round_prefer_duration = float('inf')

        # Log pacer line (visible in your plots)
        prev_pd = self._last_round_prefer_duration
        if (prev_pd is None) or (abs(self.round_prefer_duration - prev_pd) > 1e-9):
            logging.info(
                "[Oort.Pacer] round=%d: preferred_duration=%s (threshold=%0.1f%%)%s",
                self.training_round,
                ("inf" if self.round_prefer_duration == float('inf') else f"{self.round_prefer_duration:.3f}s"),
                self.round_threshold,
                ("" if prev_pd is None else f", was {prev_pd:.3f}s" if prev_pd != float('inf') else ", was inf")
            )
            self._last_round_prefer_duration = self.round_prefer_duration

        # How many from each pool?
        want_explore = int(e * numOfSamples)
        want_exploit = numOfSamples - want_explore

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

            # If all explored scores are identical/zero, sample uniformly
            if scores and max(scores.values()) - min(scores.values()) > 1e-12:
                sorted_keys = sorted(scores, key=scores.get, reverse=True)
                # probabilistic sampling from an augmented candidate set,
                # like upstream Oort (respecting stochasticity)
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
            "round_threshold": float(self.round_threshold),
            "pacer_step": int(self.args.pacer_step),
            "pacer_delta": float(self.args.pacer_delta),
            "preferred_duration": (None if self.round_prefer_duration == float('inf') else float(self.round_prefer_duration))
        }
    
def create_pyramid_training_selector(args):
    return _pyramid_training_selector(args)


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

            # If missing stats, fall back to no dropout / base steps
            if t_comp_prev is None or t_total_prev is None or t_comp_prev <= 0.0:
                self._overrides[cid] = {"dropout_p": 0.0, "local_steps": I_fix}
                continue

            t_comm_prev = max(0.0, t_total_prev - t_comp_prev)
            t_prime = t_comp_prev + (1.0 - P_i) * t_comm_prev

            if not (T < float('inf')):  # no pacer yet -> base steps
                I_i = I_fix
            else:
                extra = beta * max(T - t_prime, 0.0) / max(t_comp_prev / float(I_fix), 1e-8)
                I_i = int(max(I_fix, round((1.0 + extra) * I_fix)))

            self._overrides[cid] = {"dropout_p": float(P_i), "local_steps": int(I_i)}

        # For any remaining seen client not in rankables (edge cases), use defaults
        for cid in picked_exploit:
            if cid not in self._overrides:
                self._overrides[cid] = {"dropout_p": 0.0, "local_steps": I_fix}

        return picked

    def get_overrides(self):
        return self._overrides
