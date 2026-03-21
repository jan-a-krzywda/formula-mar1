import numpy as np
import random

# --- Team action space: one discrete per car (cars ordered by id within team) ---
# Racing: 0–2 pit S/M/H; 3–5 stay Harvest / Boost / Standard
# Pre-race (pending_starting_tyres): 0–2 = starting tyre Soft / Medium / Hard (same logits, masked)
ACT_PIT_SOFT = 0
ACT_PIT_MEDIUM = 1
ACT_PIT_HARD = 2
ACT_STAY_HRV = 3
ACT_STAY_BOOST = 4
ACT_STAY_STD = 5
NUM_CAR_ACTIONS = 6
NUM_CARS = 22  # 11 teams × 2 drivers
TIMETOWER_CLIP_SEC = 60.0  # gap to leader: clip to ±60 s, then /60 → [-1, 1]; leader is 0
# lap_fraction + 2 cars × (compound, tyre_age, battery, override, gap_ahead, gap_behind)
# + full timetower (NUM_CARS gaps to leader) + pending_starting_tyres
TEAM_OBS_DIM = 1 + 2 * 6 + NUM_CARS + 1


def encode_pace_pit_to_action(pace: int, pit_cmd: int) -> int:
    """Map legacy (pace, pit) with pit in {0,7,8,9} to a single 0..5 action."""
    if pit_cmd >= 7:
        return pit_cmd - 7  # 7→soft, 8→med, 9→hard
    if pace == 0:
        return ACT_STAY_HRV
    if pace == 2:
        return ACT_STAY_BOOST
    return ACT_STAY_STD


class F1TeamEnv:
    def __init__(self, total_laps=60):
        self.driver_map = {
            "BlueCow":     [("VER", "Max Versplatton"), ("LAW", "Liam Awesome")],
            "Merciless":   [("RUS", "Forge Hustle"),    ("ANT", "Kimi Macaroni")],
            "Furrari":     [("LEC", "Chuck LeClutch"),  ("HAM", "Louis Hamstring")],
            "McPapaya":    [("NOR", "Lando Chuckris"),  ("PIA", "Osco Pastry")],
            "Astonishing": [("ALO", "Nando Alfonso"),   ("STR", "Lance Scroll")],
            "Alpain":      [("GAS", "Peter Ghastly"),   ("COL", "Franky Colapunch")],
            "Billiams":    [("ALB", "Alex Album"),      ("SAI", "Carlos Signs")],
            "ToroLoco":    [("HAD", "Isaac Badger"),    ("LIN", "Artie Lindblad")],
            "Sober":       [("HUL", "Nico Bulkensmear"),("BOR", "Gabe Tortellini")],
            "Hassle":      [("OCO", "Esteban Acorn"),   ("BEA", "Ollie Birdman")],
            "CaddyShack":  [("PER", "Surge Perez"),     ("BOT", "Battery Voltas")]
        }
        self.teams = list(self.driver_map.keys())
        self.total_laps = total_laps
        self.reset() 

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_lap = 0
        all_drivers = []
        for team_name in self.teams:
            for driver_idx in [0, 1]:
                drv_code, drv_name = self.driver_map[team_name][driver_idx]
                all_drivers.append({"id": drv_code, "full_name": drv_name, "team": team_name})
                
        random.shuffle(all_drivers)
        
        self.cars = []
        for grid_pos, driver_data in enumerate(all_drivers):
            driver_data.update({
                "total_race_time": grid_pos * 1.5, 
                "tyre_compound": 2,                
                "tyre_age": 0.0,
                "tyre_wear": 0.0,
                "battery": 1.0,
                "pit_stops": 0,
                "compounds_used": {2},  # placeholder until pending_starting_tyres step (default medium in obs)
                "last_lap_time": 0.0, 
                "status": "GRID",
                "current_pace_cmd": 1,
                "override_unlocked": False
            })
            self.cars.append(driver_data)

        self.benchmark_pit_wiggles = {}  # per-team random wiggle for benchmark pit laps; cleared each reset
        self.pending_starting_tyres = True

        return self.get_observations()

    def get_observations(self):
        obs = {}
        lap_fraction = self.current_lap / self.total_laps
        
        # Sort cars by race time (leader first)
        sorted_cars = sorted(self.cars, key=lambda x: x["total_race_time"])
        
        for team in self.teams:
            team_obs = [lap_fraction]
            
            team_cars = [car for car in self.cars if car["team"] == team]
            team_cars.sort(key=lambda x: x["id"])
            
            for car in team_cars:
                pos = sorted_cars.index(car)
                n = len(sorted_cars)
                if pos == 0:
                    gap_ahead = 0.0
                else:
                    gap_ahead = car["total_race_time"] - sorted_cars[pos - 1]["total_race_time"]
                if pos >= n - 1:
                    gap_behind = 0.0
                else:
                    gap_behind = sorted_cars[pos + 1]["total_race_time"] - car["total_race_time"]
                gap_ahead_n = np.clip(gap_ahead, -30.0, 30.0) / 30.0
                gap_behind_n = np.clip(gap_behind, -30.0, 30.0) / 30.0

                team_obs.extend([
                    car["tyre_compound"] / 3.0,
                    car["tyre_age"] / 50.0,
                    car["battery"],
                    float(car["override_unlocked"]),
                    gap_ahead_n,
                    gap_behind_n,
                ])

            leader_time = sorted_cars[0]["total_race_time"]
            for c in sorted_cars:
                gap_to_leader = c["total_race_time"] - leader_time
                team_obs.append(
                    float(np.clip(gap_to_leader, -TIMETOWER_CLIP_SEC, TIMETOWER_CLIP_SEC) / TIMETOWER_CLIP_SEC)
                )

            team_obs.append(1.0 if self.pending_starting_tyres else 0.0)
            obs[team] = np.array(team_obs, dtype=np.float32)

        return obs

    def _process_car_lap(self, car, action_index):
        """Apply one lap for `car` given discrete action 0..5 (see ACT_* constants)."""
        base_lap_time = 85.0
        pit_loss_time = 25.0
        lab_time_std = 0.2

        a = int(action_index)
        if a <= ACT_PIT_HARD:
            pit_cmd = a + 1  # 1=S, 2=M, 3=H
            pace_cmd = 1     # benchmark uses STD when pitting
        elif a == ACT_STAY_HRV:
            pit_cmd, pace_cmd = 0, 0
        elif a == ACT_STAY_BOOST:
            pit_cmd, pace_cmd = 0, 2
        else:
            pit_cmd, pace_cmd = 0, 1

        car["status"] = "OUT"
        car["current_pace_cmd"] = pace_cmd
        
        current_pit_loss = 0.0 
        if pit_cmd > 0:
            current_pit_loss = pit_loss_time 
            car["tyre_compound"] = pit_cmd 
            car["compounds_used"].add(pit_cmd)
            car["tyre_age"] = 0
            car["tyre_wear"] = 0.0
            car["pit_stops"] += 1
            car["status"] = "PIT"
            
        pace_modifier = 0.0
        status_mod = 1.0  # HRV=0.9, BST=1.1, STD=1, OVR=1.2

        if pace_cmd == 0:
            pace_modifier = 0.6
            car["battery"] = min(1.0, car["battery"] + 0.5)
            status_mod = 0.9
            if car["status"] != "PIT": car["status"] = "HRV"
        elif pace_cmd == 2:
            if car["override_unlocked"] and car["battery"] > 0.2:
                pace_modifier = -1.2
                car["battery"] -= 0.25
                status_mod = 1.2
                if car["status"] != "PIT": car["status"] = "OVR"
            elif not car["override_unlocked"] and car["battery"] > 0.15:
                pace_modifier = -0.6
                car["battery"] -= 0.15
                status_mod = 1.1
                if car["status"] != "PIT": car["status"] = "BST"
            else:
                # Boost requested but not enough charge — same lap as STD
                status_mod = 1.0
                if car["status"] != "PIT": car["status"] = "STD"
        else:
            status_mod = 1.0
            if car["status"] != "PIT": car["status"] = "STD"

        # Tyre model: tyre_wear accumulates; penalty = tyre_wear**3
        tyre_pace_deltas = {1: -1.2, 2: 0.0, 3: 1.2}
        tyre_deg_rates = {1: 0.15, 2: 0.08, 3: 0.05}
        compound = car["tyre_compound"]
        car["tyre_wear"] += tyre_deg_rates[compound] * status_mod
        deg_penalty = car["tyre_wear"] ** 3

        lap_noise = np.random.normal(0.0, lab_time_std)
        lap_time = base_lap_time + tyre_pace_deltas[compound] + deg_penalty + pace_modifier + lap_noise + current_pit_loss

        car["last_lap_time"] = lap_time
        car["total_race_time"] += lap_time
        car["tyre_age"] += 1

    def _resolve_overtakes(self, grid_order):
        for i in range(1, len(grid_order)):
            attacker = grid_order[i]
            defender = grid_order[i-1]
            
            if attacker["total_race_time"] < defender["total_race_time"]:
                overtake_chance = 0.4
                if attacker["status"] == "OVR": overtake_chance += 0.4
                elif attacker["status"] == "BST": overtake_chance += 0.3
                if defender["status"] in ["OVR", "BST"]: overtake_chance -= 0.2 
                    
                tyre_delta = defender["tyre_age"] - attacker["tyre_age"]
                overtake_chance += (tyre_delta * 0.05)
                overtake_chance = max(0.05, min(0.95, overtake_chance))
                
                if random.random() < overtake_chance:
                    attacker["total_race_time"] += 0.2
                    defender["total_race_time"] += 0.2
                    attacker["last_lap_time"] += 0.2
                    defender["last_lap_time"] += 0.2
                else:
                    time_lost = (defender["total_race_time"] + 0.1) - attacker["total_race_time"]
                    attacker["total_race_time"] += time_lost
                    attacker["last_lap_time"] += time_lost

    def step(self, actions):
        """Returns observation, reward, done, and info dictionary for RL training."""

        if self.pending_starting_tyres:
            for car in self.cars:
                team_action = np.asarray(actions[car["team"]], dtype=np.int32).reshape(-1)
                team_cars = sorted([c for c in self.cars if c["team"] == car["team"]], key=lambda x: x["id"])
                slot = 0 if car["id"] == team_cars[0]["id"] else 1
                idx = int(team_action[slot])
                idx = max(0, min(2, idx))
                compound = idx + 1  # 1=S, 2=M, 3=H
                car["tyre_compound"] = compound
                car["compounds_used"] = {compound}
                car["tyre_age"] = 0.0
                car["tyre_wear"] = 0.0
            self.pending_starting_tyres = False
            observations = self.get_observations()
            rewards = {team: 0.0 for team in self.teams}
            dones = {team: False for team in self.teams}
            infos = {team: {} for team in self.teams}
            return observations, rewards, dones, infos

        starting_order = {car["id"]: i for i, car in enumerate(self.cars)}
        starting_compound_counts = {car["id"]: len(car["compounds_used"]) for car in self.cars}
        self.current_lap += 1
        grid_order = list(self.cars) 
        
        
        for i, car in enumerate(grid_order):
            if i == 0:
                car["override_unlocked"] = False
            else:
                gap_to_car_ahead = car["total_race_time"] - grid_order[i-1]["total_race_time"]
                car["override_unlocked"] = (gap_to_car_ahead < 1.0)

        for car in self.cars:
            team_action = np.asarray(actions[car["team"]], dtype=np.int32).reshape(-1)
            team_cars = sorted([c for c in self.cars if c["team"] == car["team"]], key=lambda x: x["id"])
            car_slot = 0 if car["id"] == team_cars[0]["id"] else 1
            self._process_car_lap(car, team_action[car_slot])

        self._resolve_overtakes(grid_order)
        self.cars.sort(key=lambda x: x["total_race_time"])

        rewards = {team: 0.0 for team in self.teams}

        # Pace reward: lap time minus most pit loss (dense signal)
        pace_times = []
        for c in self.cars:
            pit_loss = 25.0 if c["status"] == "PIT" else 0.0
            pace_times.append(c["last_lap_time"] - 0.95 * pit_loss)
        for i, car in enumerate(self.cars):
            rewards[car["team"]] += (87 - pace_times[i]) / 500.0

        # Position delta among non-pit cars only (ignore pit reordering)
        cars_by_id = {c["id"]: c for c in self.cars}
        non_pit_ids = [c["id"] for c in self.cars if c["status"] != "PIT"]
        prev_order_non_pit = sorted(non_pit_ids, key=lambda cid: starting_order[cid])
        current_order_non_pit = list(non_pit_ids)

        for car_id in non_pit_ids:
            team = cars_by_id[car_id]["team"]
            prev_pos = prev_order_non_pit.index(car_id)
            current_pos = current_order_non_pit.index(car_id)

            if current_pos < prev_pos:
                num_spots_gained = prev_pos - current_pos
                rewards[team] += num_spots_gained * 0.01
            elif current_pos > prev_pos:
                num_spots_lost = current_pos - prev_pos
                rewards[team] -= num_spots_lost * 0.01

        progress = self.current_lap / max(1, self.total_laps)
        for car in self.cars:
            team = car["team"]
            used_cnt = len(car["compounds_used"])
            if used_cnt < 2:
                rewards[team] -= 0.03 * (progress ** 2)
            if starting_compound_counts.get(car["id"], 1) < 2 and used_cnt >= 2:
                rewards[team] += 0.5

        done = self.current_lap >= self.total_laps
        dones = {team: done for team in self.teams}
        
        if done:
            for team in self.teams:
                team_cars = [c for c in self.cars if c["team"] == team]
                if any(len(c["compounds_used"]) < 2 for c in team_cars):
                    rewards[team] -= 10.0
                else:
                    pos_weights = np.linspace(1, -1, 22)
                    position_bonus = sum(pos_weights[self.cars.index(car)] for car in team_cars)
                    rewards[team] += position_bonus

        observations = self.get_observations()
        infos = {team: {} for team in self.teams} 
        
        return observations, rewards, dones, infos


# Benchmark strategies: 1-stop M(20)->H(40), 2-stop M(25)->M(25)->S(10); ±2 lap random wiggle per pit
BENCHMARK_1STOP_PIT_LAP = 21
BENCHMARK_2STOP_PIT_LAPS = (26, 51)
BENCHMARK_2STOP_COMPOUNDS = (8, 7)
BENCHMARK_WIGGLE = 1  # pit lap = base ± random in [0, WIGGLE] (1-2 laps wiggle room)

def get_benchmark_action(env, team, strategy="1stop", fixed_laps=False, random_energy=False):
    """Fixed 1-stop (M→H) or 2-stop (M→M→S) benchmark; optional ±1 lap wiggle on pit timing.

    By default, stay-out laps use **standard** pace only (MODE column stays STD). Set
    ``random_energy=True`` to sample Harvest / Boost / Standard on those laps (for viz / stress tests).
    """
    if fixed_laps:
        pit_laps = (BENCHMARK_1STOP_PIT_LAP,) if strategy == "1stop" else BENCHMARK_2STOP_PIT_LAPS
    else:
        if not hasattr(env, "benchmark_pit_wiggles"):
            env.benchmark_pit_wiggles = {}
        if team not in env.benchmark_pit_wiggles:
            if strategy == "1stop":
                w = random.randint(-BENCHMARK_WIGGLE, BENCHMARK_WIGGLE)
                env.benchmark_pit_wiggles[team] = (BENCHMARK_1STOP_PIT_LAP + w,)
            else:
                w1 = random.randint(-BENCHMARK_WIGGLE, BENCHMARK_WIGGLE)
                w2 = random.randint(-BENCHMARK_WIGGLE, BENCHMARK_WIGGLE)
                env.benchmark_pit_wiggles[team] = (
                    BENCHMARK_2STOP_PIT_LAPS[0] + w1,
                    BENCHMARK_2STOP_PIT_LAPS[1] + w2,
                )
        pit_laps = env.benchmark_pit_wiggles[team]

    team_cars = sorted([c for c in env.cars if c["team"] == team], key=lambda c: c["id"])
    next_lap = env.current_lap + 1
    actions = []
    for car in team_cars:
        pace = 1
        pit_cmd = 0
        pits_done = car["pit_stops"]
        if strategy == "1stop":
            if pits_done == 0 and next_lap >= pit_laps[0]:
                pit_cmd = 9  # Hard
        else:
            if pits_done < len(pit_laps) and next_lap >= pit_laps[pits_done]:
                pit_cmd = BENCHMARK_2STOP_COMPOUNDS[pits_done]
        a = encode_pace_pit_to_action(pace, pit_cmd)
        if random_energy and a >= ACT_STAY_HRV:
            a = random.randint(ACT_STAY_HRV, ACT_STAY_STD)
        actions.append(a)
    return np.array(actions, dtype=np.int32)