import numpy as np
import random

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
                "compounds_used": {2},  # start on medium; track for two-compound rule
                "last_lap_time": 0.0, 
                "status": "GRID",
                "current_pace_cmd": 1,
                "override_unlocked": False
            })
            self.cars.append(driver_data)

        self.benchmark_pit_wiggles = {}  # per-team random wiggle for benchmark pit laps; cleared each reset
            
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
            # Reference for time tower: car number 1 (first in team order by id)
            ref_time = team_cars[0]["total_race_time"]
            # Time tower: gap from ref car; normalized ±120s -> [-1, 1]. Negative = ahead, positive = behind.
            time_tower = [np.clip((c["total_race_time"] - ref_time) / 120.0, -1.0, 1.0) for c in sorted_cars]
            
            for car in team_cars:
                # Find this specific car's position in the global standings
                pos = sorted_cars.index(car)
                gap_ahead = 0.0 if pos == 0 else car["total_race_time"] - sorted_cars[pos-1]["total_race_time"]
                
                has_pitted = 1.0 if car["pit_stops"] > 0 else 0.0
                
                team_obs.extend([
                    car["tyre_compound"] / 3.0,     
                    car["tyre_age"] / 50.0,         
                    car["battery"],                 
                    float(car["override_unlocked"]),
                    car["last_lap_time"],
                    has_pitted,
                    pos / 21.0 ])
                
            # Append the 22-car time tower (gaps relative to car 1, normalized ±120s -> [-1, 1])
            team_obs.extend(time_tower)
            
            # The final vector is now size 37
            obs[team] = np.array(team_obs, dtype=np.float32)
            
        return obs

    def _process_car_lap(self, car, team_action):

        # Consistent mapping of actions
        base_lap_time = 85.0 
        pit_loss_time = 25.0 
        lab_time_std = 0.05
        
        team_cars = sorted([c for c in self.cars if c["team"] == car["team"]], key=lambda x: x["id"])
        is_car_2 = (car["id"] == team_cars[1]["id"])

        # --- THE ACTION MAPPING FIX ---
        pace_cmd = team_action[2] if is_car_2 else team_action[0]
        raw_pit_cmd = team_action[3] if is_car_2 else team_action[1]
        
        # Map the 10 network outputs back to 4 physical actions
        pit_cmd_map = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:2, 9:3}
        pit_cmd = pit_cmd_map.get(raw_pit_cmd, 0)

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
            pace_modifier = 1.5
            car["battery"] = min(1.0, car["battery"] + 0.25)
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
            status_mod = 1.0
            if car["status"] != "PIT": car["status"] = "STD"

        # Tyre model: tyre_wear accumulates; penalty = tyre_wear**3
        tyre_pace_deltas = {1: -1.2, 2: 0.0, 3: 1.0}
        tyre_deg_rates = {1: 0.12, 2: 0.06, 3: 0.03}
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
                if attacker["status"] == "OVR": overtake_chance += 0.5
                elif attacker["status"] == "BST": overtake_chance += 0.5
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
            team_action = actions[car["team"]]
            self._process_car_lap(car, team_action)

        self._resolve_overtakes(grid_order)
        self.cars.sort(key=lambda x: x["total_race_time"])
        
       # ==========================================
        # RL REWARD CALCULATION
        # ==========================================
        rewards = {team: 0.0 for team in self.teams}
        
        # 1. Calculate 'Pace' Lap Time (excluding pit lane time)
        pace_times = []
        for c in self.cars:
            # We remove the 25s pit penalty from the DENSE reward calculation
            # so the AI sees its "pure driving speed"
            pit_loss = 25.0 if c["status"] == "PIT" else 0.0
            pace_times.append(c["last_lap_time"] - pit_loss)
            
        # 2. Dense Reward (Pure Pace + Strategic Cost)
        for i, car in enumerate(self.cars):
            team = car["team"]
            
            # Reward for being faster than an 86.0s benchmark
            time_delta = 87 - pace_times[i]
            #if time_delta < 0:
            rewards[team] += time_delta/500.
        # 2b. Position-change reward ONLY for on-track overtakes
        #     Ignore places gained/lost that are purely due to pit stops this lap.
        #     We do this by comparing ordering among only the cars that did NOT pit.
        cars_by_id = {c["id"]: c for c in self.cars}
        non_pit_ids = [c["id"] for c in self.cars if c["status"] != "PIT"]
        # Previous order restricted to the cars that are not in the pits this lap
        prev_order_non_pit = sorted(non_pit_ids, key=lambda cid: starting_order[cid])
        # Current order among the same set (self.cars is already sorted by race time)
        current_order_non_pit = [c_id for c_id in non_pit_ids]

        for car_id in non_pit_ids:
            team = cars_by_id[car_id]["team"]
            prev_pos = prev_order_non_pit.index(car_id)
            current_pos = current_order_non_pit.index(car_id)

            if current_pos < prev_pos:
                num_spots_gained = prev_pos - current_pos
                rewards[team] += (num_spots_gained * 0.001)
            elif current_pos > prev_pos:
                num_spots_lost = current_pos - prev_pos
                rewards[team] -= (num_spots_lost * 0.001)

        # 2c. Rule-shaping: encourage satisfying the 2-compound rule BEFORE race end
        # - Small per-lap penalty if a car hasn't used 2 compounds yet (grows later in race)
        # - One-time bonus when a car reaches 2 compounds (usually via a pit to a new compound)
        progress = self.current_lap / max(1, self.total_laps)  # in (0, 1]
        for car in self.cars:
            team = car["team"]
            used_cnt = len(car["compounds_used"])
            if used_cnt < 2:
                rewards[team] -= 0.03 * (progress ** 2)
            if starting_compound_counts.get(car["id"], 1) < 2 and used_cnt >= 2:
                rewards[team] += 0.5
    
                    
        # 3. Terminal Reward (position-based + two-compound rule)
        done = self.current_lap >= self.total_laps
        dones = {team: done for team in self.teams}
        
        if done:
            # Final order (cars already sorted by total_race_time)
            for team in self.teams:
                team_cars = [c for c in self.cars if c["team"] == team]
                if any(len(c["compounds_used"]) < 2 for c in team_cars):
                    rewards[team] -= 10.0  # DQ: each car must use two different compounds
                else:
                    # Position bonus: 22 cars → 22 slots (pos 1..22 → index 0..21)
                    position_bonus = 0.0
                    pos_weights = np.logspace(0, 1, 22)  # length 22, not 21
                    for car in team_cars:
                        pos = self.cars.index(car) + 1  # 1-based position
                        position_bonus += pos_weights[pos - 1]
                    rewards[team] += position_bonus

        observations = self.get_observations()
        infos = {team: {} for team in self.teams} 
        
        return observations, rewards, dones, infos


# Benchmark strategies: 1-stop M(20)->H(40), 2-stop M(25)->M(25)->S(10); ±2 lap random wiggle per pit
BENCHMARK_1STOP_PIT_LAP = 21
BENCHMARK_2STOP_PIT_LAPS = (26, 51)
BENCHMARK_2STOP_COMPOUNDS = (8, 7)
BENCHMARK_WIGGLE = 2  # pit lap = base ± random in [0, WIGGLE] (1-2 laps wiggle room)

def get_benchmark_action(env, team, strategy="1stop", fixed_laps=False):
    """
    Baseline agent: pace = 1 (standard).
    - fixed_laps=False: pit laps have random wiggle per episode (1-stop lap 21±2, 2-stop 26±2, 51±2).
    - fixed_laps=True: deterministic pit laps (1-stop lap 21, 2-stop laps 26 and 51). Use for BC.
    Returns [pace1, pit1, pace2, pit2] with pit = 0 or 7/8/9 (S/M/H).
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
    out = []
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
        out.append(pace)
        out.append(pit_cmd)
    return np.array(out, dtype=np.int32)


def get_equal_pitstop_action(env, team, n_pit_stops):
    """
    Legacy baseline: pace = 1, pit at equally spaced laps (2 or 3 stops).
    Prefer get_benchmark_action for fixed M->H or M->M->S strategies.
    """
    total_laps = env.total_laps
    pit_laps = [round((k + 1) * total_laps / (n_pit_stops + 1)) for k in range(n_pit_stops)]
    team_cars = sorted([c for c in env.cars if c["team"] == team], key=lambda c: c["id"])
    next_lap = env.current_lap + 1
    out = []
    for car in team_cars:
        pace = 1
        pit_cmd = 0
        pits_done = car["pit_stops"]
        if pits_done < len(pit_laps) and next_lap >= pit_laps[pits_done]:
            used = car["compounds_used"]
            if 1 not in used:
                pit_cmd = 7
            elif 3 not in used:
                pit_cmd = 9
            else:
                pit_cmd = 8
        out.append(pace)
        out.append(pit_cmd)
    return np.array(out, dtype=np.int32)