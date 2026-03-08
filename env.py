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
                "battery": 1.0,
                "pit_stops": 0,
                "last_lap_time": 0.0, 
                "status": "GRID",
                "current_pace_cmd": 1,
                "override_unlocked": False
            })
            self.cars.append(driver_data)
            
        return self.get_observations()

    def get_observations(self):
        obs = {}
        lap_fraction = self.current_lap / self.total_laps
        
        # --- NEW: THE GLOBAL TRACK MAP ---
        # Sort cars by race time to figure out everyone's actual position
        sorted_cars = sorted(self.cars, key=lambda x: x["total_race_time"])
        leader_time = sorted_cars[0]["total_race_time"] if len(sorted_cars) > 0 else 0.0
        
        # Create an array of 22 numbers showing how far behind the leader EVERY car is.
        # We divide by 120.0 seconds to keep the neural network inputs normalized between 0.0 and 1.0.
        global_gaps = [min((c["total_race_time"] - leader_time) / 120.0, 1.0) for c in sorted_cars]
        
        for team in self.teams:
            team_obs = [lap_fraction]
            
            team_cars = [car for car in self.cars if car["team"] == team]
            team_cars.sort(key=lambda x: x["id"])
            
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
                    pos / 21.0  # NEW: The car now explicitly knows its own grid position!
                ])
                
            # Append the 22-car global traffic map to the end of the team's observation
            team_obs.extend(global_gaps)
            
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
            car["tyre_age"] = 0
            car["pit_stops"] += 1
            car["status"] = "PIT"
            
        # ... [Keep your existing Pace Modifiers (HRV, OVR, STD) and Tyre Math here] ...
        pace_modifier = 0.0
        deg_modifier = 1.0
        
        if pace_cmd == 0:   
            pace_modifier = 1.5   
            car["battery"] = min(1.0, car["battery"] + 0.25)
            deg_modifier = 0.5
            if car["status"] != "PIT": car["status"] = "HRV"
        elif pace_cmd == 2: 
            if car["override_unlocked"] and car["battery"] > 0.2:
                pace_modifier = -1.2  
                car["battery"] -= 0.25 
                deg_modifier = 1.5    
                if car["status"] != "PIT": car["status"] = "OVR" 
            elif not car["override_unlocked"] and car["battery"] > 0.15:
                pace_modifier = -0.6
                car["battery"] -= 0.15 
                deg_modifier = 1.2
                if car["status"] != "PIT": car["status"] = "BST"
        else:
            if car["status"] != "PIT": car["status"] = "STD"
            
        tyre_pace_deltas = {1: -1.2, 2: 0.0, 3: 1.0} 
        tyre_deg_rates = {1: 0.2, 2: 0.1, 3: 0.05}  
        compound = car["tyre_compound"]
        
        deg_penalty = (car["tyre_age"] * tyre_deg_rates[compound]) ** 3 
        lap_noise = np.random.normal(0.0, lab_time_std)
        
        lap_time = base_lap_time + tyre_pace_deltas[compound] + deg_penalty + pace_modifier + lap_noise + current_pit_loss
        
        car["last_lap_time"] = lap_time
        car["total_race_time"] += lap_time
        car["tyre_age"] += (1.0 * deg_modifier)

    def _resolve_overtakes(self, grid_order):
        for i in range(1, len(grid_order)):
            attacker = grid_order[i]
            defender = grid_order[i-1]
            
            if attacker["total_race_time"] < defender["total_race_time"]:
                overtake_chance = 0.3
                if attacker["status"] == "OVR": overtake_chance += 0.2
                elif attacker["status"] == "BST": overtake_chance += 0.1
                if defender["status"] in ["OVR", "BST"]: overtake_chance -= 0.1 
                    
                tyre_delta = defender["tyre_age"] - attacker["tyre_age"]
                overtake_chance += (tyre_delta * 0.05)
                overtake_chance = max(0.05, min(0.95, overtake_chance))
                
                if random.random() < overtake_chance:
                    attacker["total_race_time"] += 0.5
                    defender["total_race_time"] += 0.5
                    attacker["last_lap_time"] += 0.5
                    defender["last_lap_time"] += 0.5
                else:
                    time_lost = (defender["total_race_time"] + 0.3) - attacker["total_race_time"]
                    attacker["total_race_time"] += time_lost
                    attacker["last_lap_time"] += time_lost

    def step(self, actions):
        """Returns observation, reward, done, and info dictionary for RL training."""
        
        starting_order = {car["id"]: i for i, car in enumerate(self.cars)}
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
            pace_times.append(c["last_lap_time"])
            
        # 2. Dense Reward (Pure Pace + Strategic Cost)
        for i, car in enumerate(self.cars):
            team = car["team"]
            
            # Reward for being faster than an 86.0s benchmark
            time_delta = 90.0 - pace_times[i]
            if time_delta < 0:
                rewards[team] += time_delta/250.
        for i, car in enumerate(self.cars):
            team = car["team"]
            prev_pos = starting_order[car["id"]]
            current_pos = i
            
            # If current_pos < prev_pos, they gained a spot (overtook)
            if current_pos < prev_pos:
                num_spots_gained = prev_pos - current_pos
                # Give a small dense reward per car overtaken
                # 0.1 is usually a good weight to encourage aggression without 
                # making the AI dive-bomb recklessly.
                rewards[team] += (num_spots_gained * 0.1)
            elif current_pos > prev_pos:
                # Optional: Penalty for being overtaken
                num_spots_lost = current_pos - prev_pos
                rewards[team] -= (num_spots_lost * 0.05)
    
                    
        # 3. Terminal Reward (Constructor Points & Disqualification)
        done = self.current_lap >= self.total_laps
        dones = {team: done for team in self.teams}
        
        if done:
            f1_points = np.linspace(0,1, 22)
            for position, car in enumerate(self.cars):
                team = car["team"]
                
                if car["pit_stops"] == 0:
                    rewards[team] -= 1.0 # DQ still stays as a massive deterrent
                else:
                    rewards[team] += f1_points[position] 

        observations = self.get_observations()
        infos = {team: {} for team in self.teams} 
        
        return observations, rewards, dones, infos