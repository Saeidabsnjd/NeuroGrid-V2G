import numpy as np

class SmartGridEnv:
    """
    A simulated microgrid environment for Multi-Agent Reinforcement Learning (MARL).
    Simulates base load, solar generation, and a fleet of EVs.
    Time steps: 24 steps representing 1-hour intervals in a day.
    """
    def __init__(self, num_evs=10, p_base=0.15, congestion_factor=0.01, degradation_cost=0.015, departure_penalty_coef=15.0):
        self.num_evs = num_evs
        self.p_base = p_base
        self.congestion_factor = congestion_factor
        self.degradation_cost = degradation_cost
        self.departure_penalty_coef = departure_penalty_coef
        
        self.num_steps = 24  # 24 hours
        self.dt = 1.0        # 1 hour per step
        
        # 1. Base Load Profile (kWh) - typical double peak residential curve
        # Peaks around 8 AM (index 8) and 7 PM (index 19)
        self.base_load_profile = np.array([
            15, 12, 10, 10, 12, 18, 25, 35, 40, 38, 35, 32, 
            30, 32, 35, 38, 42, 48, 55, 52, 45, 35, 25, 18
        ], dtype=float) * (num_evs / 10.0)  # Scale load with number of EVs
        
        # 2. Solar Generation Profile (kWh) - bell curve centered at noon (index 12)
        self.solar_profile = np.array([
            0, 0, 0, 0, 0, 1, 5, 12, 20, 28, 35, 38, 
            40, 38, 35, 28, 20, 12, 5, 1, 0, 0, 0, 0
        ], dtype=float) * (num_evs / 10.0) * 1.2 # Scale solar with number of EVs
        
        # 3. EV Parameter Configurations
        self.battery_capacity = 50.0  # kWh
        self.charge_rate = 7.0        # kW (Level 2 charging)
        self.discharge_rate = 7.0     # kW (V2G discharge capability)
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        
        # Commute schedules (Arrival time, Departure time, Target SOC)
        # Standard commute: Arrive home in the evening, leave in the morning
        self.ev_profiles = []
        np.random.seed(42) # Keep seed for consistent environment layout
        for i in range(num_evs):
            # Arrival between 16:00 and 20:00 (indices 16 to 20)
            arrival = int(np.random.randint(16, 21))
            # Departure between 07:00 and 09:00 (indices 7 to 9)
            departure = int(np.random.randint(7, 10))
            # Initial SOC on arrival
            init_soc = float(np.random.uniform(0.15, 0.35))
            # Target SOC at departure
            target_soc = 0.9
            
            self.ev_profiles.append({
                'arrival': arrival,
                'departure': departure,
                'init_soc': init_soc,
                'target_soc': target_soc
            })
            
        self.reset()

    def reset(self):
        self.current_step = 0
        
        # Current states of each EV battery
        self.soc = np.array([p['init_soc'] for p in self.ev_profiles], dtype=float)
        
        # Track history for reporting/graphs
        self.net_load_history = []
        self.price_history = []
        self.ev_loads_history = []  # Step-by-step load of each EV
        self.ev_soc_history = []    # Step-by-step SOC of each EV
        
        return self._get_states()

    def is_plugged_in(self, ev_idx, step):
        """
        Helper to check if EV is plugged in at a given time step.
        Commutes are overnight, meaning plugged in from arrival to midnight,
        and midnight to departure.
        """
        arr = self.ev_profiles[ev_idx]['arrival']
        dep = self.ev_profiles[ev_idx]['departure']
        
        if arr > dep:
            # Over-night connection (e.g. arrive at 18:00, leave at 08:00)
            return step >= arr or step < dep
        else:
            # Day-time connection (e.g. arrive at 08:00, leave at 17:00)
            return arr <= step < dep

    def _get_states(self):
        """
        Generate states for all active agents.
        State vector: [t/24, SOC_t, P_t_est, time_to_departure/24, SOC_target]
        """
        states = []
        for i in range(self.num_evs):
            # Normalised time
            t_norm = self.current_step / float(self.num_steps)
            soc_val = self.soc[i]
            
            # Estimated price (based on base load + solar alone for this step)
            net_base = self.base_load_profile[self.current_step] - self.solar_profile[self.current_step]
            est_price = self.p_base + self.congestion_factor * net_base
            est_price_norm = np.clip(est_price / 0.5, 0, 1) # Normalise relative to 0.5 $/kWh
            
            # Time to departure
            dep = self.ev_profiles[i]['departure']
            if self.is_plugged_in(i, self.current_step):
                if self.current_step >= self.ev_profiles[i]['arrival']:
                    # Before midnight
                    steps_left = (24 - self.current_step) + dep
                else:
                    # After midnight
                    steps_left = dep - self.current_step
            else:
                steps_left = 0
            steps_left_norm = steps_left / 24.0
            
            target_soc = self.ev_profiles[i]['target_soc']
            
            state = np.array([
                t_norm,
                soc_val,
                est_price_norm,
                steps_left_norm,
                target_soc
            ], dtype=float)
            
            states.append(state)
            
        return states

    def step(self, actions):
        """
        Executes one step in the environment.
        actions: list of size num_evs, each is 0 (Charge), 1 (Discharge), 2 (Idle)
        """
        step = self.current_step
        ev_powers = np.zeros(self.num_evs)
        degradation_costs = np.zeros(self.num_evs)
        
        # 1. Update EV states and calculate power loads
        for i in range(self.num_evs):
            if not self.is_plugged_in(i, step):
                # If not plugged in, EV is idle and cannot exchange power
                ev_powers[i] = 0.0
                continue
                
            action = actions[i]
            
            if action == 0:  # Charge
                # Max energy we can add based on rate, efficiency, and battery capacity limits
                max_charge_kwh = self.charge_rate * self.dt
                possible_charge_soc = max_charge_kwh * self.charge_eff / self.battery_capacity
                
                actual_charge_soc = min(possible_charge_soc, 1.0 - self.soc[i])
                self.soc[i] += actual_charge_soc
                
                # Grid load created by charging
                ev_powers[i] = (actual_charge_soc * self.battery_capacity) / self.charge_eff
                degradation_costs[i] = self.degradation_cost * ev_powers[i]
                
            elif action == 1:  # Discharge (V2G)
                # Max energy we can extract based on rate, efficiency, and safety SOC limit (e.g. 0.15)
                max_discharge_kwh = self.discharge_rate * self.dt
                possible_discharge_soc = max_discharge_kwh / (self.battery_capacity * self.discharge_eff)
                
                # Prevent discharging below 15% state of charge
                min_soc = 0.15
                actual_discharge_soc = min(possible_discharge_soc, max(0.0, self.soc[i] - min_soc))
                self.soc[i] -= actual_discharge_soc
                
                # Grid power feed-in (negative load)
                ev_powers[i] = - (actual_discharge_soc * self.battery_capacity) * self.discharge_eff
                degradation_costs[i] = self.degradation_cost * abs(ev_powers[i])
                
            else:  # Idle
                ev_powers[i] = 0.0
                degradation_costs[i] = 0.0

        # 2. Dynamic Pricing Engine
        base_demand = self.base_load_profile[step]
        solar_gen = self.solar_profile[step]
        ev_total_demand = np.sum(ev_powers)
        
        net_load = base_demand - solar_gen + ev_total_demand
        
        # Real-time price equation
        price = self.p_base + self.congestion_factor * net_load
        price = max(0.02, price)  # Floor price at 2 cents/kWh to prevent negative infinity loops
        
        # 3. Calculate rewards
        rewards = np.zeros(self.num_evs)
        for i in range(self.num_evs):
            if not self.is_plugged_in(i, step):
                rewards[i] = 0.0
                continue
                
            # Financial component
            power_flow = ev_powers[i]
            if power_flow > 0:  # Charging cost
                financial_reward = - price * power_flow
            elif power_flow < 0:  # Discharging revenue (with V2G feed-in tariff support)
                financial_reward = price * abs(power_flow) * 0.95
            else:
                financial_reward = 0.0
                
            # Wear cost
            wear_penalty = degradation_costs[i]
            
            # Departure Penalty (applied at the step before departure)
            dep_penalty = 0.0
            next_step = (step + 1) % 24
            if next_step == self.ev_profiles[i]['departure']:
                target = self.ev_profiles[i]['target_soc']
                if self.soc[i] < target:
                    dep_penalty = self.departure_penalty_coef * ((target - self.soc[i]) ** 2)
            
            rewards[i] = financial_reward - wear_penalty - dep_penalty

        # Save history
        self.net_load_history.append(net_load)
        self.price_history.append(price)
        self.ev_loads_history.append(ev_powers.copy())
        self.ev_soc_history.append(self.soc.copy())
        
        # Advance time
        self.current_step += 1
        done = self.current_step >= self.num_steps
        
        next_states = self._get_states() if not done else [np.zeros(5) for _ in range(self.num_evs)]
        dones = [done] * self.num_evs
        
        info = {
            'net_load': net_load,
            'price': price,
            'ev_total_demand': ev_total_demand,
            'base_demand': base_demand,
            'solar_gen': solar_gen
        }
        
        return next_states, rewards, dones, info
