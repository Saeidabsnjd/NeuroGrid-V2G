import numpy as np
from scipy.optimize import minimize

class CentralizedOptimizer:
    """
    Computes the centralized global Social Optimum (SO) for the smart grid.
    Minimizes total electricity cost + battery degradation + departure penalties
    across all EVs, subject to battery SOC constraints.
    """
    def __init__(self, env):
        self.env = env
        self.num_evs = env.num_evs
        self.num_steps = env.num_steps
        self.dt = env.dt
        
        # Build plug-in schedules for indexing
        # For each EV, we store the list of hour indices it is plugged in
        self.connected_hours = []
        self.var_offsets = [0] # Starting index of variables for each EV in the flat x vector
        
        for i in range(self.num_evs):
            hours = []
            for t in range(self.num_steps):
                if env.is_plugged_in(i, t):
                    hours.append(t)
            self.connected_hours.append(hours)
            self.var_offsets.append(self.var_offsets[-1] + 2 * len(hours)) # 2 vars per connected hour: P_charge, P_discharge
            
        self.total_vars = self.var_offsets[-1]

    def _get_ev_schedules(self, x):
        """
        Parses flat optimization vector x into charge/discharge powers for all hours.
        Returns:
            charge_powers: shape (num_steps, num_evs)
            discharge_powers: shape (num_steps, num_evs)
        """
        charge_powers = np.zeros((self.num_steps, self.num_evs))
        discharge_powers = np.zeros((self.num_steps, self.num_evs))
        
        for i in range(self.num_evs):
            offset = self.var_offsets[i]
            hours = self.connected_hours[i]
            for idx, t in enumerate(hours):
                charge_powers[t, i] = x[offset + 2 * idx]
                discharge_powers[t, i] = x[offset + 2 * idx + 1]
                
        return charge_powers, discharge_powers

    def _get_soc_trajectories(self, charge_powers, discharge_powers):
        """Computes the SOC trajectories of all EVs over the 24 hour period."""
        socs = np.zeros((self.num_steps + 1, self.num_evs))
        # Set initial SOC
        for i in range(self.num_evs):
            socs[0, i] = self.env.ev_profiles[i]['init_soc']
            
        # Standard overnight simulation stepping
        for t in range(self.num_steps):
            for i in range(self.num_evs):
                if self.env.is_plugged_in(i, t):
                    c_p = charge_powers[t, i]
                    d_p = discharge_powers[t, i]
                    
                    # Update SOC
                    c_soc = (c_p * self.env.charge_eff * self.dt) / self.env.battery_capacity
                    d_soc = (d_p * self.dt) / (self.env.battery_capacity * self.env.discharge_eff)
                    
                    socs[t+1, i] = socs[t, i] + c_soc - d_soc
                else:
                    socs[t+1, i] = socs[t, i] # Outside plug-in time, SOC remains constant
                    
        return socs

    def objective_function(self, x):
        charge_powers, discharge_powers = self._get_ev_schedules(x)
        socs = self._get_soc_trajectories(charge_powers, discharge_powers)
        
        total_cost = 0.0
        
        # 1. Energy Costs step by step
        for t in range(self.num_steps):
            base_load = self.env.base_load_profile[t]
            solar_gen = self.env.solar_profile[t]
            
            ev_load = 0.0
            for i in range(self.num_evs):
                if self.env.is_plugged_in(i, t):
                    # Charge grid power + discharge grid power (which is negative)
                    c_grid = charge_powers[t, i] / self.env.charge_eff if charge_powers[t, i] > 0 else 0.0
                    d_grid = - discharge_powers[t, i] * self.env.discharge_eff if discharge_powers[t, i] > 0 else 0.0
                    ev_load += (c_grid + d_grid)
                    
            net_load = base_load - solar_gen + ev_load
            price = self.env.p_base + self.env.congestion_factor * net_load
            price = max(0.02, price) # Clamp price floor
            
            # The system-wide grid cost is: Net Load * Price
            total_cost += net_load * price * self.dt

        # 2. Degradation Costs
        for i in range(self.num_evs):
            offset = self.var_offsets[i]
            hours = self.connected_hours[i]
            for idx in range(len(hours)):
                c_p = x[offset + 2 * idx]
                d_p = x[offset + 2 * idx + 1]
                
                # Degradation proportional to power exchanged
                total_cost += self.env.degradation_cost * (c_p + d_p) * self.dt

        # 3. Soft Departure Penalties
        for i in range(self.num_evs):
            dep = self.env.ev_profiles[i]['departure']
            target = self.env.ev_profiles[i]['target_soc']
            
            # SOC at departure step
            final_soc = socs[dep, i]
            if final_soc < target:
                total_cost += self.env.departure_penalty_coef * ((target - final_soc) ** 2)
                
        return total_cost

    def get_constraints(self):
        constraints = []
        
        # SOC constraint helper: must stay between 0.15 and 1.0 at every step
        def make_soc_constraint(ev_idx, step_idx, is_lower):
            if is_lower:
                return lambda x: self._get_single_soc(x, ev_idx, step_idx) - 0.15
            else:
                return lambda x: 1.0 - self._get_single_soc(x, ev_idx, step_idx)

        # Build individual constraints for optimizer
        for i in range(self.num_evs):
            hours = self.connected_hours[i]
            for k in range(len(hours) + 1):
                # Add inequality constraints
                constraints.append({
                    'type': 'ineq',
                    'fun': make_soc_constraint(i, k, is_lower=True)
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': make_soc_constraint(i, k, is_lower=False)
                })
                
        return constraints

    def _get_single_soc(self, x, ev_idx, step_in_connection):
        """Helper to compute SOC for a specific EV at step k of connection."""
        hours = self.connected_hours[ev_idx]
        offset = self.var_offsets[ev_idx]
        
        soc = self.env.ev_profiles[ev_idx]['init_soc']
        for k in range(step_in_connection):
            if k >= len(hours):
                break
            c_p = x[offset + 2 * k]
            d_p = x[offset + 2 * k + 1]
            
            c_soc = (c_p * self.env.charge_eff * self.dt) / self.env.battery_capacity
            d_soc = (d_p * self.dt) / (self.env.battery_capacity * self.env.discharge_eff)
            soc += c_soc - d_soc
            
        return soc

    def solve(self):
        """Solves the optimization problem using Scipy minimize (SLSQP)."""
        # Initial guess: simple uniform charge to meet target
        x0 = np.zeros(self.total_vars)
        for i in range(self.num_evs):
            offset = self.var_offsets[i]
            hours = self.connected_hours[i]
            init = self.env.ev_profiles[i]['init_soc']
            target = self.env.ev_profiles[i]['target_soc']
            
            # Simple charging pattern guess
            soc_needed = max(0.0, target - init)
            energy_needed = (soc_needed * self.env.battery_capacity) / self.env.charge_eff
            charge_per_step = min(self.env.charge_rate, energy_needed / max(1, len(hours)))
            
            for idx in range(len(hours)):
                x0[offset + 2 * idx] = charge_per_step # Initial guess for charge
                x0[offset + 2 * idx + 1] = 0.0         # Initial guess for discharge is 0
                
        # Bounds for each variable: [0, self.env.charge_rate] or [0, self.env.discharge_rate]
        bounds = []
        for i in range(self.num_evs):
            hours = self.connected_hours[i]
            for _ in range(len(hours)):
                bounds.append((0.0, self.env.charge_rate))     # Charge bounds
                bounds.append((0.0, self.env.discharge_rate))  # Discharge bounds
                
        constraints = self.get_constraints()
        
        print("Running centralized global optimization (SLSQP)...")
        res = minimize(
            self.objective_function, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-5}
        )
        
        if not res.success:
            print(f"Optimization warning: {res.message}")
            
        # Extract optimal values
        charge_powers, discharge_powers = self._get_ev_schedules(res.x)
        socs = self._get_soc_trajectories(charge_powers, discharge_powers)
        
        # Calculate hourly profiles
        net_load = np.zeros(self.num_steps)
        prices = np.zeros(self.num_steps)
        ev_loads = np.zeros((self.num_steps, self.num_evs))
        
        for t in range(self.num_steps):
            base_load = self.env.base_load_profile[t]
            solar_gen = self.env.solar_profile[t]
            
            for i in range(self.num_evs):
                if self.env.is_plugged_in(i, t):
                    c_grid = charge_powers[t, i] / self.env.charge_eff if charge_powers[t, i] > 0 else 0.0
                    d_grid = - discharge_powers[t, i] * self.env.discharge_eff if discharge_powers[t, i] > 0 else 0.0
                    ev_loads[t, i] = c_grid + d_grid
                    
            net_load[t] = base_demand = base_load - solar_gen + np.sum(ev_loads[t, :])
            prices[t] = max(0.02, self.env.p_base + self.env.congestion_factor * net_load[t])
            
        return {
            'success': res.success,
            'total_cost': res.fun,
            'charge_powers': charge_powers,
            'discharge_powers': discharge_powers,
            'socs': socs[:-1, :], # match 24 steps
            'net_load': net_load,
            'prices': prices,
            'ev_loads': ev_loads
        }
