import os
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

from grid_env import SmartGridEnv
from marl_agents import QLearningAgent, DQNAgent
from optimization import CentralizedOptimizer

# Global Simulation State
sim_state = {
    'is_training': False,
    'current_epoch': 0,
    'max_epochs': 500,
    'rewards_history': [],
    'loss_history': [],
    'poa_history': [],
    'active_alg': 'dqn',
    'num_evs': 10,
    'epsilon': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.985
}

# Thread and Lock configuration
train_thread = None
state_lock = threading.Lock()

# Env, Agents, and Optimizer baselines
env = None
agents = []
so_results = None

def init_simulation():
    global env, agents, so_results, sim_state
    sim_state['current_epoch'] = 0
    sim_state['rewards_history'] = []
    sim_state['loss_history'] = []
    sim_state['poa_history'] = []
    sim_state['epsilon'] = 1.0
    
    # 1. Initialize environment
    env = SmartGridEnv(num_evs=sim_state['num_evs'])
    
    # 2. Solve Centralized Social Optimum Baseline
    optimizer = CentralizedOptimizer(env)
    so_results = optimizer.solve()
    
    # 3. Create Decentralized MARL Agents
    agents = []
    for _ in range(sim_state['num_evs']):
        if sim_state['active_alg'] == 'q_learning':
            agents.append(QLearningAgent())
        else:
            agents.append(DQNAgent(state_size=5, action_size=3))

# Initialize at launch
init_simulation()

def compute_episode_cost(env_history):
    """
    Computes system-wide grid costs (electricity + degradation + departure penalty)
    from env logs to match optimization metrics.
    """
    total_cost = 0.0
    
def compute_episode_cost(local_env, env_history):
    """
    Computes system-wide grid costs (electricity + degradation + departure penalty)
    from env logs to match optimization metrics.
    """
    total_cost = 0.0
    
    # Grid energy costs
    for t in range(local_env.num_steps):
        net_load = env_history['net_loads'][t]
        price = env_history['prices'][t]
        total_cost += net_load * price * local_env.dt
        
    # Wear degradation costs
    for t in range(local_env.num_steps):
        ev_powers = env_history['ev_loads'][t]
        for p in ev_powers:
            total_cost += local_env.degradation_cost * abs(p) * local_env.dt
            
    # Soft departure penalties
    for i in range(local_env.num_evs):
        dep = local_env.ev_profiles[i]['departure']
        target = local_env.ev_profiles[i]['target_soc']
        final_soc = env_history['ev_socs'][dep - 1][i] # SOC just before departure
        if final_soc < target:
            total_cost += local_env.departure_penalty_coef * ((target - final_soc) ** 2)
            
    return total_cost

def run_evaluation_episode():
    """Runs a single evaluation episode with greedy policies (epsilon=0)."""
    # Create isolated environment for evaluation to avoid thread conflicts
    local_env = SmartGridEnv(num_evs=env.num_evs)
    states = local_env.reset()
    done = False
    
    eval_history = {
        'net_loads': [],
        'prices': [],
        'ev_loads': [],
        'ev_socs': [],
        'base_demands': [],
        'solar_gens': []
    }
    
    while not done:
        actions = []
        for i in range(local_env.num_evs):
            # Select greedy action (epsilon=0) under thread-safe lock
            with state_lock:
                action = agents[i].choose_action(states[i], epsilon=0.0)
            actions.append(action)
            
        next_states, rewards, dones, info = local_env.step(actions)
        
        eval_history['net_loads'].append(float(info['net_load']))
        eval_history['prices'].append(float(info['price']))
        eval_history['ev_loads'].append(local_env.ev_loads_history[-1].tolist())
        eval_history['ev_socs'].append(local_env.ev_soc_history[-1].tolist())
        eval_history['base_demands'].append(float(info['base_demand']))
        eval_history['solar_gens'].append(float(info['solar_gen']))
        
        states = next_states
        done = dones[0]
        
    eval_cost = compute_episode_cost(local_env, eval_history)
    return eval_history, eval_cost

def training_worker():
    global sim_state, env, agents, so_results
    
    print("Background training thread started.")
    
    while True:
        with state_lock:
            if not sim_state['is_training'] or sim_state['current_epoch'] >= sim_state['max_epochs']:
                sim_state['is_training'] = False
                break
            
            epoch = sim_state['current_epoch']
            epsilon = sim_state['epsilon']
            active_alg = sim_state['active_alg']

        # 1. Run training episode using isolated environment to avoid conflicts
        train_env = SmartGridEnv(num_evs=env.num_evs)
        states = train_env.reset()
        done = False
        episode_rewards = []
        losses = []
        
        while not done:
            actions = []
            for i in range(train_env.num_evs):
                with state_lock:
                    action = agents[i].choose_action(states[i], epsilon)
                actions.append(action)
                
            next_states, rewards, dones, info = train_env.step(actions)
            episode_rewards.append(np.sum(rewards))
            
            # Learn under lock to prevent concurrent weight modification issues
            for i in range(train_env.num_evs):
                with state_lock:
                    if active_alg == 'q_learning':
                        agents[i].learn(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    else:
                        agents[i].store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                        loss = agents[i].learn()
                        if loss > 0:
                            losses.append(loss)
                            
            states = next_states
            done = dones[0]

        # Update DQN Target Networks periodically under lock
        if active_alg == 'dqn' and epoch % 5 == 0:
            with state_lock:
                for i in range(env.num_evs):
                    agents[i].update_target_network()

        # 2. Evaluate current policy (greedy)
        eval_history, eval_cost = run_evaluation_episode()
        
        # Calculate Price of Anarchy (PoA)
        so_cost = so_results['total_cost']
        poa = eval_cost / so_cost if so_cost > 0 else 1.0
        poa = max(1.0, poa) # PoA is mathematically >= 1.0

        # Update stats
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(losses) if losses else 0.0
        
        with state_lock:
            sim_state['rewards_history'].append(float(avg_reward))
            sim_state['loss_history'].append(float(avg_loss))
            sim_state['poa_history'].append(float(poa))
            
            # Decay exploration rate
            sim_state['epsilon'] = max(sim_state['epsilon_min'], epsilon * sim_state['epsilon_decay'])
            sim_state['current_epoch'] += 1
            
        # Give a small break to prevent 100% CPU lock
        time.sleep(0.01)
        
    print("Background training thread stopped.")

class SimHTTPRequestHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Silence HTTP console logs to keep output clean
        return

    def do_GET(self):
        global sim_state, env, agents, so_results
        
        # Simple router
        if self.path == '/api/status':
            self.send_json_response(sim_state)
            
        elif self.path == '/api/run_episode':
            # Run a greedy evaluation episode to get current curves
            eval_history, eval_cost = run_evaluation_episode()
            
            response_data = {
                'marl': {
                    'net_load': eval_history['net_loads'],
                    'prices': eval_history['prices'],
                    'ev_loads': eval_history['ev_loads'],
                    'ev_socs': eval_history['ev_socs'],
                    'total_cost': eval_cost
                },
                'so': {
                    'net_load': so_results['net_load'].tolist(),
                    'prices': so_results['prices'].tolist(),
                    'ev_loads': so_results['ev_loads'].sum(axis=1).tolist(),
                    'ev_socs': so_results['socs'].tolist(),
                    'total_cost': so_results['total_cost']
                },
                'grid_baselines': {
                    'base_demand': eval_history['base_demands'],
                    'solar_gen': eval_history['solar_gens']
                },
                'ev_profiles': env.ev_profiles
            }
            self.send_json_response(response_data)
            
        else:
            # Serve static files
            self.serve_static_file()

    def do_POST(self):
        global sim_state, train_thread
        
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        else:
            data = {}
        
        if self.path == '/api/config':
            with state_lock:
                if not sim_state['is_training']:
                    new_num_evs = int(data.get('num_evs', sim_state['num_evs']))
                    sim_state['active_alg'] = data.get('alg_type', sim_state['active_alg'])
                    sim_state['max_epochs'] = int(data.get('max_epochs', sim_state['max_epochs']))
                    
                    # Only re-initialize and solve Scipy optimization if the EV count has actually changed!
                    if new_num_evs != sim_state['num_evs'] or env is None:
                        sim_state['num_evs'] = new_num_evs
                        init_simulation()
                    else:
                        # Just reset the histories for the new training run instantly
                        sim_state['current_epoch'] = 0
                        sim_state['rewards_history'] = []
                        sim_state['loss_history'] = []
                        sim_state['poa_history'] = []
                        sim_state['epsilon'] = 1.0
                        
                    self.send_json_response({'status': 'configured'})
                else:
                    self.send_error_response(400, 'Cannot configure while training is active.')
                    
        elif self.path == '/api/start':
            with state_lock:
                if not sim_state['is_training']:
                    sim_state['is_training'] = True
                    train_thread = threading.Thread(target=training_worker)
                    train_thread.daemon = True
                    train_thread.start()
                    self.send_json_response({'status': 'started'})
                else:
                    self.send_json_response({'status': 'already_running'})
                    
        elif self.path == '/api/stop':
            with state_lock:
                sim_state['is_training'] = False
            self.send_json_response({'status': 'stopped'})
            
        elif self.path == '/api/reset':
            with state_lock:
                sim_state['is_training'] = False
            # Wait for thread to exit
            if train_thread:
                train_thread.join(timeout=1.0)
            init_simulation()
            self.send_json_response({'status': 'reset'})
            
        else:
            self.send_error_response(404, 'API Route Not Found')

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode('utf-8'))

    def serve_static_file(self):
        # Set file root directory to web/
        web_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
        
        # Clean path to prevent dir traversal
        parsed_path = self.path.split('?')[0]
        if parsed_path == '/':
            parsed_path = '/index.html'
            
        file_path = os.path.join(web_root, parsed_path.lstrip('/'))
        
        # Ensure we are inside web_root directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(web_root)):
            self.send_error(403, "Access Denied")
            return
            
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, "File Not Found")
            return
            
        # Determine mime type
        _, ext = os.path.splitext(file_path)
        mime_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon'
        }
        mime = mime_types.get(ext.lower(), 'application/octet-stream')
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimHTTPRequestHandler)
    print(f"Simulation dashboard server running locally on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulation server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
