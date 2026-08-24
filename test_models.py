import numpy as np
import torch
from grid_env import SmartGridEnv
from marl_agents import QLearningAgent, DQNAgent
from optimization import CentralizedOptimizer

def test_environment():
    print("Testing SmartGridEnv...")
    env = SmartGridEnv(num_evs=5)
    
    # Check profiles
    assert len(env.ev_profiles) == 5
    assert len(env.base_load_profile) == 24
    assert len(env.solar_profile) == 24
    
    # Check reset
    states = env.reset()
    assert len(states) == 5
    assert all(len(s) == 5 for s in states)
    assert np.allclose(env.soc, [p['init_soc'] for p in env.ev_profiles])
    
    # Check step
    # Actions: 2 Charge, 1 Discharge, 2 Idle
    actions = [0, 0, 1, 2, 2]
    next_states, rewards, dones, info = env.step(actions)
    
    assert len(next_states) == 5
    assert len(rewards) == 5
    assert len(dones) == 5
    assert 'net_load' in info
    assert 'price' in info
    assert 'ev_total_demand' in info
    
    # Check state bounds after action
    # Charging should increase SOC
    if env.is_plugged_in(0, 0):
        assert env.soc[0] > env.ev_profiles[0]['init_soc']
    # Discharging should decrease SOC (or stay same if clamped to min SOC)
    if env.is_plugged_in(2, 0):
        assert env.soc[2] <= env.ev_profiles[2]['init_soc']
        
    print("SmartGridEnv tests passed successfully!")

def test_centralized_optimizer():
    print("Testing CentralizedOptimizer...")
    env = SmartGridEnv(num_evs=4)
    optimizer = CentralizedOptimizer(env)
    res = optimizer.solve()
    
    assert res['success'] == True
    assert len(res['net_load']) == 24
    assert len(res['prices']) == 24
    assert res['total_cost'] > 0
    assert res['socs'].shape == (24, 4)
    
    print("CentralizedOptimizer tests passed successfully!")

def test_agents():
    print("Testing RL Agents...")
    
    # 1. Tabular Q-Learning Agent
    q_agent = QLearningAgent()
    state = [0.0, 0.25, 0.3, 0.5, 0.9]
    d_state = q_agent.discretize_state(state)
    assert isinstance(d_state, tuple)
    assert len(d_state) == 4
    
    action = q_agent.choose_action(state, epsilon=0.1)
    assert action in [0, 1, 2]
    
    next_state = [0.04, 0.28, 0.35, 0.46, 0.9]
    q_agent.learn(state, action, 0.5, next_state, False)
    # Check that Q-table is updated
    assert len(q_agent.q_table) > 0
    
    # 2. PyTorch DQN Agent
    dqn_agent = DQNAgent(state_size=5, action_size=3, batch_size=4)
    
    action_dqn = dqn_agent.choose_action(state, epsilon=0.1)
    assert action_dqn in [0, 1, 2]
    
    # Store mock memory transitions to test training step
    for _ in range(10):
        dqn_agent.store_transition(state, 0, 1.0, next_state, False)
        
    assert len(dqn_agent.memory) == 10
    
    loss = dqn_agent.learn()
    assert loss >= 0.0 # Loss should calculate correctly
    
    print("RL Agents tests passed successfully!")

if __name__ == "__main__":
    print("=== STARTING COMPONENT VERIFICATION ===")
    test_environment()
    test_centralized_optimizer()
    test_agents()
    print("=== ALL TESTS PASSED SUCCESSFULLY ===")
