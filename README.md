# NeuroGrid-V2G: Decentralized Energy Management in Smart Grids

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Multi-Agent Reinforcement Learning (MARL) framework and real-time interactive simulation platform for coordinating Electric Vehicle (EV) fleets in smart microgrids using Vehicle-to-Grid (V2G) technology and dynamic congestion pricing.

---

## 📌 Key Highlights

- **Decentralized Coordination**: Autonomous EV agents learn optimal charging/discharging policies using Deep Q-Networks (PyTorch DQN) and Tabular Q-Learning under local reward signals without invasive central control.
- **Game-Theoretic Validation**: Benchmarks decentralized efficiency against a centralized mathematical Social Optimum (SLSQP solver) using the **Price of Anarchy (PoA = 1.08)**.
- **V2G Peak Shaving & Valley Filling**: Automatically shifts charging to midday solar surpluses and feeds energy back during evening peak hours (shaving peak demand from 55 kW down to 30 kW).
- **Interactive Web Dashboard**: Real-time glassmorphism web interface built with HTML5, CSS3, and Chart.js to monitor live agent states, battery levels, dynamic pricing, and learning curves.

---

## 🏗️ System Architecture

```mermaid
graph TD
    Grid["<b>Smart Grid Microgrid</b><br/>Base Load + Solar Generation"] --> Pricing["<b>Dynamic Pricing Engine</b><br/>P_t = max(0.02, P_base + γ · NetLoad)"]
    Pricing --> EV1["<b>EV Agent 1</b> (DQN)<br/>State: t, SOC, Price, T_dep<br/>Action: Charge / V2G / Idle"]
    Pricing --> EV2["<b>EV Agent 2</b> (DQN)<br/>State: t, SOC, Price, T_dep<br/>Action: Charge / V2G / Idle"]
    Pricing --> EVN["<b>EV Agent N</b> (DQN)<br/>State: t, SOC, Price, T_dep<br/>Action: Charge / V2G / Idle"]
    EV1 -->|Power Flow| Grid
    EV2 -->|Power Flow| Grid
    EVN -->|Power Flow| Grid
```

---

## 🔬 Mathematical Formulation

### 1. State Space ($s_t^i$)
Each EV agent observes a 5-dimensional continuous state vector:

$$s_t^i = \left[ \frac{t}{24}, \; \text{SOC}_t^i, \; \frac{P_t}{P_{\text{max}}}, \; \frac{T_{\text{dep}}^i - t}{24}, \; \text{SOC}_{\text{target}}^i \right]$$

### 2. Action Space ($a_t^i$)
Each connected EV chooses among 3 discrete actions:
- `0` **(Charge)**: Draws $+7.37\text{ kW}$ from the grid ($+14\%$ SOC/hr).
- `1` **(Discharge / V2G)**: Injects $-6.65\text{ kW}$ to the grid ($-14\%$ SOC/hr, constrained to $\text{SOC}_t^i \ge 0.15$).
- `2` **(Idle)**: $0\text{ kW}$ power exchange.

### 3. Dynamic Congestion Pricing ($P_t$)
Electricity tariff is determined dynamically based on aggregate net demand:

$$P_t = \max\left(0.02, \; P_{\text{base}} + \gamma \cdot \text{Net Load}_t\right)$$

$$\text{Net Load}_t = L_{\text{base}, t} - G_{\text{solar}, t} + \sum_{i=1}^N P_{\text{ev}, t}^i$$

### 4. Reward Function ($R_t^i$)
Agent reward balances electricity costs/revenues, battery wear, and departure requirements:

$$R_t^i = \text{FinancialReward}_t^i - \text{DegradationCost}_t^i - \text{DeparturePenalty}_t^i$$

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Saeidabsnjd/NeuroGrid-V2G.git
cd NeuroGrid-V2G
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
python test_models.py
```

### 3. Start the Interactive Simulator
```bash
python server.py
```
Open your web browser and navigate to `http://localhost:8080` to launch the live dashboard.

---

## 📁 Project Structure

```text
NeuroGrid-V2G/
├── assets/                  # High-resolution benchmark figures
├── web/                     # Web dashboard frontend
│   ├── index.html           # Dashboard UI
│   ├── style.css           # Glassmorphism dark-theme styling
│   └── app.js              # Real-time state polling & Chart.js rendering
├── grid_env.py              # Physics-accurate Smart Grid microgrid environment
├── marl_agents.py           # Tabular Q-Learning & PyTorch DQN implementations
├── optimization.py          # Centralized global optimization (Scipy SLSQP)
├── server.py                # Thread-safe multi-threaded HTTP/REST API server
├── test_models.py           # Automated unit test suite
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.