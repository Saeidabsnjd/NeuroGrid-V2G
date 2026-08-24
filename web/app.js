// Global dashboard elements
let loadChart = null;
let priceChart = null;
let learningChart = null;

let isPolling = false;
let pollIntervalId = null;
let playIntervalId = null;

let currentPlayHour = 0;
let simulationData = null; // Stored results of /api/run_episode

// Custom Chart.js vertical time-step indicator plugin
const verticalLinePlugin = {
    id: 'verticalLine',
    afterDraw: (chart) => {
        const options = chart.options.plugins.verticalLine;
        if (options && options.show && options.index !== undefined) {
            const xIndex = options.index;
            const meta = chart.getDatasetMeta(0);
            if (meta && meta.data && meta.data[xIndex]) {
                const x = meta.data[xIndex].x;
                const ctx = chart.ctx;
                const topY = chart.scales.y.top;
                const bottomY = chart.scales.y.bottom;
                
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(x, topY);
                ctx.lineTo(x, bottomY);
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = 'rgba(0, 210, 255, 0.45)';
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.restore();
            }
        }
    }
};
Chart.register(verticalLinePlugin);

// Helper for UI tooltips
const tooltipEl = document.getElementById('tooltip');
function showTooltip(e, content) {
    tooltipEl.innerHTML = content;
    tooltipEl.style.opacity = 1;
    tooltipEl.style.left = (e.pageX + 15) + 'px';
    tooltipEl.style.top = (e.pageY + 10) + 'px';
}
function hideTooltip() {
    tooltipEl.style.opacity = 0;
}

// Chart Options Config (Shared Styles)
const getChartOptions = (yLabel, showIndicator = false) => ({
    responsive: true,
    maintainAspectRatio: false,
    color: '#94a3b8',
    scales: {
        x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter' } }
        },
        y: {
            title: { display: true, text: yLabel, color: '#94a3b8', font: { family: 'Inter', weight: '600' } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter' } }
        }
    },
    plugins: {
        legend: { labels: { color: '#f8fafc', font: { family: 'Inter', size: 11 } } },
        verticalLine: { show: showIndicator, index: 0 }
    }
});

// Initialize Chart.js Instances
function initCharts() {
    const hours = Array.from({length: 24}, (_, i) => `${String(i).padStart(2, '0')}:00`);
    
    // 1. Load Chart
    const ctxLoad = document.getElementById('chart-load').getContext('2d');
    loadChart = new Chart(ctxLoad, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [
                { label: 'Base Load (Home)', data: [], borderColor: '#64748b', borderDash: [4, 4], borderWidth: 1.5, fill: false, tension: 0.2 },
                { label: 'Solar Generation', data: [], borderColor: '#10b981', borderDash: [2, 2], borderWidth: 1.5, fill: false, tension: 0.2 },
                { label: 'Net Load (MARL)', data: [], borderColor: '#00d2ff', borderWidth: 3, pointBackgroundColor: '#00d2ff', fill: false, tension: 0.2 },
                { label: 'Net Load (Optimum SO)', data: [], borderColor: '#f59e0b', borderWidth: 2, fill: false, tension: 0.2 }
            ]
        },
        options: getChartOptions('Electricity Demand (kW)', true)
    });

    // 2. Price Chart
    const ctxPrice = document.getElementById('chart-price').getContext('2d');
    priceChart = new Chart(ctxPrice, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [
                { label: 'Dynamic Price (MARL)', data: [], borderColor: '#00d2ff', borderWidth: 3, fill: false, tension: 0.2 },
                { label: 'Price (Optimum SO)', data: [], borderColor: '#f59e0b', borderWidth: 2, fill: false, tension: 0.2 }
            ]
        },
        options: getChartOptions('Price ($ / kWh)', true)
    });

    // 3. Learning Curves
    const ctxLearning = document.getElementById('chart-learning').getContext('2d');
    learningChart = new Chart(ctxLearning, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Avg Reward', data: [], borderColor: '#f59e0b', borderWidth: 1.5, yAxisID: 'yReward', fill: false, tension: 0.1 },
                { label: 'Price of Anarchy', data: [], borderColor: '#00d2ff', borderWidth: 2.5, yAxisID: 'yPoa', fill: false, tension: 0.1 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#94a3b8',
            scales: {
                x: {
                    title: { display: true, text: 'Epochs', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' }
                },
                yReward: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Average Reward', color: '#f59e0b' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                yPoa: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Price of Anarchy (PoA)', color: '#00d2ff' },
                    grid: { drawOnChartArea: false },
                    min: 1.0,
                    max: 3.5
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });
}

// Fetch current status and update dashboard stats
async function updateStatus() {
    try {
        const res = await fetch('/api/status');
        const state = await res.json();
        
        // Update stats text
        document.getElementById('val-epoch').innerText = `${state.current_epoch} / ${state.max_epochs}`;
        document.getElementById('progress-bar').style.width = `${(state.current_epoch / state.max_epochs) * 100}%`;
        document.getElementById('val-epsilon').innerText = state.epsilon.toFixed(3);
        
        const loss = state.loss_history.length > 0 ? state.loss_history[state.loss_history.length - 1] : 0;
        document.getElementById('val-loss').innerText = loss.toFixed(4);
        
        const poa = state.poa_history.length > 0 ? state.poa_history[state.poa_history.length - 1] : 1.0;
        document.getElementById('val-poa').innerText = poa.toFixed(2);
        
        // Update Poa warning indicator colors
        const poaInd = document.getElementById('poa-indicator');
        if (poa <= 1.05) {
            poaInd.innerText = "Ideal Efficiency";
            poaInd.className = "stat-indicator poa-good";
        } else if (poa <= 1.25) {
            poaInd.innerText = "Slight Congestion Loss";
            poaInd.className = "stat-indicator poa-warn";
        } else {
            poaInd.innerText = "High Inefficiency";
            poaInd.className = "stat-indicator poa-bad";
        }
        
        // Update learning curve chart histories
        const epochs = Array.from({length: state.rewards_history.length}, (_, i) => i + 1);
        learningChart.data.labels = epochs;
        learningChart.data.datasets[0].data = state.rewards_history;
        learningChart.data.datasets[1].data = state.poa_history;
        learningChart.update('none'); // silent update

        // If training has stopped from backend, update UI buttons
        if (!state.is_training && isPolling) {
            stopPolling();
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            toggleInputs(false);
            
            // Reload the final trained curves
            fetchAndRenderEpisode();
        }
    } catch (e) {
        console.error("Error fetching status API", e);
    }
}

// Fetch and draw actual charging curves
async function fetchAndRenderEpisode() {
    try {
        const res = await fetch('/api/run_episode');
        simulationData = await res.json();
        
        // Update curves data
        loadChart.data.datasets[0].data = simulationData.grid_baselines.base_demand;
        loadChart.data.datasets[1].data = simulationData.grid_baselines.solar_gen;
        loadChart.data.datasets[2].data = simulationData.marl.net_load;
        loadChart.data.datasets[3].data = simulationData.so.net_load;
        loadChart.update();
        
        priceChart.data.datasets[0].data = simulationData.marl.prices;
        priceChart.data.datasets[1].data = simulationData.so.prices;
        priceChart.update();
        
        // Update EV microgrid nodes online badge
        document.getElementById('active-nodes-count').innerText = `${env_num_evs} Agents Online`;
        
        // Start or reset play animation loop
        startPlaybackLoop();
        
    } catch (e) {
        console.error("Error running evaluation episode", e);
    }
}

// Playback Animation Loop (Hour-by-hour stepping)
function startPlaybackLoop() {
    if (playIntervalId) clearInterval(playIntervalId);
    
    currentPlayHour = 0;
    
    // Draw initial EV cards
    renderEVCards();
    
    playIntervalId = setInterval(() => {
        if (!simulationData) return;
        
        // Ticks hours: 0 -> 23 -> 0
        currentPlayHour = (currentPlayHour + 1) % 24;
        
        // Update vertical tracking lines in charts
        loadChart.options.plugins.verticalLine.index = currentPlayHour;
        loadChart.update('none');
        
        priceChart.options.plugins.verticalLine.index = currentPlayHour;
        priceChart.update('none');
        
        // Refresh states in EV cards
        updateEVCardsAtHour(currentPlayHour);
        
    }, 850); // Tick every 850ms
}

// Render Initial EV Nodes Cards
let env_num_evs = 10;
function renderEVCards() {
    const grid = document.getElementById('ev-nodes-grid');
    grid.innerHTML = '';
    
    // Dynamically match length of incoming simulation profiles
    if (simulationData && simulationData.ev_profiles) {
        env_num_evs = simulationData.ev_profiles.length;
    }
    
    for (let i = 0; i < env_num_evs; i++) {
        const card = document.createElement('div');
        card.className = 'node-card';
        card.id = `ev-card-${i}`;
        
        card.innerHTML = `
            <div class="node-card-header">
                <span class="node-name">EV Agent ${i+1}</span>
                <span class="node-status-dot status-idle" id="ev-dot-${i}"></span>
            </div>
            <div class="node-card-body">
                <div class="battery-row">
                    <div class="battery-track">
                        <div class="battery-fill" id="ev-fill-${i}" style="width: 0%"></div>
                    </div>
                    <span class="battery-text" id="ev-soc-${i}">0%</span>
                </div>
                <div class="node-info-text" id="ev-info-${i}">Commuting</div>
            </div>
        `;
        
        // Tooltip listeners
        card.addEventListener('mousemove', (e) => {
            if (!simulationData) return;
            const profile = simulationData.ev_profiles[i];
            const socVal = (simulationData.marl.ev_socs[currentPlayHour][i] * 100).toFixed(0);
            const loadVal = simulationData.marl.ev_loads[currentPlayHour][i].toFixed(1);
            
            let status = "Commuting (Driving)";
            const arr = profile.arrival;
            const dep = profile.departure;
            const isPlugged = isPluggedIn(arr, dep, currentPlayHour);
            
            if (isPlugged) {
                if (loadVal > 0) status = "Charging (Drawing grid power)";
                else if (loadVal < 0) status = "Discharging (V2G Feed-in active)";
                else status = "Idling (Grid connected)";
            }

            const content = `
                <h4>EV Agent ${i+1} Details</h4>
                <div class="tooltip-row"><span>Status:</span><span class="tooltip-val">${status}</span></div>
                <div class="tooltip-row"><span>Commute Schedule:</span><span class="tooltip-val">Arrive ${arr}:00 | Depart ${dep}:00</span></div>
                <div class="tooltip-row"><span>Battery Level:</span><span class="tooltip-val">${socVal}% / ${(profile.target_soc * 100)}% Target</span></div>
                <div class="tooltip-row"><span>Load Exchange:</span><span class="tooltip-val">${loadVal} kW</span></div>
            `;
            showTooltip(e, content);
        });
        
        card.addEventListener('mouseleave', hideTooltip);
        
        grid.appendChild(card);
    }
}

// Utility to check plugin state matches backend schedule
function isPluggedIn(arr, dep, hour) {
    if (arr > dep) {
        return hour >= arr || hour < dep;
    } else {
        return hour >= arr && hour < dep;
    }
}

// Update EV battery levels and flows at current animated play hour
function updateEVCardsAtHour(hour) {
    if (!simulationData) return;
    
    for (let i = 0; i < env_num_evs; i++) {
        try {
            const profile = simulationData.ev_profiles[i];
            const card = document.getElementById(`ev-card-${i}`);
            const dot = document.getElementById(`ev-dot-${i}`);
            const fill = document.getElementById(`ev-fill-${i}`);
            const socText = document.getElementById(`ev-soc-${i}`);
            const info = document.getElementById(`ev-info-${i}`);
            
            if (!profile || !card) continue;
            
            const isPlugged = isPluggedIn(profile.arrival, profile.departure, hour);
            const soc = simulationData.marl.ev_socs[hour][i];
            const load = simulationData.marl.ev_loads[hour][i];
            
            // Update battery graphical bar
            if (fill) fill.style.width = `${soc * 100}%`;
            if (socText) socText.innerText = `${(soc * 100).toFixed(0)}%`;
            
            if (isPlugged) {
                card.classList.remove('offline-card');
                
                if (load > 0) {
                    // Charging
                    dot.className = "node-status-dot status-charge";
                    info.innerText = `Charging (+${load.toFixed(1)} kW)`;
                } else if (load < 0) {
                    // Discharging (V2G)
                    dot.className = "node-status-dot status-discharge";
                    info.innerText = `Discharging (${load.toFixed(1)} kW)`;
                } else {
                    // Idle
                    dot.className = "node-status-dot status-idle";
                    info.innerText = "Connected (Idle)";
                }
            } else {
                // Not connected (driving)
                card.classList.add('offline-card');
                dot.className = "node-status-dot status-offline";
                info.innerText = "Commuting / Driving";
            }
        } catch (e) {
            console.error("Error updating card " + i + " at hour " + hour, e);
        }
    }
}

// Polling routines
function startPolling() {
    if (pollIntervalId) clearInterval(pollIntervalId);
    isPolling = true;
    updateStatus();
    pollIntervalId = setInterval(updateStatus, 800);
}

function stopPolling() {
    isPolling = false;
    if (pollIntervalId) {
        clearInterval(pollIntervalId);
        pollIntervalId = null;
    }
}

function toggleInputs(disabled) {
    document.getElementById('alg-select').disabled = disabled;
    document.getElementById('num-evs').disabled = disabled;
    document.getElementById('max-epochs').disabled = disabled;
}

// Setup Event Listeners
function setupListeners() {
    
    // 1. Train button click
    document.getElementById('btn-start').addEventListener('click', async () => {
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-stop').disabled = false;
        toggleInputs(true);
        
        // Trigger config changes first
        const alg_type = document.getElementById('alg-select').value;
        const num_evs = parseInt(document.getElementById('num-evs').value);
        env_num_evs = num_evs;
        const max_epochs = parseInt(document.getElementById('max-epochs').value);
        
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alg_type, num_evs, max_epochs })
            });
            
            // Start background training
            await fetch('/api/start', { method: 'POST' });
            startPolling();
            
        } catch (e) {
            console.error("Error starting training simulation", e);
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            toggleInputs(false);
        }
    });

    // 2. Stop button click
    document.getElementById('btn-stop').addEventListener('click', async () => {
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        
        await fetch('/api/stop', { method: 'POST' });
        stopPolling();
        toggleInputs(false);
    });

    // 3. Reset button click
    document.getElementById('btn-reset').addEventListener('click', async () => {
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        toggleInputs(false);
        
        stopPolling();
        
        await fetch('/api/reset', { method: 'POST' });
        
        // Clear learning chart
        learningChart.data.labels = [];
        learningChart.data.datasets[0].data = [];
        learningChart.data.datasets[1].data = [];
        learningChart.update();
        
        // Clear current states
        document.getElementById('val-epoch').innerText = "0 / 500";
        document.getElementById('progress-bar').style.width = `0%`;
        document.getElementById('val-epsilon').innerText = "1.000";
        document.getElementById('val-loss').innerText = "0.000";
        document.getElementById('val-poa').innerText = "1.00";
        
        // Refresh curves
        fetchAndRenderEpisode();
    });
}

// Window load init
window.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    setupListeners();
    // Fetch and render initial status
    fetchAndRenderEpisode();
    
    // Sync with server if training is active in the background
    try {
        const res = await fetch('/api/status');
        const state = await res.json();
        if (state.is_training) {
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            toggleInputs(true);
            startPolling();
        }
    } catch (e) {
        console.error("Error syncing training status on load", e);
    }
});
