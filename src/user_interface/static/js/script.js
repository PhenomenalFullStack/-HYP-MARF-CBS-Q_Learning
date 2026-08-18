// script.js

(function () {
  "use strict";

  // DOM Elements
  const dom = {
    numVehicles: document.getElementById("num_vehicles"),
    brakingDelay: document.getElementById("braking_delay"),
    sensorNoiseToggle: document.getElementById("sensor_noise_toggle"),
    sensorNoiseStatus: document.getElementById("sensor_noise_status"),
    commLatency: document.getElementById("comm_latency"),
    deviationThreshold: document.getElementById("deviation_threshold"),

    runCbs: document.getElementById("run-cbs"),
    runHybrid: document.getElementById("run-hybrid"),
    generatePlots: document.getElementById("generate_plots"),

    vehiclesReached: document.getElementById("vehicles-reached"),
    progressBar: document.getElementById("progress-bar"),

    collisionsCount: document.getElementById("collisions-count"),
    replansCount: document.getElementById("replans-count"),
    timeStepsCount: document.getElementById("time-steps-count"),

    gridCanvas: document.getElementById("grid-canvas"),
    scheduleDisplay: document.getElementById("schedule-display"),

    metric1: document.getElementById("metric1"),
    metric2: document.getElementById("metric2"),
    chartType: document.getElementById("chartType"),
    collisionChart: document.getElementById("collision-chart"),
  };

  // Colours used for each vehicle, shared by the vehicle dots and the goal markers
  const VEHICLE_COLOURS = {
    A: "#ff0000",
    B: "#0044ff",
    C: "#00aa00",
    D: "#ff8800",
    E: "#9900cc",
    F: "#ff00ff",
    G: "#090310",
    H: "#00ccc2",
  };

  // Chart.js instance
  let chartInstance = null;

  // State
  let lastResult = null;
  let animationInterval = null;
  let animationIndex = 0;
  let vehicleGoals = {};
  let previousPositions = {};
  let baselineData = null;
  let hybridData = null;
  let currentDataSource = 'baseline'; // 'baseline' or 'hybrid'
  let lastGridStateForRedraw = null;

  // Load both datasets
  async function loadDatasets() {
    try {
      // Load baseline (intersection_schedules.json)
      let response = await fetch('/intersection_schedules.json');
      if (!response.ok) throw new Error('Failed to load baseline dataset');
      baselineData = await response.json();
      console.log(`Loaded baseline: ${Object.keys(baselineData).length} configurations`);

      // Load hybrid (simulation_results.json)
      response = await fetch('/simulation_results.json');
      if (!response.ok) throw new Error('Failed to load hybrid dataset');
      hybridData = await response.json();
      console.log(`Loaded hybrid: ${hybridData.length} configurations`);

      return true;
    } catch (error) {
      console.error('Error loading datasets:', error);
      baselineData = null;
      hybridData = null;
      return false;
    }
  }

  // Helper to map float to integer (0/1)
  function mapToInt(value) {
    return Math.abs(value) < 0.001 ? 0 : 1;
  }

  // Extract data for charts based on user selections and current data source
  function extractChartData() {
    const metric1 = dom.metric1.value;
    const metric2 = dom.metric2.value;
    const numVehicles = parseInt(dom.numVehicles.value) || 4;
    const sensorNoise = dom.sensorNoiseToggle.checked ? 0.1 : 0.0;
    const commLatency = parseFloat(dom.commLatency.value) || 0;
    const brakingDelay = parseFloat(dom.brakingDelay.value) || 0;

    console.log(`Extracting chart data from ${currentDataSource} with:`, { numVehicles, sensorNoise, commLatency, brakingDelay });

    let configs = [];
    let labels = [];

    if (currentDataSource === 'baseline') {
      if (!baselineData) {
        console.error('Baseline data not loaded');
        return null;
      }
      for (const [key, entry] of Object.entries(baselineData)) {
        const params = entry.parameters || {};
        if (params.num_vehicles !== numVehicles) continue;
        if (Math.abs((params.sensor_noise || 0) - sensorNoise) > 0.001) continue;
        if (Math.abs((params.comm_latency || 0) - commLatency) > 0.001) continue;
        const bd = params.braking_delay || 0;
        if (Math.abs(bd - brakingDelay) > 0.001) continue;
        configs.push(entry);
        labels.push(`BD=${bd.toFixed(1)}`);
      }
    } else { // hybrid
      if (!hybridData) {
        console.error('Hybrid data not loaded');
        return null;
      }
      // Map UI values to integers (0/1)
      const uiSn = mapToInt(sensorNoise);
      const uiCl = mapToInt(commLatency);
      const uiBd = mapToInt(brakingDelay);

      for (const entry of hybridData) {
        if (entry.num_vehicles !== numVehicles) continue;
        const cfg = entry.configuration || {};
        const sn = cfg.sensor_noise ?? 0;
        const cl = cfg.comm_latency ?? 0;
        const bd = cfg.braking_delay ?? 0;
        if (sn !== uiSn) continue;
        if (cl !== uiCl) continue;
        if (bd !== uiBd) continue;
        configs.push(entry);
        labels.push(`BD=${bd}`);
      }
    }

    console.log(`Found ${configs.length} configurations matching the parameters`);

    if (configs.length === 0) {
      console.warn('No configurations found for the selected parameters');
      return null;
    }

    // Sort by braking delay
    configs.sort((a, b) => {
      let bdA, bdB;
      if (currentDataSource === 'baseline') {
        bdA = a.parameters?.braking_delay || 0;
        bdB = b.parameters?.braking_delay || 0;
      } else {
        bdA = a.configuration?.braking_delay || 0;
        bdB = b.configuration?.braking_delay || 0;
      }
      return bdA - bdB;
    });

    const data1 = [];
    const data2 = [];

    for (const config of configs) {
      let cbsSteps = 0, hybridSteps = 0, collisions = 0, replans = 0, delay = 0;

      if (currentDataSource === 'baseline') {
        const cbs = config.cbs || {};
        cbsSteps = cbs.steps || 0;
        collisions = cbs.collisions || 0;
        replans = cbs.replans_triggered || 0;
        delay = cbs.delay_steps || 0;
        hybridSteps = 0;
      } else {
        const cbsSchedule = config.cbs_schedule || {};
        const actualSchedule = config.actual_schedule || {};
        const countSteps = (schedule) => {
          let maxSteps = 0;
          for (const vid of Object.keys(schedule)) {
            const steps = schedule[vid];
            if (steps && steps.length > maxSteps) maxSteps = steps.length;
          }
          return maxSteps;
        };
        cbsSteps = countSteps(cbsSchedule);
        hybridSteps = countSteps(actualSchedule);
        collisions = config.collisions || 0;
        replans = config.replans_triggered || 0;
        delay = Math.max(0, hybridSteps - cbsSteps);
      }

      let val1 = 0;
      let val2 = 0;

      switch (metric1) {
        case 'cbs':
          val1 = cbsSteps;
          break;
        case 'hybrid':
          val1 = hybridSteps || cbsSteps;
          break;
        case 'replans':
          val1 = replans;
          break;
        default:
          val1 = 0;
      }

      switch (metric2) {
        case 'collisions':
          val2 = collisions;
          break;
        case 'timesteps':
          val2 = cbsSteps || hybridSteps;
          break;
        case 'delay':
          val2 = delay;
          break;
        default:
          val2 = 0;
      }

      data1.push(val1);
      data2.push(val2);
    }

    console.log('Chart data prepared:', { labels: labels.slice(0, configs.length), data1, data2 });

    return {
      labels: labels.slice(0, configs.length),
      data1: data1,
      data2: data2,
      metric1Label: getMetricLabel(metric1),
      metric2Label: getMetricLabel(metric2),
      configs: configs,
      source: currentDataSource
    };
  }

  function getMetricLabel(metric) {
    const labels = {
      'cbs': 'CBS Steps',
      'hybrid': 'Hybrid Steps',
      'replans': 'Number of Replans',
      'collisions': 'Collisions',
      'timesteps': 'Time Steps',
      'delay': 'Average Delay (steps)'
    };
    return labels[metric] || metric;
  }

  // Generate chart based on user selections
  function generateChart() {
    const chartType = dom.chartType.value;
    const chartData = extractChartData();

    if (!chartData || chartData.data1.length === 0) {
      alert('No data found for the selected parameters. Try different values.\n\nTip: Try changing the "Number of Autonomous Vehicles" or "Braking delay probability" values.');
      return;
    }

    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }

    const canvas = dom.collisionChart;
    const ctx = canvas.getContext('2d');

    const container = canvas.parentElement;
    const containerWidth = container.clientWidth || 400;
    canvas.width = containerWidth;
    canvas.height = Math.min(containerWidth * 0.5, 300);

    const datasets = [
      {
        label: chartData.metric1Label,
        data: chartData.data1,
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
        yAxisID: 'y'
      },
      {
        label: chartData.metric2Label,
        data: chartData.data2,
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
        yAxisID: 'y1'
      }
    ];

    if (chartType === 'pie' || chartType === 'radar') {
      const pieData = chartData.data1;
      const pieLabels = chartData.labels.map((label, i) =>
        `${label} (${pieData[i]})`
      );

      const chartConfig = {
        type: chartType,
        data: {
          labels: pieLabels,
          datasets: [{
            label: chartData.metric1Label,
            data: pieData,
            backgroundColor: [
              'rgba(255, 99, 132, 0.8)',
              'rgba(54, 162, 235, 0.8)',
              'rgba(255, 206, 86, 0.8)',
              'rgba(75, 192, 192, 0.8)',
              'rgba(153, 102, 255, 0.8)',
              'rgba(255, 159, 64, 0.8)',
              'rgba(199, 199, 199, 0.8)',
              'rgba(83, 102, 255, 0.8)'
            ],
            borderColor: [
              'rgba(255, 99, 132, 1)',
              'rgba(54, 162, 235, 1)',
              'rgba(255, 206, 86, 1)',
              'rgba(75, 192, 192, 1)',
              'rgba(153, 102, 255, 1)',
              'rgba(255, 159, 64, 1)',
              'rgba(199, 199, 199, 1)',
              'rgba(83, 102, 255, 1)'
            ],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                padding: 10,
                font: { size: 10 }
              }
            },
            title: {
              display: true,
              text: `${chartData.metric1Label} by Disturbance (${chartData.source})`,
              font: { size: 14, weight: 'bold' }
            }
          }
        }
      };

      chartInstance = new Chart(ctx, chartConfig);
      updateChartCaption(chartData, chartType);
      return;
    }

    const chartConfig = {
      type: chartType,
      data: {
        labels: chartData.labels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              padding: 10,
              font: { size: 11 },
              usePointStyle: true
            }
          },
          title: {
            display: true,
            text: `${chartData.metric1Label} vs ${chartData.metric2Label} (${chartData.source})`,
            font: { size: 14, weight: 'bold' }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                let value = context.parsed.y || context.parsed.r || 0;
                if (label) {
                  label += ': ';
                }
                label += value.toFixed(2);
                return label;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Disturbance',
              font: { weight: 'bold' }
            },
            grid: {
              display: false
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: chartData.metric1Label,
              font: { weight: 'bold' }
            },
            grid: {
              color: 'rgba(0,0,0,0.05)'
            }
          },
          y1: {
            position: 'right',
            beginAtZero: true,
            title: {
              display: true,
              text: chartData.metric2Label,
              font: { weight: 'bold' }
            },
            grid: {
              drawOnChartArea: false,
              color: 'rgba(0,0,0,0.05)'
            }
          }
        }
      }
    };

    if (chartType === 'scatter') {
      const scatterData = chartData.labels.map((label, i) => ({
        x: chartData.data1[i],
        y: chartData.data2[i]
      }));

      chartConfig.data = {
        datasets: [{
          label: `${chartData.metric1Label} vs ${chartData.metric2Label} (${chartData.source})`,
          data: scatterData,
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgba(54, 162, 235, 1)',
          pointRadius: 6,
          pointHoverRadius: 8
        }]
      };

      chartConfig.options.scales = {
        x: {
          title: {
            display: true,
            text: chartData.metric1Label,
            font: { weight: 'bold' }
          },
          grid: {
            color: 'rgba(0,0,0,0.05)'
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: chartData.metric2Label,
            font: { weight: 'bold' }
          },
          grid: {
            color: 'rgba(0,0,0,0.05)'
          }
        }
      };
    }

    chartInstance = new Chart(ctx, chartConfig);
    updateChartCaption(chartData, chartType);
  }

  function updateChartCaption(chartData, chartType) {
    const caption = document.querySelector('.plot-caption');
    if (caption) {
      const numConfigs = chartData.labels.length;
      caption.textContent = `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} Graph showing ${chartData.metric1Label} vs ${chartData.metric2Label} (${chartData.source})`;
    }
  }

  // Helpers
  function setCanvasHeight(heightInPixels) {
    dom.gridCanvas.style.height = heightInPixels + "px";
  }

  function setButtonsEnabled(enabled) {
    dom.runCbs.disabled = !enabled;
    dom.runHybrid.disabled = !enabled;
    dom.runCbs.style.opacity = enabled ? "1" : "0.5";
    dom.runHybrid.style.opacity = enabled ? "1" : "0.5";
    dom.runCbs.style.cursor = enabled ? "pointer" : "not-allowed";
    dom.runHybrid.style.cursor = enabled ? "pointer" : "not-allowed";
  }

  function getParameters() {
    return {
      num_vehicles: parseInt(dom.numVehicles.value, 10) || 4,
      braking_delay: parseFloat(dom.brakingDelay.value) || 0.0,
      sensor_noise: dom.sensorNoiseToggle.checked ? 0.1 : 0.0,
      comm_latency: parseFloat(dom.commLatency.value) || 0.0,
      deviation_threshold: parseFloat(dom.deviationThreshold.value) || 2,
    };
  }

  // Run Simulation
  function runSimulation(endpoint) {
    // Set data source based on endpoint
    if (endpoint === '/run_cbs') {
      currentDataSource = 'baseline';
    } else if (endpoint === '/run_hybrid') {
      currentDataSource = 'hybrid';
    }

    const parameters = getParameters();
    setButtonsEnabled(false);
    dom.progressBar.style.width = "30%";
    setCanvasHeight(600); // change canvas height for new simulation
 
    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parameters),
    })
      .then(function (response) {
        dom.progressBar.style.width = "70%";
        if (!response.ok) {
          throw new Error("Server responded with status " + response.status);
        }
        return response.json();
      })
      .then(function (result) {
        dom.progressBar.style.width = "100%";
        lastResult = result;

        vehicleGoals = {};
        if (result.schedules && result.schedules.length > 0) {
          for (const schedule of result.schedules) {
            const steps = schedule.steps;
            if (steps.length > 0) {
              const lastStep = steps[steps.length - 1];
              vehicleGoals[schedule.vehicle_id] = {
                row: lastStep.row,
                column: lastStep.column,
              };
            }
          }
        }

        dom.collisionsCount.textContent = result.collisions || 0;
        dom.replansCount.textContent = result.replans_triggered || 0;
        dom.timeStepsCount.textContent = result.total_time_steps || 0;

        if (result.schedules && result.schedules.length > 0) {
          updateScheduleDisplay(result.schedules);
        } else {
          dom.scheduleDisplay.innerHTML =
            '<p style="color:#666;text-align:center;">No schedule data</p>';
        }

        if (result.grid_states && result.grid_states.length > 1) {
          animateSimulation(result);
        } else if (result.grid_states && result.grid_states.length === 1) {
          drawGrid(result.grid_states[0], {}, performance.now());
          const totalVehicles =
            result.vehicles_total || parameters.num_vehicles;
          dom.vehiclesReached.textContent =
            (result.vehicles_reached_goal || 0) + " / " + totalVehicles;
          dom.timeStepsCount.textContent = result.total_time_steps || 0;
        } else {
          clearGrid();
          dom.vehiclesReached.textContent = "0 / " + parameters.num_vehicles;
        }

        setButtonsEnabled(true);
        setTimeout(function () {
          dom.progressBar.style.width = "0%";
        }, 800);
      })
      .catch(function (error) {
        console.error("Simulation error:", error);
        dom.progressBar.style.width = "0%";
        setButtonsEnabled(true);
        alert("Error running simulation. Check the console for details.");
      });
  }

  // Grid Animation
  function animateSimulation(result) {
    const gridStates = result.grid_states;
    const reachedPerStep = result.vehicles_reached_per_step || [];
    const totalVehicles = result.vehicles_total || getParameters().num_vehicles;
    const finalTimeSteps = result.total_time_steps || 0;

    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }

    animationIndex = 0;
    previousPositions = {};

    drawGrid(gridStates[0], getWaitingStatus(gridStates[0]), performance.now());
    dom.vehiclesReached.textContent =
      ((reachedPerStep[0] && reachedPerStep[0].length) || 0) +
      " / " +
      totalVehicles;
    dom.timeStepsCount.textContent = "0";

    const milliseconds_per_step = 3000;

    animationInterval = setInterval(function () {
      animationIndex++;

      if (animationIndex >= gridStates.length) {
        clearInterval(animationInterval);
        animationInterval = null;
        dom.vehiclesReached.textContent =
          result.vehicles_reached_goal + " / " + totalVehicles;
        dom.timeStepsCount.textContent = finalTimeSteps;
        dom.progressBar.style.width = "100%";
        return;
      }

      const state = gridStates[animationIndex];
      const waitingStatus = getWaitingStatus(state);
      drawGrid(state, waitingStatus, performance.now());

      const reachedSet = reachedPerStep[animationIndex] || [];
      dom.vehiclesReached.textContent =
        reachedSet.length + " / " + totalVehicles;

      const progress = (animationIndex / gridStates.length) * 100;
      dom.progressBar.style.width = Math.min(progress, 100) + "%";
    }, milliseconds_per_step);
  }

  function getWaitingStatus(currentState) {
    const waiting = {};
    for (const [vehicleId, position] of Object.entries(currentState)) {
      const previous = previousPositions[vehicleId] || null;
      const goal = vehicleGoals[vehicleId] || null;

      const vehicleIsWaiting =
        previous !== null &&
        previous.row === position.row &&
        previous.column === position.column &&
        (goal === null ||
          goal.row !== position.row ||
          goal.column !== position.column);

      waiting[vehicleId] = vehicleIsWaiting;
    }

    for (const [vehicleId, position] of Object.entries(currentState)) {
      previousPositions[vehicleId] = {
        row: position.row,
        column: position.column,
      };
    }

    return waiting;
  }

  // Grid Drawing
  function drawGrid(gridState, waitingStatus, timestamp) {
    const canvas = dom.gridCanvas;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const cellSize = width / 10;

    lastGridStateForRedraw = gridState;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#cccccc";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(width, i * cellSize);
      ctx.stroke();
    }

    // Sensor noise: 8 red cells around each vehicle (3x3 minus its own cell)
    drawSensorNoiseOverlay(gridState, cellSize);

    // Goal markers: vehicle's destination cell
    drawGoalMarkers(cellSize);

    for (const [vehicleId, position] of Object.entries(gridState)) {
      const row = position.row;
      const column = position.column;
      const x = column * cellSize;
      const y = row * cellSize;
      const isWaiting = waitingStatus[vehicleId] || false;

      if (isWaiting) {
        const phase = (timestamp / 500) % 2;
        const brightness = Math.abs(phase - 1);
        const alpha = 0.3 + brightness * 0.5;

        ctx.save();
        ctx.shadowColor = "rgba(255, 0, 0, 0.8)";
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(
          x + cellSize / 2,
          y + cellSize / 2,
          cellSize * 0.55,
          0,
          2 * Math.PI,
        );
        ctx.fillStyle = "rgba(255, 200, 0, " + alpha + ")";
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 200, 0, " + alpha * 0.7 + ")";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.restore();
      }

      ctx.shadowBlur = 4;
      ctx.shadowColor = "rgba(0,0,0,0.2)";
      ctx.fillStyle = VEHICLE_COLOURS[vehicleId] || "#888888";
      ctx.beginPath();
      ctx.arc(
        x + cellSize / 2,
        y + cellSize / 2,
        cellSize * 0.35,
        0,
        2 * Math.PI,
      );
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(vehicleId, x + cellSize / 2, y + cellSize / 2);
    }

    ctx.fillStyle = "#ff6f00";
    ctx.globalAlpha = 0.15;
    ctx.fillRect(4 * cellSize, 4 * cellSize, 2 * cellSize, 2 * cellSize);
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(4 * cellSize, 4 * cellSize, 2 * cellSize, 2 * cellSize);
  }

  // 3x3 minus its own cell while sensor noise is on
  function drawSensorNoiseOverlay(gridState, cellSize) {
    if (!dom.sensorNoiseToggle || !dom.sensorNoiseToggle.checked) return;

    const canvas = dom.gridCanvas;
    const ctx = canvas.getContext("2d");
    const gridSize = 10;

    for (const position of Object.values(gridState)) {
      const row = position.row;
      const column = position.column;

      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue; 
          const r = row + dr;
          const c = column + dc;
          if (r < 0 || r >= gridSize || c < 0 || c >= gridSize) continue;

          ctx.fillStyle = "rgba(255, 0, 0, 0.35)";
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          ctx.strokeStyle = "rgba(255, 75, 75, 0.6)";
          ctx.lineWidth = 1;
          ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
      }
    }
  }

  // vehicle's destination cell
  function drawGoalMarkers(cellSize) {
    const canvas = dom.gridCanvas;
    const ctx = canvas.getContext("2d");

    for (const [vehicleId, goal] of Object.entries(vehicleGoals)) {
      const x = goal.column * cellSize;
      const y = goal.row * cellSize;
      const colour = VEHICLE_COLOURS[vehicleId] || "#888888";

      ctx.save();

      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 3, y + 3, cellSize - 6, cellSize - 6);
      ctx.setLineDash([]);

      const poleX = x + cellSize - 9;
      const poleTopY = y + 4;
      const poleBottomY = y + cellSize - 4;

      ctx.strokeStyle = "#333333";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.stroke();

      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(poleX, poleTopY);
      ctx.lineTo(poleX, poleTopY + 6);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }
  }

  function clearGrid() {
    const canvas = dom.gridCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    lastGridStateForRedraw = null;
  }

  // Schedule Table
  function updateScheduleDisplay(schedules) {
    const container = dom.scheduleDisplay;

    if (!schedules || schedules.length === 0) {
      container.innerHTML =
        '<p style="color:#666;text-align:center;">No schedule data</p>';
      return;
    }

    let html =
      "<table><thead><tr>" +
      "<th>Vehicle</th><th>Time</th><th>Position (Row, Col)</th>" +
      "</tr></thead><tbody>";

    for (const schedule of schedules) {
      const vehicleId = schedule.vehicle_id;
      const steps = schedule.steps;
      const displaySteps = steps.slice(0, 15);

      for (const step of displaySteps) {
        html +=
          "<tr>" +
          "<td><strong>" +
          vehicleId +
          "</strong></td>" +
          "<td>" +
          step.time +
          "</td>" +
          "<td>(" +
          step.row +
          ", " +
          step.column +
          ")</td>" +
          "</tr>";
      }

      if (steps.length > 15) {
        html +=
          '<tr><td colspan="3" style="color:#888;font-style:italic;">' +
          "... (" +
          (steps.length - 15) +
          " more steps)" +
          "</td></tr>";
      }
    }

    html += "</tbody></table>";
    container.innerHTML = html;
  }

  // Event Listeners
  dom.runCbs.addEventListener("click", function () {
    runSimulation("/run_cbs");
  });

  dom.runHybrid.addEventListener("click", function () {
    runSimulation("/run_hybrid");
  });

  dom.generatePlots.addEventListener("click", function () {
    generateChart();
  });

  dom.sensorNoiseToggle.addEventListener("change", function () {
    dom.sensorNoiseStatus.textContent = dom.sensorNoiseToggle.checked
      ? "0.1 m (on)"
      : "0.0 m (off)";

    if (lastGridStateForRedraw) {
      drawGrid(lastGridStateForRedraw, {}, performance.now());
    }
  });

  // Initialize
  async function init() {
    await loadDatasets();

    clearGrid();
    dom.vehiclesReached.textContent = "0 / " + getParameters().num_vehicles;
    dom.timeStepsCount.textContent = "0";
    dom.sensorNoiseStatus.textContent = dom.sensorNoiseToggle.checked
      ? "0.1 m (on)"
      : "0.0 m (off)";

    const ctx = dom.collisionChart.getContext('2d');
    ctx.clearRect(0, 0, dom.collisionChart.width, dom.collisionChart.height);
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(0, 0, dom.collisionChart.width, dom.collisionChart.height);
    ctx.fillStyle = '#666';
    ctx.font = '16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Select metrics and chart type, then click "Generate Chart"', dom.collisionChart.width / 2, dom.collisionChart.height / 2 - 10);
    ctx.font = '14px Inter, sans-serif';
    ctx.fillStyle = '#999';
    ctx.fillText('Data loaded: baseline=' + (baselineData ? Object.keys(baselineData).length : 0) + ' configs, hybrid=' + (hybridData ? hybridData.length : 0) + ' configs', dom.collisionChart.width / 2, dom.collisionChart.height / 2 + 25);

    console.log("Autonomous Navigation System UI ready.");
  }

  init();

})();