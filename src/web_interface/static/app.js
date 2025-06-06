document.addEventListener('DOMContentLoaded', () => {
    const cameraFeed = document.getElementById('camera-feed');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const connectionIndicator = document.getElementById('connection-indicator');
    const connectionText = document.getElementById('connection-text');
    const frameSize = document.getElementById('frame-size');
    const fpsElement = document.getElementById('fps');

    const manualModeBtn = document.getElementById('manual-mode');
    const autonomousModeBtn = document.getElementById('autonomous-mode');
    const kartTypeBtn = document.getElementById('kart-type');
    const hunterTypeBtn = document.getElementById('hunter-type');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const currentMode = document.getElementById('current-mode');
    const currentCarType = document.getElementById('current-car-type');
    const runningStatus = document.getElementById('running-status');
    const alertBox = document.getElementById('alert');

    const canConnectionIndicator = document.getElementById('can-connection-indicator');
    const canConnectionText = document.getElementById('can-connection-text');
    const canMessageCount = document.getElementById('can-message-count');
    const canLog = document.getElementById('can-log');
    const hunterDataSection = document.getElementById('hunter-data');
    const kartDataSection = document.getElementById('kart-data');

    const hunterSpeed = document.getElementById('hunter-speed');
    const hunterSteering = document.getElementById('hunter-steering');
    const hunterBody = document.getElementById('hunter-body');
    const hunterControl = document.getElementById('hunter-control');
    const hunterBrake = document.getElementById('hunter-brake');
    const hunterMovementTimestamp = document.getElementById('hunter-movement-timestamp');
    const hunterStatusTimestamp = document.getElementById('hunter-status-timestamp');

    const kartSpeed = document.getElementById('kart-speed');
    const kartSteering = document.getElementById('kart-steering');
    const kartThrottle = document.getElementById('kart-throttle');
    const kartBraking = document.getElementById('kart-braking');
    const kartGear = document.getElementById('kart-gear');
    const kartBreakCurrent = document.getElementById('kart-break-current');
    const kartBreakTarget = document.getElementById('kart-break-target');
    const kartBreakDirection = document.getElementById('kart-break-direction');
    const kartBreakStatus = document.getElementById('kart-break-status');
    const kartMotionTimestamp = document.getElementById('kart-motion-timestamp');
    const kartControlsTimestamp = document.getElementById('kart-controls-timestamp');
    const kartBreakingTimestamp = document.getElementById('kart-breaking-timestamp');

    const frontViewBtn = document.getElementById('front-view');
    const leftViewBtn = document.getElementById('left-view');
    const rightViewBtn = document.getElementById('right-view');
    const topdownViewBtn = document.getElementById('topdown-view');
    const stitchedViewBtn = document.getElementById('stitched-view');
    const currentView = document.getElementById('current-view');
    const linesViewBtn = document.getElementById('lines-view');
    const objectsViewBtn = document.getElementById('objects-view');

    const loggerToggle = document.getElementById('logger-toggle');
    const loggerStatus = document.getElementById('logger-status');
    const loggerInfo = document.getElementById('logger-info');
    const logPath = document.getElementById('log-path');

    const state = {
        mode: null,
        carType: null,
        running: false,
        cameraConnected: false,
        canConnected: false,
        messageCount: 0,
        loggerEnabled: false,
        loggerAvailable: false,
        loggerPath: null
    };

    const highlightKeys = {
        hunter_movement: ['speed', 'steering'],
        hunter_status: ['body_status', 'control_mode', 'brake_status'],
        kart_steering: ['steering_raw'],
        kart_breaking: ['current_pot', 'target_pot', 'error'],
        kart_throttle: ['throttle_voltage', 'braking', 'gear'],
        kart_speed: ['speed'],
        hunter_movement_control: ['speed', 'steering'],
        hunter_control_mode: ['mode'],
        hunter_parking_control: ['engaged'],
        kart_steering_control: ['steering_angle'],
        kart_throttle_control: ['throttle', 'gear'],
        kart_break_control: ['brake_value']
    };

    let cameraSocket = null;
    let canSocket = null;

    let frameCount = 0;
    let lastFrameTime = Date.now();

    function connectCamera() {
        if (cameraSocket) {
            cameraSocket.close();
        }

        cameraPlaceholder.textContent = "Connecting to camera...";
        cameraPlaceholder.classList.remove('hidden');
        cameraFeed.classList.add('hidden');

        connectionIndicator.className = 'status-indicator disconnected';
        connectionText.textContent = 'Connecting...';

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/camera`;

        cameraSocket = new WebSocket(wsUrl);

        cameraSocket.onopen = () => {
            console.log('Camera WebSocket connected');
            state.cameraConnected = true;
            connectionIndicator.className = 'status-indicator connected';
            connectionText.textContent = 'Connected';

            startPingInterval(cameraSocket);
        };

        cameraSocket.onmessage = (event) => {
            try {
                if (event.data === 'pong') return;

                const data = JSON.parse(event.data);

                if (data.type === 'frame') {

                    cameraPlaceholder.classList.add('hidden');
                    cameraFeed.classList.remove('hidden');

                    cameraFeed.src = `data:image/jpeg;base64,${data.data}`;

                    if (data.view_mode) {
                        updateCameraViewButtons(data.view_mode);
                    }

                    frameCount++;
                    const now = Date.now();
                    if (now - lastFrameTime >= 1000) {
                        const frameRate = Math.round(frameCount * 1000 / (now - lastFrameTime));
                        fpsElement.textContent = frameRate;

                        frameCount = 0;
                        lastFrameTime = now;
                    }

                    if (!cameraFeed.naturalWidth) {
                        cameraFeed.onload = () => {
                            frameSize.textContent = `${cameraFeed.naturalWidth} x ${cameraFeed.naturalHeight}`;
                        };
                    } else {
                        frameSize.textContent = `${cameraFeed.naturalWidth} x ${cameraFeed.naturalHeight}`;
                    }
                }
                else if (data.type === 'view_changed') {
                    updateCameraViewButtons(data.view_mode);
                    showAlert(`Camera view changed to ${data.view_mode}`, true);
                }
            } catch (error) {
                console.error('Error processing camera message:', error);
            }
        };

        cameraSocket.onclose = () => {
            console.log('Camera WebSocket disconnected');
            state.cameraConnected = false;
            connectionIndicator.className = 'status-indicator disconnected';
            connectionText.textContent = 'Disconnected';

            cameraPlaceholder.textContent = "Camera disconnected. Attempting to reconnect...";
            cameraPlaceholder.classList.remove('hidden');
            cameraFeed.classList.add('hidden');

            setTimeout(connectCamera, 5000);
        };

        cameraSocket.onerror = (error) => {
            console.error('Camera WebSocket error:', error);
            connectionIndicator.className = 'status-indicator disconnected';
            connectionText.textContent = 'Connection error';
        };
    }

    function connectCAN() {
        if (canSocket) {
            canSocket.close();
        }

        canConnectionIndicator.className = 'status-indicator disconnected';
        canConnectionText.textContent = 'Connecting...';

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/can`;
        
        console.log(`Connecting to CAN WebSocket at ${wsUrl}`);
        canSocket = new WebSocket(wsUrl);

        canSocket.onopen = () => {
            console.log('CAN WebSocket connected successfully!');
            state.canConnected = true;
            canConnectionIndicator.className = 'status-indicator connected';
            canConnectionText.textContent = 'Connected';

            startPingInterval(canSocket);

            canSocket.send("ping");
            console.log("Sent initial ping to server");
        };
        
        canSocket.onmessage = (event) => {
            try {
                if (event.data === 'pong') {
                    return;
                }
                
                const message = JSON.parse(event.data);
                
                processCAN(message);

                state.messageCount++;
                canMessageCount.textContent = state.messageCount;
            } catch (error) {
                console.error('Error processing CAN message:', error);
                console.error('Raw data:', event.data);
            }
        };
        
        canSocket.onclose = (event) => {
            console.log('CAN WebSocket disconnected', event);
            state.canConnected = false;
            canConnectionIndicator.className = 'status-indicator disconnected';
            canConnectionText.textContent = 'Disconnected';

            setTimeout(connectCAN, 5000);
        };
        
        canSocket.onerror = (error) => {
            console.error('CAN WebSocket error:', error);
            canConnectionIndicator.className = 'status-indicator disconnected';
            canConnectionText.textContent = 'Connection error';
        };
    }

    function processCAN(message) {
        console.log("Processing CAN message:", message);
        
        try {
            const { timestamp, type, id, data } = JSON.parse(message);

            const date = new Date(timestamp);
            const formattedTime = `Last update: ${date.toLocaleTimeString()}`;

            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="timestamp">${timestamp}</span> <span class="id">${id}</span> <span class="type">${type}</span>`;
            
            const keys = highlightKeys[type] || [];
            
            if (Object.keys(data).length > 0) {
                logEntry.innerHTML += ' | ';
                logEntry.innerHTML += Object.entries(data)
                    .map(([key, value]) => {
                        const highlighted = keys.includes(key) ? 'highlighted' : '';
                        return `<span class="key">${key}:</span> <span class="${highlighted}">${value}</span>`;
                    })
                    .join(' | ');
            }
            
            canLog.appendChild(logEntry);

            while (canLog.childElementCount > 100) {
                canLog.removeChild(canLog.firstChild);
            }

            canLog.scrollTop = canLog.scrollHeight;

            switch (type) {
                case 'hunter_movement':
                    hunterSpeed.textContent = data.speed;
                    hunterSteering.textContent = data.steering;
                    hunterMovementTimestamp.textContent = formattedTime;
                    break;
                case 'hunter_status':
                    hunterBody.textContent = data.body_status;
                    hunterControl.textContent = data.control_mode;
                    hunterBrake.textContent = data.brake_status;
                    hunterStatusTimestamp.textContent = formattedTime;
                    break;
                case 'kart_speed':
                    kartSpeed.textContent = data.speed;
                    kartMotionTimestamp.textContent = formattedTime;
                    break;
                case 'kart_steering':
                    kartSteering.textContent = data.steering_raw;
                    kartMotionTimestamp.textContent = formattedTime;
                    break;
                case 'kart_throttle':
                    kartThrottle.textContent = data.throttle_voltage;
                    kartBraking.textContent = data.braking;
                    kartGear.textContent = data.gear;
                    kartControlsTimestamp.textContent = formattedTime;
                    break;
                case 'kart_breaking':
                    kartBreakCurrent.textContent = data.current_pot;
                    kartBreakTarget.textContent = data.target_pot;
                    kartBreakDirection.textContent = data.direction;
                    kartBreakStatus.textContent = data.error;
                    kartBreakingTimestamp.textContent = formattedTime;
                    break;

                case 'hunter_movement_control':
                    updateControlCard('hunter-movement-control', 'Hunter Movement Command', {
                        'Command Speed': data.speed !== undefined ? data.speed.toFixed(2) + ' m/s' : 'N/A',
                        'Command Steering': data.steering !== undefined ? data.steering.toFixed(2) + ' rad' : 'N/A'
                    });
                    break;
                    
                case 'hunter_control_mode':
                    updateControlCard('hunter-control-mode', 'Hunter Control Mode', {
                        'Mode': data.mode || 'Unknown'
                    });
                    break;
                    
                case 'hunter_parking_control':
                    updateControlCard('hunter-parking-control', 'Hunter Parking', {
                        'Parking': data.engaged ? 'Engaged' : 'Disengaged'
                    });
                    break;
                    
                case 'kart_steering_control':
                    updateControlCard('kart-steering-control', 'Kart Steering Command', {
                        'Command Angle': data.steering_angle !== undefined ? data.steering_angle.toFixed(2) : 'N/A'
                    });
                    break;
                    
                case 'kart_throttle_control':
                    updateControlCard('kart-throttle-control', 'Kart Throttle Command', {
                        'Command Throttle': data.throttle !== undefined ? data.throttle : 'N/A',
                        'Gear': data.gear || 'N/A'
                    });
                    break;
                    
                case 'kart_break_control':
                    updateControlCard('kart-break-control', 'Kart Brake Command', {
                        'Command Brake': data.brake_value !== undefined ? data.brake_value : 'N/A'
                    });
                    break;
            }
            
            state.messageCount++;
            canMessageCount.textContent = state.messageCount;
            
        } catch (error) {
            console.error('Error processing CAN message:', error);
        }
    }

    function updateControlCard(id, title, values) {
        let card = document.getElementById(id);
        
        if (!card) {
            const controlGrid = document.querySelector('#control-messages .can-data-grid');

            if (!controlGrid) {
                console.error("Control messages container not found! Element with ID 'control-messages' or child '.can-data-grid' is missing.");
                return;
            }
            
            card = document.createElement('div');
            card.id = id;
            card.className = 'can-data-card';
            controlGrid.appendChild(card);
        }
        
        let cardContent = `<div class="can-data-title">${title}</div>`;
        
        for (const [label, value] of Object.entries(values)) {
            cardContent += `<div class="can-data-value">${label}: <span class="value-highlight">${value}</span></div>`;
        }
        
        card.innerHTML = cardContent;
    }

    function startPingInterval(socket) {
        const pingInterval = setInterval(() => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send('ping');
            } else {
                clearInterval(pingInterval);
            }
        }, 5000);
    }

    function showAlert(message, isSuccess = false) {
        alertBox.textContent = message;
        alertBox.className = isSuccess ? 'alert success' : 'alert';

        setTimeout(() => {
            alertBox.className = 'alert hidden';
        }, 3000);
    }

    function updateButtonStates() {

        manualModeBtn.className = state.mode === 'manual' ? 'btn active' : 'btn';
        autonomousModeBtn.className = state.mode === 'autonomous' ? 'btn active' : 'btn';

        kartTypeBtn.className = state.carType === 'kart' ? 'btn active' : 'btn';
        hunterTypeBtn.className = state.carType === 'hunter' ? 'btn active' : 'btn';

        startBtn.disabled = state.running || !state.mode || !state.carType;
        startBtn.className = state.running || !state.mode || !state.carType ? 'btn btn-success btn-disabled' : 'btn btn-success';
        stopBtn.disabled = !state.running;
        stopBtn.className = !state.running ? 'btn btn-danger btn-disabled' : 'btn btn-danger';

        currentMode.textContent = state.mode ? state.mode.charAt(0).toUpperCase() + state.mode.slice(1) : 'Not set';
        currentCarType.textContent = state.carType ? state.carType.charAt(0).toUpperCase() + state.carType.slice(1) : 'Not set';
        runningStatus.textContent = state.running ? 'Yes' : 'No';

        if (state.carType === 'hunter') {
            hunterDataSection.classList.remove('hidden');
            kartDataSection.classList.add('hidden');
        } else if (state.carType === 'kart') {
            kartDataSection.classList.remove('hidden');
            hunterDataSection.classList.add('hidden');
        } else {
            kartDataSection.classList.add('hidden');
            hunterDataSection.classList.add('hidden');
        }
    }

    async function apiRequest(endpoint, data, forceMethod = null) {
        try {

            const postEndpoints = ['start', 'stop', 'mode', 'car-type'];
            const getEndpoints = ['status'];
            

            let method;
            if (forceMethod) {
                method = forceMethod;
            } else if (postEndpoints.includes(endpoint)) {
                method = 'POST';
            } else if (getEndpoints.includes(endpoint)) {
                method = 'GET';
            } else {
                method = 'POST';
            }
            
            console.log(`Sending ${method} request to /api/${endpoint}`);
            
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            if (method === 'POST') {
                options.body = JSON.stringify(data || {});
            }
            
            const response = await fetch(`/api/${endpoint}`, options);
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            showAlert(`Error: ${error.message}`);
            return null;
        }
    }

    async function getStatus() {
        const response = await apiRequest('status');
        if (response && response.success) {
            state.mode = response.status.mode;
            state.carType = response.status.car_type;
            state.running = response.status.running;

            const loggerResponse = await apiRequest('logger/status');
            if (loggerResponse && loggerResponse.success) {
                state.loggerAvailable = true;
                state.loggerEnabled = loggerResponse.status.enabled;
                state.loggerPath = loggerResponse.status.log_dir;
                updateLoggerUI();
            } else {
                state.loggerAvailable = false;
                updateLoggerUI();
            }

            updateButtonStates();
        }
    }

    function updateLoggerUI() {
        if (!state.loggerAvailable) {
            loggerToggle.disabled = true;
            loggerStatus.textContent = 'Not Available';
            loggerInfo.classList.add('hidden');
            return;
        }

        loggerToggle.disabled = false;
        loggerToggle.checked = state.loggerEnabled;
        loggerStatus.textContent = state.loggerEnabled ? 'On' : 'Off';
        
        if (state.loggerEnabled && state.loggerPath) {
            loggerInfo.classList.remove('hidden');
            logPath.textContent = state.loggerPath;
        } else {
            loggerInfo.classList.add('hidden');
        }
    }

    async function toggleLogger() {
        const enabled = loggerToggle.checked;
        const endpoint = enabled ? 'logger/start' : 'logger/stop';
        
        loggerToggle.disabled = true;
        loggerStatus.textContent = 'Updating...';
        
        const response = await apiRequest(endpoint);
        
        if (response && response.success) {
            state.loggerEnabled = response.status.enabled;
            state.loggerPath = response.status.log_dir;
            showAlert(response.message, true);
        } else {
            loggerToggle.checked = state.loggerEnabled;
            showAlert(`Logger operation failed: ${response ? response.message : 'Unknown error'}`, false);
        }
        
        updateLoggerUI();
    }

    manualModeBtn.addEventListener('click', async () => {
        if (state.mode === 'manual') return;

        const response = await apiRequest('mode', { mode: 'manual' });
        if (response && response.success) {
            state.mode = 'manual';
            state.running = response.status.running;
            updateButtonStates();
            showAlert('Manual mode selected', true);
        }
    });

    autonomousModeBtn.addEventListener('click', async () => {
        if (state.mode === 'autonomous') return;

        const response = await apiRequest('mode', { mode: 'autonomous' });
        if (response && response.success) {
            state.mode = 'autonomous';
            state.running = response.status.running;
            updateButtonStates();
            showAlert('Autonomous mode selected', true);
        }
    });

    kartTypeBtn.addEventListener('click', async () => {
        if (state.carType === 'kart') return;

        const response = await apiRequest('car-type', { car_type: 'kart' });
        if (response && response.success) {
            state.carType = 'kart';
            state.running = response.status.running;
            updateButtonStates();
            showAlert('Kart selected', true);
        }
    });

    hunterTypeBtn.addEventListener('click', async () => {
        if (state.carType === 'hunter') return;

        const response = await apiRequest('car-type', { car_type: 'hunter' });
        if (response && response.success) {
            state.carType = 'hunter';
            state.running = response.status.running;
            updateButtonStates();
            showAlert('Hunter selected', true);
        }
    });

    startBtn.addEventListener('click', async () => {
        if (state.running || !state.mode || !state.carType) return;

        const response = await apiRequest('start', {
            mode: state.mode,
            car_type: state.carType
        }, 'POST');
        
        if (response && response.success) {
            state.running = true;
            updateButtonStates();
            showAlert('Controller started!', true);
        } else {
            showAlert(`Failed to start: ${response?.message || 'Unknown error'}`, false);
        }
    });

    stopBtn.addEventListener('click', async () => {
        if (!state.running) return;

        const response = await apiRequest('stop', null, 'POST');

        console.log("Stop response:", response);
        
        if (response && response.success) {
            state.running = false;
            updateButtonStates();
            showAlert('Controller stopped', true);
        } else {
            console.error("Failed to stop controller properly:", response);
            state.running = false;
            updateButtonStates();
            showAlert('Controller may not have stopped properly', false);
        }
    });

    [frontViewBtn, leftViewBtn, rightViewBtn, topdownViewBtn, stitchedViewBtn, linesViewBtn, objectsViewBtn].forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.id.replace('-view', '');
            if (cameraSocket && cameraSocket.readyState === WebSocket.OPEN) {
                cameraSocket.send(`view:${view}`);
            }
        });
    });

    loggerToggle.addEventListener('change', async () => {
        await toggleLogger();
    });

    function updateCameraViewButtons(activeView) {

        const viewButtons = [frontViewBtn, leftViewBtn, rightViewBtn, topdownViewBtn, stitchedViewBtn, linesViewBtn, objectsViewBtn];
        viewButtons.forEach(btn => btn.classList.remove('active'));

        let viewName = 'Unknown';
        switch (activeView) {
            case 'front':
                frontViewBtn.classList.add('active');
                viewName = 'Front';
                break;
            case 'left':
                leftViewBtn.classList.add('active');
                viewName = 'Left';
                break;
            case 'right':
                rightViewBtn.classList.add('active');
                viewName = 'Right';
                break;
            case 'topdown':
                topdownViewBtn.classList.add('active');
                viewName = 'Top-Down';
                break;
            case 'stitched':
                stitchedViewBtn.classList.add('active');
                viewName = 'Stitched';
                break;
            case 'lines':
                linesViewBtn.classList.add('active');
                viewName = 'Lines';
                break;
            case 'objects':
                objectsViewBtn.classList.add('active');
                viewName = 'Objects';
                break;
        }
        
        currentView.textContent = viewName;
    }

    async function init() {
        connectCamera();
        connectCAN();

        await getStatus();

        updateButtonStates();
    }

    init();
});