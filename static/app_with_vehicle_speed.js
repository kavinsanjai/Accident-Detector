// Advanced Bike Visualization with Speed-Aware Detection

// Get all slider elements
const speedSlider = document.getElementById('speed');
const accXSlider = document.getElementById('acc-x');
const accYSlider = document.getElementById('acc-y');
const accZSlider = document.getElementById('acc-z');
const gyroXSlider = document.getElementById('gyro-x');
const gyroYSlider = document.getElementById('gyro-y');
const gyroZSlider = document.getElementById('gyro-z');

// Get all value display elements
const speedValue = document.getElementById('speed-value');
const accXValue = document.getElementById('acc-x-value');
const accYValue = document.getElementById('acc-y-value');
const accZValue = document.getElementById('acc-z-value');
const gyroXValue = document.getElementById('gyro-x-value');
const gyroYValue = document.getElementById('gyro-y-value');
const gyroZValue = document.getElementById('gyro-z-value');

// Get bike visualization elements
const vehicle = document.getElementById('vehicle');
const speedometer = document.getElementById('speedometer');
const speedometerDisplay = document.getElementById('speedometer-display');
const vehicleStatus = document.getElementById('vehicle-status');
const arrowX = document.getElementById('arrow-x');
const arrowY = document.getElementById('arrow-y');
const arrowZ = document.getElementById('arrow-z');
const rotationIndicator = document.getElementById('rotation');

// Get result elements
const detectionResult = document.getElementById('detection-result');
const confidenceDisplay = document.getElementById('confidence');
const resultSpeed = document.getElementById('result-speed');
const gForceDisplay = document.getElementById('g-force');
const rotationDisplay = document.getElementById('rotation-value');

// Preset scenarios data (will be loaded from API)
let presetScenarios = {};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkMLModelStatus();
    loadPresetScenarios();
    updateAllDisplays();
    attachEventListeners();
});

// Check if ML model is available
async function checkMLModelStatus() {
    try {
        const response = await fetch('/api/model_status');
        const status = await response.json();
        
        const mlStatusDiv = document.getElementById('ml-status');
        const mlRadio = document.querySelector('input[name="model"][value="ml"]');
        
        if (!status.ml_available) {
            // ML model not available
            if (mlStatusDiv) {
                mlStatusDiv.style.display = 'block';
                mlStatusDiv.innerHTML = '⚠️ ML model not loaded. Using rule-based system. <br><small>To train: python ml_accident_detector.py</small>';
            }
            if (mlRadio) {
                mlRadio.disabled = true;
                mlRadio.parentElement.style.opacity = '0.5';
                mlRadio.parentElement.style.cursor = 'not-allowed';
            }
        } else {
            // ML model is available
            if (mlStatusDiv) {
                mlStatusDiv.style.display = 'none';
            }
            console.log('✅ ML model is ready to use!');
        }
    } catch (error) {
        console.error('Error checking ML model status:', error);
    }
}

// Load preset scenarios from server
async function loadPresetScenarios() {
    try {
        const response = await fetch('/api/presets');
        presetScenarios = await response.json();
    } catch (error) {
        console.error('Error loading presets:', error);
    }
}

// Attach event listeners to all sliders
function attachEventListeners() {
    const sliders = [speedSlider, accXSlider, accYSlider, accZSlider, 
                     gyroXSlider, gyroYSlider, gyroZSlider];
    
    sliders.forEach(slider => {
        slider.addEventListener('input', updateAllDisplays);
    });
}

// Update all displays
function updateAllDisplays() {
    updateValueDisplays();
    updateVehicleVisualization();
    updateSpeedometer();
    updateVehicleStatus();
}

// Update numeric value displays
function updateValueDisplays() {
    speedValue.textContent = `${speedSlider.value} km/h`;
    accXValue.textContent = `${parseFloat(accXSlider.value).toFixed(1)} G`;
    accYValue.textContent = `${parseFloat(accYSlider.value).toFixed(1)} G`;
    accZValue.textContent = `${parseFloat(accZSlider.value).toFixed(1)} G`;
    gyroXValue.textContent = `${parseFloat(gyroXSlider.value).toFixed(1)} °/s`;
    gyroYValue.textContent = `${parseFloat(gyroYSlider.value).toFixed(1)} °/s`;
    gyroZValue.textContent = `${parseFloat(gyroZSlider.value).toFixed(1)} °/s`;
}

// Update speedometer display with speed-based styling
function updateSpeedometer() {
    const speed = parseFloat(speedSlider.value);
    speedometerDisplay.textContent = speed;
    
    // Remove all speed classes
    speedometer.classList.remove('speed-warning', 'speed-danger');
    
    // Add appropriate class based on speed
    if (speed > 80) {
        speedometer.classList.add('speed-danger');
    } else if (speed > 50) {
        speedometer.classList.add('speed-warning');
    }
}

// Update bike visualization based on forces
function updateVehicleVisualization() {
    const accX = parseFloat(accXSlider.value);
    const accY = parseFloat(accYSlider.value);
    const accZ = parseFloat(accZSlider.value);
    const gyroX = parseFloat(gyroXSlider.value);
    const gyroY = parseFloat(gyroYSlider.value);
    const gyroZ = parseFloat(gyroZSlider.value);
    
    // Calculate magnitudes
    const accMag = Math.sqrt(accX**2 + accY**2 + accZ**2);
    const gyroMag = Math.sqrt(gyroX**2 + gyroY**2 + gyroZ**2);
    
    // Apply transformations to bike
    let transform = '';
    
    // Tilt based on acceleration
    const tiltX = Math.max(-20, Math.min(20, accY * 2)); // Side tilt
    const tiltY = Math.max(-20, Math.min(20, -accX * 2)); // Forward/backward tilt
    
    // Rotation based on gyroscope
    const rotateZ = Math.max(-30, Math.min(30, gyroZ * 1));
    
    // Vertical displacement based on Z-axis
    const verticalDisplacement = (accZ - 9.8) * 2;
    
    transform = `
        translateY(${verticalDisplacement}px) 
        rotateX(${tiltY}deg) 
        rotateY(${tiltX}deg) 
        rotateZ(${rotateZ}deg)
    `;
    
    vehicle.style.transform = transform;
    
    // Show force arrows
    updateForceArrows(accX, accY, accZ);
    
    // Show rotation indicator
    if (gyroMag > 10) {
        rotationIndicator.classList.add('active');
        rotationIndicator.style.animationDuration = `${Math.max(0.3, 2 - gyroMag/20)}s`;
    } else {
        rotationIndicator.classList.remove('active');
    }
    
    // Add crash effect for severe impacts
    if (accMag > 20 || gyroMag > 30) {
        vehicle.classList.add('crash-effect');
        setTimeout(() => vehicle.classList.remove('crash-effect'), 500);
    }
}

// Update force arrows
function updateForceArrows(accX, accY, accZ) {
    const threshold = 5; // Minimum force to show arrow
    
    // X-axis arrow (horizontal)
    if (Math.abs(accX) > threshold) {
        arrowX.classList.add('active');
        const width = Math.min(100, Math.abs(accX) * 2);
        arrowX.style.width = `${width}px`;
        arrowX.style.left = accX > 0 ? '-50px' : `${150 + 50 - width}px`;
        arrowX.style.background = accX > 0 ? '#e74c3c' : '#3498db';
    } else {
        arrowX.classList.remove('active');
    }
    
    // Y-axis arrow (vertical from top)
    if (Math.abs(accY) > threshold) {
        arrowY.classList.add('active');
        const height = Math.min(100, Math.abs(accY) * 2);
        arrowY.style.height = `${height}px`;
        arrowY.style.background = accY > 0 ? '#2ecc71' : '#e67e22';
    } else {
        arrowY.classList.remove('active');
    }
    
    // Z-axis arrow (vertical from bottom)
    const zDeviation = Math.abs(accZ - 9.8);
    if (zDeviation > threshold) {
        arrowZ.classList.add('active');
        const height = Math.min(100, zDeviation * 2);
        arrowZ.style.height = `${height}px`;
        arrowZ.style.background = accZ > 9.8 ? '#9b59b6' : '#f39c12';
    } else {
        arrowZ.classList.remove('active');
    }
}

// Update bike status text
function updateVehicleStatus() {
    const speed = parseFloat(speedSlider.value);
    const accX = parseFloat(accXSlider.value);
    const accZ = parseFloat(accZSlider.value);
    const accMag = Math.sqrt(
        accX**2 + 
        parseFloat(accYSlider.value)**2 + 
        accZ**2
    );
    const gyroMag = Math.sqrt(
        parseFloat(gyroXSlider.value)**2 + 
        parseFloat(gyroYSlider.value)**2 + 
        parseFloat(gyroZSlider.value)**2
    );
    
    let status = '';
    
    // Determine status based on parameters
    if (accMag > 30 || gyroMag > 30) {
        status = '💥 SEVERE IMPACT DETECTED!';
    } else if (accMag > 20 || gyroMag > 20) {
        status = '⚠️ High Impact Forces';
    } else if (speed > 80 && accMag > 10) {
        status = '🚨 High-Speed Danger Zone';
    } else if (accX < -10 && speed > 30) {
        status = '🛑 Hard Braking';
    } else if (accX > 5 && speed < 20) {
        status = '🏎️ Accelerating';
    } else if (speed > 80) {
        status = '🏁 High-Speed Cruising';
    } else if (speed > 40) {
        status = '🚗 Normal Riding';
    } else if (speed > 5) {
        status = '🚙 City Riding';
    } else {
        status = '🅿️ Bike Stationary';
    }
    
    vehicleStatus.textContent = status;
}

// Load preset scenario
function loadPreset(presetName) {
    const preset = presetScenarios[presetName];
    if (!preset) {
        console.error('Preset not found:', presetName);
        return;
    }
    
    // Set all slider values
    speedSlider.value = preset.params.speed || 0;
    accXSlider.value = preset.params.acc_x;
    accYSlider.value = preset.params.acc_y;
    accZSlider.value = preset.params.acc_z;
    gyroXSlider.value = preset.params.gyro_x;
    gyroYSlider.value = preset.params.gyro_y;
    gyroZSlider.value = preset.params.gyro_z;
    
    // Update all displays
    updateAllDisplays();
    
    // Show notification
    vehicleStatus.textContent = `📋 Loaded: ${preset.name}`;
    setTimeout(() => updateVehicleStatus(), 2000);
}

// Detect accident
async function detectAccident() {
    // Show loading state
    detectionResult.textContent = '🔄 Analyzing...';
    detectionResult.className = 'detection-result';
    
    // Get selected model
    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    
    // Gather sensor data
    const sensorData = {
        speed: parseFloat(speedSlider.value),
        acc_x: parseFloat(accXSlider.value),
        acc_y: parseFloat(accYSlider.value),
        acc_z: parseFloat(accZSlider.value),
        gyro_x: parseFloat(gyroXSlider.value),
        gyro_y: parseFloat(gyroYSlider.value),
        gyro_z: parseFloat(gyroZSlider.value),
        model_type: selectedModel  // Add model selection
    };
    
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(sensorData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error detecting accident:', error);
        detectionResult.textContent = '❌ Error: ' + error.message;
        detectionResult.className = 'detection-result';
    }
}

// Display detection results with enhanced confidence display
function displayResults(result) {
    // Get confidence percentage
    const confidencePercent = result.confidence_percent || (result.confidence * 100);
    const severity = result.severity || 'UNKNOWN';
    const severityColor = result.severity_color || '#6B7280';
    const explanation = result.explanation || {};
    const modelUsed = result.model_used || 'Unknown';
    
    // Update detection result with severity badge and explanation
    if (result.is_accident) {
        detectionResult.innerHTML = `
            <div class="result-header-flex">
                <span class="result-icon">🚨</span>
                <span class="result-text">ACCIDENT DETECTED!</span>
                <span class="severity-badge" style="background: ${severityColor}">${severity}</span>
            </div>
            
            <!-- Simple Status -->
            <div class="explanation-section">
                <h3 style="color: ${severityColor}; margin-top: 15px;">${explanation.simple_status || 'Accident Detected'}</h3>
            </div>
            
            <!-- What Happened -->
            <div class="explanation-box what-happened">
                <h4>📖 What Happened?</h4>
                <p>${explanation.what_happened || result.reason}</p>
            </div>
            
            <!-- Why Detected -->
            <div class="explanation-box why-detected">
                <h4>🔍 Why Was This Detected?</h4>
                <p>${explanation.why_detected || 'Sensor readings indicated abnormal forces consistent with an accident scenario.'}</p>
            </div>
            
            <!-- Simple Analogy -->
            <div class="explanation-box analogy">
                <h4>💡 Easy to Understand:</h4>
                <p>${explanation.analogy || 'The forces detected were much higher than normal cycling conditions.'}</p>
            </div>
            
            <!-- Safety Tip -->
            <div class="explanation-box safety-tip" style="background: #FEF3C7; border-left: 4px solid #F59E0B;">
                <h4>⚠️ Safety Reminder:</h4>
                <p>${explanation.safety_tip || 'Always wear protective gear and stay alert while cycling.'}</p>
            </div>
            
            <!-- Model Used -->
            <div style="margin-top: 15px; padding: 10px; background: rgba(52, 152, 219, 0.1); border-radius: 5px; font-size: 13px;">
                <strong>🤖 Model Used:</strong> ${modelUsed}
            </div>
            
            <!-- Technical Details (collapsed by default) -->
            <details class="technical-details">
                <summary>🔧 Technical Details (Click to expand)</summary>
                <div class="result-reason">${result.reason}</div>
            </details>
        `;
        detectionResult.className = 'detection-result result-accident';
    } else {
        detectionResult.innerHTML = `
            <div class="result-header-flex">
                <span class="result-icon">✅</span>
                <span class="result-text">SAFE - No Accident Detected</span>
                <span class="severity-badge" style="background: ${severityColor}">${severity}</span>
            </div>
            
            <!-- Simple Status -->
            <div class="explanation-section">
                <h3 style="color: ${severityColor}; margin-top: 15px;">${explanation.simple_status || 'Safe Riding'}</h3>
            </div>
            
            <!-- What Happened -->
            <div class="explanation-box what-happened" style="background: #ECFDF5; border-left: 4px solid #22C55E;">
                <h4>📖 What's Happening?</h4>
                <p>${explanation.what_happened || 'All sensor readings are within normal, safe ranges.'}</p>
            </div>
            
            <!-- Why Safe -->
            <div class="explanation-box why-detected" style="background: #ECFDF5; border-left: 4px solid #22C55E;">
                <h4>✅ Why Is This Safe?</h4>
                <p>${explanation.why_detected || 'Forces and rotation are typical for normal cycling conditions.'}</p>
            </div>
            
            <!-- Simple Analogy -->
            <div class="explanation-box analogy">
                <h4>💡 Easy to Understand:</h4>
                <p>${explanation.analogy || 'Everything is smooth and controlled, like walking on a flat path.'}</p>
            </div>
            
            <!-- Safety Tip -->
            <div class="explanation-box safety-tip" style="background: #ECFDF5; border-left: 4px solid #22C55E;">
                <h4>✅ Keep It Up:</h4>
                <p>${explanation.safety_tip || 'You\'re riding safely! Maintain awareness and enjoy your ride.'}</p>
            </div>
            
            <!-- Model Used -->
            <div style="margin-top: 15px; padding: 10px; background: rgba(52, 152, 219, 0.1); border-radius: 5px; font-size: 13px;">
                <strong>🤖 Model Used:</strong> ${modelUsed}
            </div>
            
            <!-- Technical Details (collapsed by default) -->
            <details class="technical-details">
                <summary>🔧 Technical Details (Click to expand)</summary>
                <div class="result-reason">${result.reason}</div>
            </details>
        `;
        detectionResult.className = 'detection-result result-safe';
    }
    
    // Update metrics with improved confidence display
    confidenceDisplay.textContent = `${confidencePercent.toFixed(1)}%`;
    confidenceDisplay.style.color = severityColor;
    confidenceDisplay.style.fontWeight = 'bold';
    confidenceDisplay.style.fontSize = '1.5em';
    
    resultSpeed.textContent = `${result.metrics.speed.toFixed(1)} km/h`;
    gForceDisplay.textContent = `${result.metrics.acc_magnitude.toFixed(1)} G`;
    rotationDisplay.textContent = `${result.metrics.gyro_magnitude.toFixed(1)} °/s`;
    
    // Add visual indicator bars for metrics
    if (result.metrics.acc_magnitude > 20) {
        gForceDisplay.style.color = '#DC2626';
        gForceDisplay.style.fontWeight = 'bold';
    } else if (result.metrics.acc_magnitude > 10) {
        gForceDisplay.style.color = '#EA580C';
    } else {
        gForceDisplay.style.color = '#22C55E';
    }
    
    if (result.metrics.gyro_magnitude > 30) {
        rotationDisplay.style.color = '#DC2626';
        rotationDisplay.style.fontWeight = 'bold';
    } else if (result.metrics.gyro_magnitude > 15) {
        rotationDisplay.style.color = '#EA580C';
    } else {
        rotationDisplay.style.color = '#22C55E';
    }
}

// Check ML model availability
async function checkMLAvailability() {
    try {
        const response = await fetch('/api/model_status');
        const status = await response.json();
        
        const mlRadio = document.querySelector('input[name="model"][value="ml"]');
        const mlStatus = document.getElementById('ml-status');
        
        if (!status.ml_available) {
            mlRadio.disabled = true;
            mlStatus.style.display = 'block';
            mlStatus.innerHTML = '⚠️ ML model not loaded. Train it first using: <code>python ml_accident_detector.py</code>';
        } else {
            mlStatus.style.display = 'none';
        }
    } catch (error) {
        console.error('Error checking ML status:', error);
    }
}

// Initialize on load
window.onload = function() {
    updateAllDisplays();
    checkMLAvailability();
};
