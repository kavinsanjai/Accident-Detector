# 🚀 START ACCIDENT DETECTION SIMULATOR
# =====================================

Write-Host "=" -NoNewline; Write-Host "=" * 69
Write-Host "🚴 ACCIDENT DETECTION SIMULATOR - STARTUP" -ForegroundColor Cyan
Write-Host "=" -NoNewline; Write-Host "=" * 69

# Check if Python is available
Write-Host "`n📋 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.7+" -ForegroundColor Red
    pause
    exit 1
}

# Check if requirements are installed
Write-Host "`n📦 Checking dependencies..." -ForegroundColor Yellow
$packages = @("flask", "flask-cors", "numpy", "pandas")
$missingPackages = @()

foreach ($package in $packages) {
    $installed = python -c "import $($package.Replace('-', '_'))" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "⚠️ Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "`n📥 Installing required packages..." -ForegroundColor Cyan
    
    python -m pip install flask flask-cors numpy pandas joblib
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies!" -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "✅ All dependencies are installed!" -ForegroundColor Green
}

# Check if model exists
Write-Host "`n🤖 Checking model files..." -ForegroundColor Yellow
if (Test-Path "working_models\accident_detection_rules.pkl") {
    Write-Host "✅ Model found!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Model not found. Training model..." -ForegroundColor Yellow
    python working_accident_system.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to train model!" -ForegroundColor Red
        pause
        exit 1
    }
}

# Start the server
Write-Host "`n🚀 Starting web server..." -ForegroundColor Cyan
Write-Host "=" -NoNewline; Write-Host "=" * 69
Write-Host ""
Write-Host "🌐 Server will start at: http://localhost:5000" -ForegroundColor Green
Write-Host "📱 Open this URL in your browser to access the simulator" -ForegroundColor Green
Write-Host ""
Write-Host "⏹️ Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" -NoNewline; Write-Host "=" * 69
Write-Host ""

# Launch browser after a delay
Start-Sleep -Seconds 2
Start-Process "http://localhost:5000"

# Run the Flask app
python app.py
