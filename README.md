# Bike Accident Detection System 🚴‍♂️🛡️

A comprehensive machine learning system for detecting bike accidents using multimodal sensor data from accelerometer, gyroscope, magnetometer, and GPS sensors mounted on bicycle handlebars.

## 🎯 Features

- **Multimodal Sensor Fusion**: Combines data from accelerometer, gyroscope, magnetometer, and GPS
- **Advanced Feature Engineering**: Computes acceleration magnitude, jerk, angular velocity, tilt angles, and delta tilt
- **Multiple ML Models**: Random Forest, XGBoost, LSTM, and 1D CNN implementations
- **Real-time Detection**: Live accident detection with timestamp logging
- **Safety Filters**: Prevents false positives during stationary periods
- **Sliding Window Analysis**: Time-series analysis with configurable windows
- **Model Persistence**: Save/load trained models for deployment

## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Core Dependencies
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `xgboost >= 1.5.0`
- `tensorflow >= 2.8.0`
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0`

## 📁 Dataset Format

The system expects CSV files with the following structure:

### Accelerometer Data
```
timestamp;sensor_type;acc_x;acc_y;acc_z;magnitude;
```

### Gyroscope Data
```
timestamp;sensor_type;gyro_x;gyro_y;gyro_z;
```

### Magnetometer Data
```
timestamp;sensor_type;mag_x;mag_y;mag_z;
```

### GPS Data
```
sensor_type;latitude;longitude;speed;timestamp;;;
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from bike_accident_detection import BikeAccidentDetector

# Initialize detector
detector = BikeAccidentDetector(
    data_path="path/to/your/dataset",
    min_speed_threshold=5.0  # km/h
)

# Load and process data
detector.load_sensor_data()
detector.preprocess_data()
detector.engineer_features()
detector.create_sliding_windows()

# Add synthetic accidents for training
detector.create_synthetic_accidents(num_accidents=200)

# Train models
results = detector.train_models()

# Save models
detector.save_models()
```

### 2. Real-time Detection

```python
# Create real-time detector
detect_fn = detector.create_real_time_detector('xgboost')

# Example sensor reading
sensor_data = {
    'acc_x': 15.2, 'acc_y': 12.8, 'acc_z': 25.1,
    'gyro_x': 8.5, 'gyro_y': 6.2, 'gyro_z': 4.8,
    'mag_x': 15.2, 'mag_y': -8.5, 'mag_z': 12.1,
    'speed': 12.0,  # m/s
    'timestamp': int(datetime.now().timestamp() * 1000)
}

# Detect accident
is_accident, confidence, timestamp = detect_fn(sensor_data)
print(f"Accident: {is_accident}, Confidence: {confidence:.4f}")
```

### 3. Using Pre-trained Models

```python
from model_deployment import AccidentDetectionService

# Load pre-trained models
service = AccidentDetectionService("models/")

# Predict accident
is_accident, confidence, timestamp = service.predict_accident(
    sensor_data, model_name='xgboost'
)
```

## 🧪 Running the System

### Full Training Pipeline
```bash
python bike_accident_detection.py
```

### Quick Test
```bash
python quick_test.py
```

### Model Deployment Demo
```bash
python model_deployment.py
```

## 🏗️ Architecture

### Data Processing Pipeline
1. **Data Loading**: Load multimodal sensor data from CSV files
2. **Preprocessing**: Synchronize timestamps, handle missing values
3. **Feature Engineering**: Compute derived features (jerk, tilt angles, etc.)
4. **Windowing**: Create sliding time windows with statistical aggregations
5. **Labeling**: Generate accident labels with safety filters

### Machine Learning Models
1. **Random Forest**: Ensemble method with balanced class weights
2. **XGBoost**: Gradient boosting with custom scaling
3. **LSTM**: Recurrent neural network for sequential patterns
4. **1D CNN**: Convolutional network for temporal feature detection

### Real-time Detection
- **Safety Filter**: Ignores accidents when speed < threshold
- **Feature Extraction**: Real-time computation of sensor features  
- **Model Inference**: Configurable model selection for prediction
- **Timestamp Logging**: Precise accident time recording

## 📊 Feature Engineering

### Primary Features
- **Acceleration Magnitude**: `sqrt(acc_x² + acc_y² + acc_z²)`
- **Jerk**: Rate of change of acceleration
- **Angular Velocity Magnitude**: `sqrt(gyro_x² + gyro_y² + gyro_z²)`
- **Tilt Angles**: Pitch, roll, yaw using accelerometer and magnetometer
- **Delta Tilt**: Sudden orientation changes
- **Speed Features**: GPS speed, speed delta, acceleration

### Statistical Aggregations (per window)
- Mean, Maximum, Minimum
- Standard Deviation, Range
- Peak Count Detection

### Safety Features
- **Speed Threshold**: `min_speed_threshold = 5.0 km/h`
- **Stationary Filter**: Prevents false positives when not moving
- **Combined Scoring**: Multi-criteria accident probability

## 🎛️ Configuration

### Model Parameters
```python
detector = BikeAccidentDetector(
    data_path="your/data/path",
    min_speed_threshold=5.0,    # km/h - minimum speed for accident detection
)

# Window configuration
detector.window_size = 2.0      # seconds
detector.overlap = 0.5          # 50% overlap
detector.sampling_rate = 50     # Hz
```

### Model Training
```python
results = detector.train_models(
    test_size=0.2,              # 20% for testing
    verbose=True                # Progress output
)
```

## 📈 Model Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Balanced metric for imbalanced classes
- **Precision/Recall**: Accident detection performance
- **Confusion Matrix**: True/False positives and negatives

### Example Output
```
XGBOOST Results:
  Accuracy: 0.9245
  F1-Score: 0.8756
  Precision (Accident): 0.8421
  Recall (Accident): 0.9105
  Confusion Matrix: TN=850, FP=45, FN=32, TP=323
```

## 🔧 Customization

### Adding New Features
```python
def custom_feature_engineering(df):
    # Add your custom features
    df['custom_feature'] = your_computation(df)
    return df

# Extend the feature engineering pipeline
detector.engineer_features = custom_feature_engineering
```

### Custom Model Training
```python
# Add your own model
from sklearn.ensemble import GradientBoostingClassifier

custom_model = GradientBoostingClassifier()
custom_model.fit(X_train, y_train)
detector.models['custom'] = custom_model
```

## 📂 File Structure

```
bike_accident_detection/
├── bike_accident_detection.py     # Main detection system
├── model_deployment.py            # Production deployment utilities
├── quick_test.py                  # Quick testing script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── models/                        # Saved models directory
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    ├── lstm_model.h5
    ├── cnn_model.h5
    ├── scaler.pkl
    ├── feature_columns.pkl
    └── config.json
```

## 🚦 Safety Considerations

### Important Safety Notes
1. **This system is for research and development purposes**
2. **Not certified for production safety-critical applications**
3. **Always wear proper safety equipment when cycling**
4. **Regular model retraining recommended with new data**

### Safety Features Implemented
- Speed threshold filtering (no accident detection when stationary)
- Multi-criteria accident scoring
- Confidence thresholds for alerts
- Timestamp logging for incident analysis

## 🐛 Troubleshooting

### Common Issues

#### "No data loaded"
- Check data file paths
- Verify CSV file format and separators
- Ensure proper file permissions

#### "Model training failed"
- Check for sufficient training data
- Verify feature engineering completed successfully
- Consider reducing synthetic accident count for small datasets

#### "Memory errors"
- Reduce window size or overlap
- Process data in smaller batches
- Consider using a subset of features

### Performance Optimization
- Use fewer synthetic accidents for faster training
- Reduce the number of features for quicker inference
- Use traditional ML models (RF, XGBoost) for better performance on small datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- **Coordinate System**: Android sensor coordinate system (X: right, Y: up, Z: out)
- **Feature Engineering**: Based on HAR (Human Activity Recognition) literature
- **Machine Learning**: Ensemble methods and deep learning for time series classification

## 🙏 Acknowledgments

- Bike&Safe Dataset contributors
- Open-source machine learning community
- Sensor fusion research community

---

**⚠️ Disclaimer**: This system is for research and educational purposes. Always follow traffic laws and wear appropriate safety equipment while cycling.