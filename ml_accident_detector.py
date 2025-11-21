

import numpy as np
import pandas as pd
import joblib                                                   
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLAccidentDetector:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 
                              'acc_magnitude', 'gyro_magnitude', 'speed']
        print("🤖 ML BIKE ACCIDENT DETECTOR - RANDOM FOREST")
        print("=" * 60)
        print("Machine Learning approach using supervised classification")
        print("Algorithm: Random Forest Classifier")
    
    def create_features(self, df):
        """
        Create features from raw sensor data.
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Original sensor values
        if 'acc_x' in df.columns:
            features['acc_x'] = df['acc_x']
            features['acc_y'] = df['acc_y']
            features['acc_z'] = df['acc_z']
        
        if 'gyro_x' in df.columns:
            features['gyro_x'] = df['gyro_x']
            features['gyro_y'] = df['gyro_y']
            features['gyro_z'] = df['gyro_z']
        
        # Calculate magnitudes
        if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            features['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        
        if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            features['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        
        # Speed (if available, otherwise use 0)
        features['speed'] = df.get('speed', 0)
        
        return features
    
    def load_dataset(self, dataset_path):
        """
        Load and process the Bike&Safe Dataset.
        
        Args:
            dataset_path: Path to the Bike&Safe Dataset folder
            
        Returns:
            X: Features (sensor data)
            y: Labels (0=normal, 1=accident)
        """
        print("\n📂 Loading Bike&Safe Dataset...")
        
        all_data = []
        routes = ['First route', 'Second route', 'Third route']
        laps = ['First lap', 'Second lap', 'Third lap']
        
        for route in routes:
            for lap in laps:
                lap_path = os.path.join(dataset_path, route, lap)
                
                try:
                    # Load accelerometer data
                    acc_files = [f for f in os.listdir(lap_path) if 'accelerometer' in f.lower()]
                    gyro_files = [f for f in os.listdir(lap_path) if 'gyroscope' in f.lower()]
                    
                    if acc_files and gyro_files:
                        acc_df = pd.read_csv(os.path.join(lap_path, acc_files[0]))
                        gyro_df = pd.read_csv(os.path.join(lap_path, gyro_files[0]))
                        
                        # Merge accelerometer and gyroscope data
                        # Assuming both have timestamp columns
                        if len(acc_df) > 0 and len(gyro_df) > 0:
                            # Take minimum length to align data
                            min_len = min(len(acc_df), len(gyro_df))
                            
                            combined_df = pd.DataFrame({
                                'acc_x': acc_df.iloc[:min_len, 1].values,
                                'acc_y': acc_df.iloc[:min_len, 2].values,
                                'acc_z': acc_df.iloc[:min_len, 3].values,
                                'gyro_x': gyro_df.iloc[:min_len, 1].values,
                                'gyro_y': gyro_df.iloc[:min_len, 2].values,
                                'gyro_z': gyro_df.iloc[:min_len, 3].values,
                                'speed': 0  # Speed not available in dataset, default to 0
                            })
                            
                            all_data.append(combined_df)
                            print(f"✓ Loaded: {route}/{lap} - {len(combined_df)} samples")
                
                except Exception as e:
                    print(f"⚠ Skipped {route}/{lap}: {e}")
        
        if not all_data:
            raise ValueError("No data loaded! Check dataset path.")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total samples loaded: {len(combined_data)}")
        
        # Create features
        X = self.create_features(combined_data)
        
        # Create labels using physics-based rules
        # Since we don't have actual accident labels, we'll create synthetic labels
        # based on extreme sensor values (this is for training demonstration)
        y = self.create_synthetic_labels(X)
        
        print(f"📊 Dataset Statistics:")
        print(f"   - Normal riding: {(y == 0).sum()} samples ({(y == 0).sum()/len(y)*100:.1f}%)")
        print(f"   - Accidents: {(y == 1).sum()} samples ({(y == 1).sum()/len(y)*100:.1f}%)")
        
        return X, y
    
    def create_synthetic_labels(self, X):
        """
        Create sensitive accident labels that match rule-based system behavior.
        Uses SAME thresholds as WorkingAccidentDetector for consistency.
        """
        y = np.zeros(len(X))
        
        # These thresholds match the rule-based system exactly
        # so ML learns the same patterns
        
        # SEVERE ACCIDENTS (High Confidence >= 85%)
        severe = (
            (X['acc_x'].abs() > 25) |  # Extreme forward/back force
            (X['acc_y'].abs() > 22) |  # Extreme lateral force
            (X['acc_z'].abs() > 28) |  # Extreme vertical force
            (X['gyro_x'].abs() > 8) |  # Extreme roll
            (X['gyro_y'].abs() > 8) |  # Extreme pitch
            (X['acc_magnitude'] > 30) | # Extreme total acceleration
            (X['gyro_magnitude'] > 10)  # Extreme total rotation
        )
        
        # DANGEROUS SITUATIONS (Medium Confidence 70-84%)
        dangerous = (
            (X['acc_x'].abs() > 18) |  # Heavy braking/acceleration
            (X['acc_y'].abs() > 15) |  # Hard turn or side impact
            (X['acc_z'].abs() > 20) |  # Lifting or dropping
            (X['gyro_x'].abs() > 5) |  # Strong roll
            (X['gyro_y'].abs() > 5) |  # Strong pitch
            (X['acc_magnitude'] > 20) | # High acceleration
            (X['gyro_magnitude'] > 6)   # High rotation
        )
        
        # MODERATE CONCERN (Low-Medium Confidence 50-69%)
        moderate = (
            (X['acc_x'].abs() > 12) |  # Moderate braking
            (X['acc_y'].abs() > 10) |  # Moderate turn
            (X['acc_z'].abs() > 15) |  # Moderate vertical
            (X['gyro_x'].abs() > 3) |  # Moderate roll
            (X['gyro_y'].abs() > 3) |  # Moderate pitch
            (X['acc_magnitude'] > 15) | # Moderate acceleration
            (X['gyro_magnitude'] > 4)   # Moderate rotation
        )
        
        # SPEED-RELATED RISKS
        # High speed makes moderate forces more dangerous
        speed_amplified = (
            ((X['speed'] > 60) & (X['acc_magnitude'] > 12)) |
            ((X['speed'] > 50) & (X['gyro_magnitude'] > 3)) |
            ((X['speed'] > 40) & (X['acc_x'] < -10))  # Hard braking at speed
        )
        
        # COMBINED FORCES (rotation + acceleration = loss of control)
        combined = (
            (X['acc_magnitude'] > 12) & (X['gyro_magnitude'] > 3)
        )
        
        # LOW-SPEED FALLS (stationary or slow speed but high forces)
        low_speed_fall = (
            (X['speed'] < 10) & 
            ((X['gyro_magnitude'] > 5) | (X['acc_magnitude'] > 15))
        )
        
        # Mark as accident if ANY condition is met
        y[severe | dangerous | moderate | speed_amplified | combined | low_speed_fall] = 1
        
        return y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Random Forest model on sensor data.
        
        Args:
            X: Features (sensor readings)
            y: Labels (0=normal, 1=accident)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Training results with accuracy, confusion matrix, etc.
        """
        print("\n🎓 Training Random Forest Model...")
        print("=" * 60)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"📊 Data Split:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print("\n⏳ Training in progress...")
        self.model = RandomForestClassifier(
            n_estimators=150,        # Good balance of trees
            max_depth=25,            # Deeper to capture subtle patterns
            min_samples_split=5,     # More sensitive to patterns
            min_samples_leaf=2,      # Can create finer distinctions
            max_features='sqrt',     # Use sqrt of features at each split
            random_state=random_state,
            class_weight='balanced', # Handle imbalanced data
            n_jobs=-1                # Use all CPU cores
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("✅ Training complete!")
        print("\n📈 Model Evaluation:")
        print("=" * 60)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"🎯 Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"🎯 Testing Accuracy: {test_accuracy * 100:.2f}%")
        
        # Confusion Matrix
        print("\n📊 Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        print("\n   [[True Negatives  False Positives]")
        print("    [False Negatives True Positives]]")
        
        # Classification Report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Normal', 'Accident']))
        
        # Feature Importance
        print("\n🔍 Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def save_model(self, filepath='ml_accident_model.pkl'):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")
    
    def load_model(self, filepath='ml_accident_model.pkl'):
        """Load a trained model and scaler."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"✅ Model loaded from: {filepath}")
    
    def predict(self, sensor_data):
        """
        Predict if an accident occurred based on sensor data.
        
        Args:
            sensor_data: dict with keys acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed
            
        Returns:
            tuple: (is_accident: bool, confidence: float, reason: str)
        """
        if self.model is None:
            raise ValueError("No model loaded! Train or load a model first.")
        
        # Calculate magnitudes
        acc_magnitude = np.sqrt(sensor_data['acc_x']**2 + 
                               sensor_data['acc_y']**2 + 
                               sensor_data['acc_z']**2)
        gyro_magnitude = np.sqrt(sensor_data['gyro_x']**2 + 
                                sensor_data['gyro_y']**2 + 
                                sensor_data['gyro_z']**2)
        
        # Create feature vector
        features = np.array([[
            sensor_data['acc_x'],
            sensor_data['acc_y'],
            sensor_data['acc_z'],
            sensor_data['gyro_x'],
            sensor_data['gyro_y'],
            sensor_data['gyro_z'],
            acc_magnitude,
            gyro_magnitude,
            sensor_data.get('speed', 0)
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0][1]  # Probability of accident
        
        # Generate reason
        if prediction == 1:
            reason = f"ML Model detected accident pattern (confidence: {confidence*100:.1f}%)"
        else:
            reason = f"Normal riding detected (confidence: {(1-confidence)*100:.1f}%)"
        
        return bool(prediction), float(confidence), reason


def main():
    """Train and test the ML accident detector."""
    
    # Initialize detector
    detector = MLAccidentDetector()
    
    # Set dataset path
    dataset_path = r"Bike&Safe Dataset\Bike&Safe Dataset\Bike&Safe Dataset"
    
    try:
        # Load dataset
        X, y = detector.load_dataset(dataset_path)
        
        # Train model
        results = detector.train(X, y)
        
        # Save model
        detector.save_model('ml_accident_model.pkl')
        
        # Test with sample data
        print("\n" + "=" * 60)
        print("🧪 Testing Model with Sample Scenarios")
        print("=" * 60)
        
        test_scenarios = [
            {
                'name': 'Normal City Riding',
                'data': {'acc_x': 0.5, 'acc_y': 0.3, 'acc_z': 9.8, 
                        'gyro_x': 0.1, 'gyro_y': 0.2, 'gyro_z': 0.1, 'speed': 25}
            },
            {
                'name': 'Normal Highway Riding',
                'data': {'acc_x': 1.0, 'acc_y': 0.5, 'acc_z': 10.2, 
                        'gyro_x': 0.3, 'gyro_y': 0.4, 'gyro_z': 0.2, 'speed': 70}
            },
            {
                'name': 'Gentle Braking',
                'data': {'acc_x': -3.0, 'acc_y': 0.2, 'acc_z': 9.5, 
                        'gyro_x': 0.5, 'gyro_y': 0.3, 'gyro_z': 0.1, 'speed': 20}
            },
            {
                'name': 'Severe Frontal Crash',
                'data': {'acc_x': 32, 'acc_y': 12, 'acc_z': 38, 
                        'gyro_x': 18, 'gyro_y': 22, 'gyro_z': 15, 'speed': 55}
            },
            {
                'name': 'Hard Emergency Braking (Endo)',
                'data': {'acc_x': -22, 'acc_y': 2, 'acc_z': 25, 
                        'gyro_x': 3, 'gyro_y': 15, 'gyro_z': 5, 'speed': 45}
            },
            {
                'name': 'Side T-Bone Impact',
                'data': {'acc_x': 5, 'acc_y': 35, 'acc_z': 18, 
                        'gyro_x': 8, 'gyro_y': 6, 'gyro_z': 28, 'speed': 35}
            },
            {
                'name': 'Bike Tumbling/Rolling',
                'data': {'acc_x': 15, 'acc_y': 18, 'acc_z': 5, 
                        'gyro_x': 25, 'gyro_y': 30, 'gyro_z': 20, 'speed': 30}
            },
            {
                'name': 'High Speed Loss of Control',
                'data': {'acc_x': 8, 'acc_y': 12, 'acc_z': 15, 
                        'gyro_x': 12, 'gyro_y': 8, 'gyro_z': 18, 'speed': 85}
            },
            {
                'name': 'Stationary Impact (Parked)',
                'data': {'acc_x': 2, 'acc_y': 8, 'acc_z': 28, 
                        'gyro_x': 1, 'gyro_y': 2, 'gyro_z': 3, 'speed': 0}
            },
            {
                'name': 'Minor Bump (Safe)',
                'data': {'acc_x': -2, 'acc_y': 1.5, 'acc_z': 12, 
                        'gyro_x': 0.8, 'gyro_y': 0.5, 'gyro_z': 0.3, 'speed': 15}
            }
        ]
        
        for scenario in test_scenarios:
            is_accident, confidence, reason = detector.predict(scenario['data'])
            status = "🚨 ACCIDENT" if is_accident else "✅ SAFE"
            print(f"\n{scenario['name']}: {status}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Reason: {reason}")
        
        print("\n" + "=" * 60)
        print("✅ ML Training Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
