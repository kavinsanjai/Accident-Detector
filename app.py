

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
import os
from working_accident_system import WorkingAccidentDetector

# Try to import ML detector
try:
    from ml_accident_detector import MLAccidentDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML model not available. Install scikit-learn or train the model first.")

app = Flask(__name__)
CORS(app)

# Initialize the rule-based detector
detector = WorkingAccidentDetector()

# Try to load ML model if available
ml_detector = None
if ML_AVAILABLE and os.path.exists('ml_accident_model.pkl'):
    try:
        ml_detector = MLAccidentDetector()
        ml_detector.load_model('ml_accident_model.pkl')
        print("✅ ML model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load ML model: {e}")
        ml_detector = None

def generate_human_explanation(sensor_data, is_accident, confidence, technical_reason, metrics):
    """
    Generate easy-to-understand explanation for non-technical users.
    Explains what happened in everyday language.
    """
    speed = sensor_data.get('speed', 0)
    acc_x = sensor_data.get('acc_x', 0)
    acc_y = sensor_data.get('acc_y', 0)
    acc_z = sensor_data.get('acc_z', 0)
    acc_mag = metrics['acc_magnitude']
    gyro_mag = metrics['gyro_magnitude']
    
    explanation = {
        'simple_status': '',
        'what_happened': '',
        'why_detected': '',
        'analogy': '',
        'safety_tip': ''
    }
    
    if is_accident:
        # Determine accident type from sensor values
        if speed < 5 and acc_mag > 15:
            # Stationary impact
            explanation['simple_status'] = "Your bike was hit while stopped"
            explanation['what_happened'] = f"Your bike wasn't moving ({speed:.0f} km/h), but the sensors detected a significant impact force of {acc_mag:.1f}G. This indicates something collided with your parked bike, or you were rear-ended while stopped at a traffic light."
            explanation['why_detected'] = "When a bike is stationary, it should only experience gravity pulling downward (approximately 10G). Any force exceeding 15G indicates an external collision has occurred."
            explanation['analogy'] = "Imagine you're standing completely still, and someone suddenly pushes you hard from behind. You haven't moved from your position, but you definitely felt that strong impact."
            explanation['safety_tip'] = "Even when stopped, accidents can happen. Always stay aware of traffic around you, especially at intersections and traffic lights."
            
        elif acc_x < -15 and speed > 30:
            # Sudden braking / Endo
            explanation['simple_status'] = "Sudden emergency stop detected - risk of flipping"
            explanation['what_happened'] = f"You were riding at {speed:.0f} km/h and suddenly experienced extreme backward force ({abs(acc_x):.1f}G). This is characteristic of very hard braking."
            explanation['why_detected'] = "When you brake too hard on a bike, the front wheel stops abruptly but your body continues moving forward due to momentum. This can cause you to flip over the handlebars, which cyclists call an 'endo'."
            explanation['analogy'] = "Think about running at full speed and suddenly grabbing onto a fixed pole. Your legs stop immediately, but your upper body keeps moving forward because of inertia."
            explanation['safety_tip'] = "Practice gradual, controlled braking. Always use both front and rear brakes together for safer stops, and shift your weight backward when braking hard."
            
        elif gyro_mag > 25:
            # Tumbling/flipping
            explanation['simple_status'] = "Bike is tumbling or flipping"
            explanation['what_happened'] = f"The bike is rotating at {gyro_mag:.1f} degrees per second, which is extremely fast. Your bike is likely rolling over or flipping through the air."
            explanation['why_detected'] = "During normal riding, bikes rotate slowly through turns (typically 1-5 degrees per second). Rotation speeds above 25 degrees per second indicate the bike is completely out of control and tumbling."
            explanation['analogy'] = "Compare doing a controlled turn while walking versus doing an uncontrolled somersault. One is smooth and predictable, the other is chaotic spinning motion."
            explanation['safety_tip'] = "This type of rotation usually happens after a collision or when hitting a significant obstacle. Always wear a helmet and protective gear when cycling."
            
        elif acc_mag > 25:
            # Extreme impact
            explanation['simple_status'] = "Severe crash detected - extreme impact"
            explanation['what_happened'] = f"Your bike experienced {acc_mag:.1f}G of force. Normal riding typically produces around 10G. This level of force indicates a serious collision has occurred."
            explanation['why_detected'] = "Forces exceeding 25G mean something hit your bike with tremendous force. This could be a vehicle collision, hitting a solid wall, or a major fall at speed."
            explanation['analogy'] = "Consider the difference between gently placing a book on a table versus throwing it violently against a wall. The impact force is dramatically different."
            explanation['safety_tip'] = "This represents a serious accident. Immediately check yourself and any passengers for injuries. Seek medical attention if you experience any pain or disorientation."
            
        elif speed > 60 and acc_mag > 8:
            # High-speed collision
            explanation['simple_status'] = f"High-speed accident at {speed:.0f} km/h"
            explanation['what_happened'] = f"You were riding fast ({speed:.0f} km/h - that's highway speed) and hit something with {acc_mag:.1f}G of force. At high speeds, even small impacts are dangerous."
            explanation['why_detected'] = "Speed increases danger exponentially. An 8G impact at 60 km/h is much more dangerous than 8G at 20 km/h because of the kinetic energy involved."
            explanation['analogy'] = "Think of falling off a chair versus falling off a roof - the same motion, but vastly different outcomes because of the height."
            explanation['safety_tip'] = "High-speed cycling requires full protective gear. Always wear a helmet and consider body armor for speeds above 50 km/h."
            
        elif abs(acc_y) > 20:
            # Side impact
            explanation['simple_status'] = "Side impact - something hit you from the side"
            explanation['what_happened'] = f"Your bike was hit from the left or right with {abs(acc_y):.1f}G of sideways force. This often happens when a vehicle doesn't see you and turns into your path."
            explanation['why_detected'] = "Side impacts (Y-axis) are particularly dangerous for cyclists because you can't see them coming and they often knock you off balance."
            explanation['analogy'] = "Picture someone opening a door into you when you're walking past - you don't expect it and get knocked sideways."
            explanation['safety_tip'] = "Always be visible! Use lights, wear bright colors, and avoid vehicle blind spots, especially at intersections."
            
        else:
            # General accident
            explanation['simple_status'] = "Accident detected - abnormal forces"
            explanation['what_happened'] = f"Sensors detected unusual forces: {acc_mag:.1f}G impact while riding at {speed:.0f} km/h with {gyro_mag:.1f}°/s rotation. These readings are outside normal riding patterns."
            explanation['why_detected'] = "During normal cycling, forces stay relatively low and predictable. These readings indicate something unexpected happened."
            explanation['analogy'] = "It's like walking on a smooth path (predictable) versus stumbling over an unseen obstacle (sudden change)."
            explanation['safety_tip'] = "Stay alert and anticipate potential hazards. Defensive cycling can prevent accidents before they happen."
            
    else:
        # No accident - safe riding
        if speed < 5:
            explanation['simple_status'] = "All good - bike is stationary and safe"
            explanation['what_happened'] = f"Your bike is parked or stopped ({speed:.0f} km/h) with normal forces ({acc_mag:.1f}G). Everything looks normal."
            explanation['why_detected'] = "When parked, your bike should only feel gravity pulling down (about 10G). The sensors show normal readings with no unusual impacts."
            explanation['analogy'] = "It's like sitting quietly on a bench - peaceful and stable."
            explanation['safety_tip'] = "You're safe! Take your time before starting to ride."
            
        elif speed > 50:
            explanation['simple_status'] = f"High-speed cruising - riding fast but safely at {speed:.0f} km/h"
            explanation['what_happened'] = f"You're riding fast ({speed:.0f} km/h) but all forces are normal: {acc_mag:.1f}G acceleration and {gyro_mag:.1f}°/s rotation. This is smooth, controlled high-speed riding."
            explanation['why_detected'] = "High speed alone isn't dangerous if everything is controlled. Your bike is stable with no sudden impacts or loss of control."
            explanation['analogy'] = "It's like riding a smooth elevator going fast - speed without chaos."
            explanation['safety_tip'] = "Great! Maintain this smooth control. Stay focused and anticipate obstacles ahead."
            
        elif acc_mag > 12:
            explanation['simple_status'] = "Minor bump or acceleration - normal riding"
            explanation['what_happened'] = f"You felt {acc_mag:.1f}G of force at {speed:.0f} km/h. This could be from pedaling hard, going over a small bump, or turning."
            explanation['why_detected'] = "Normal cycling involves some forces from pedaling, bumps, and turns. These readings are within the safe range for regular riding."
            explanation['analogy'] = "It's like jogging over slightly uneven ground - you feel it, but you're not falling."
            explanation['safety_tip'] = "You're riding normally. Keep your eyes on the road and hands on the handlebars."
            
        else:
            explanation['simple_status'] = f"Smooth cycling - everything is perfect at {speed:.0f} km/h"
            explanation['what_happened'] = f"You're riding smoothly at {speed:.0f} km/h with minimal forces ({acc_mag:.1f}G) and stable balance ({gyro_mag:.1f}°/s rotation). This is ideal cycling."
            explanation['why_detected'] = "All sensor readings are in the normal, safe range. Your riding is smooth and controlled."
            explanation['analogy'] = "It's like gliding on ice or a smooth road - effortless and safe."
            explanation['safety_tip'] = "Perfect! You're cycling safely. Enjoy your ride."
    
    return explanation

# Preset scenarios for quick testing
PRESET_SCENARIOS = {
    'severe_collision': {
        'name': '💥 Severe Head-On Collision',
        'description': 'High-speed frontal impact with extreme forces',
        'params': {'acc_x': 45, 'acc_y': 25, 'acc_z': 55, 'gyro_x': 30, 'gyro_y': 25, 'gyro_z': 35, 'speed': 90},
        'expected': True
    },
    'side_impact': {
        'name': '🚗 Massive Side Impact',
        'description': 'Vehicle hitting from the side',
        'params': {'acc_x': 8, 'acc_y': 50, 'acc_z': 35, 'gyro_x': 15, 'gyro_y': 45, 'gyro_z': 25, 'speed': 60},
        'expected': True
    },
    'rear_hit': {
        'name': '🚛 High-Speed Rear Hit',
        'description': 'Being hit from behind at high speed',
        'params': {'acc_x': -25, 'acc_y': 15, 'acc_z': 38, 'gyro_x': -20, 'gyro_y': 12, 'gyro_z': 18, 'speed': 80},
        'expected': True
    },
    'obstacle_crash': {
        'name': '🌳 Crash Into Obstacle',
        'description': 'Hitting a fixed object like a tree or pole',
        'params': {'acc_x': 35, 'acc_y': 10, 'acc_z': 25, 'gyro_x': 20, 'gyro_y': 7, 'gyro_z': 12, 'speed': 45},
        'expected': True
    },
    'sudden_brake': {
        'name': '🚦 Sudden Brake at High Speed',
        'description': 'Emergency stop at highway speed',
        'params': {'acc_x': -18, 'acc_y': 2, 'acc_z': 12, 'gyro_x': 3, 'gyro_y': 1, 'gyro_z': 2, 'speed': 85},
        'expected': True
    },
    'normal_riding': {
        'name': '🚴 Normal Smooth Riding',
        'description': 'Regular cycling on a smooth road',
        'params': {'acc_x': 1.4, 'acc_y': 0.2, 'acc_z': 9.8, 'gyro_x': 0.1, 'gyro_y': 0.0, 'gyro_z': 0.1, 'speed': 25},
        'expected': False
    },
    'normal_braking': {
        'name': '🛑 Normal Braking',
        'description': 'Regular controlled braking',
        'params': {'acc_x': -2.0, 'acc_y': 0.8, 'acc_z': 9.2, 'gyro_x': 0.6, 'gyro_y': 0.1, 'gyro_z': 0.0, 'speed': 40},
        'expected': False
    },
    'aggressive_acceleration': {
        'name': '🏎️ Aggressive Acceleration',
        'description': 'Quick takeoff from a stop',
        'params': {'acc_x': 5.0, 'acc_y': 1.5, 'acc_z': 8.5, 'gyro_x': 1.2, 'gyro_y': 0.8, 'gyro_z': 0.5, 'speed': 70},
        'expected': False
    },
    'pothole': {
        'name': '🚧 Minor Pothole Hit',
        'description': 'Small bump in the road',
        'params': {'acc_x': 2.0, 'acc_y': 3.0, 'acc_z': 15.0, 'gyro_x': 2.0, 'gyro_y': 1.5, 'gyro_z': 1.0, 'speed': 30},
        'expected': False
    },
    'high_speed_cruise': {
        'name': '🏁 High-Speed Cruising',
        'description': 'Normal highway driving',
        'params': {'acc_x': 0.5, 'acc_y': 0.3, 'acc_z': 9.8, 'gyro_x': 0.2, 'gyro_y': 0.1, 'gyro_z': 0.1, 'speed': 100},
        'expected': False
    }
}

@app.route('/')
def index():
    """Render the main simulation page."""
    return render_template('index_with_vehicle_speed.html')

@app.route('/api/model_status')
def model_status():
    """Check which models are available."""
    return jsonify({
        'rule_based_available': True,
        'ml_available': ml_detector is not None,
        'ml_model_path': 'ml_accident_model.pkl' if ml_detector else None
    })

@app.route('/api/presets')
def get_presets():
    """Get all preset scenarios."""
    return jsonify(PRESET_SCENARIOS)

@app.route('/api/detect', methods=['POST'])
def detect_accident():
    """
    Detect accident from sensor parameters.
    
    Expected JSON format:
    {
        "acc_x": float,
        "acc_y": float,
        "acc_z": float,
        "gyro_x": float,
        "gyro_y": float,
        "gyro_z": float,
        "speed": float (optional, defaults to 0)
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        required_keys = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing parameter: {key}'}), 400
        
        # Convert to float (speed is optional)
        sensor_data = {key: float(data[key]) for key in required_keys}
        sensor_data['speed'] = float(data.get('speed', 0))  # Default speed to 0 if not provided
        
        # Get model choice (default to rule-based)
        model_type = data.get('model_type', 'rule-based')
        
        # Calculate magnitudes for response
        acc_magnitude = np.sqrt(sensor_data['acc_x']**2 + sensor_data['acc_y']**2 + sensor_data['acc_z']**2)
        gyro_magnitude = np.sqrt(sensor_data['gyro_x']**2 + sensor_data['gyro_y']**2 + sensor_data['gyro_z']**2)
        total_magnitude = acc_magnitude + gyro_magnitude
        
        # Choose detection method
        if model_type == 'ml' and ml_detector is not None:
            # Use ML model
            is_accident, confidence, reason = ml_detector.predict(sensor_data)
            model_used = "Machine Learning (Random Forest)"
        else:
            # Use rule-based model
            is_accident, confidence, reason = detector.detect_accident(sensor_data)
            model_used = "Rule-Based (Physics)"
        
        # Convert confidence to percentage for better display
        confidence_percent = confidence * 100
        
        # Determine severity level based on confidence
        if confidence_percent >= 90:
            severity = "CRITICAL"
            severity_color = "#DC2626"  # Red
        elif confidence_percent >= 70:
            severity = "HIGH"
            severity_color = "#EA580C"  # Orange
        elif confidence_percent >= 50:
            severity = "MODERATE"
            severity_color = "#F59E0B"  # Amber
        elif confidence_percent >= 40:
            severity = "LOW"
            severity_color = "#EAB308"  # Yellow
        else:
            severity = "MINIMAL"
            severity_color = "#22C55E"  # Green
        
        # Calculate metrics
        metrics = {
            'speed': float(sensor_data['speed']),
            'acc_magnitude': float(acc_magnitude),
            'gyro_magnitude': float(gyro_magnitude),
            'total_magnitude': float(total_magnitude),
            'max_acc_axis': float(max(abs(sensor_data['acc_x']), abs(sensor_data['acc_y']), abs(sensor_data['acc_z']))),
            'max_gyro_axis': float(max(abs(sensor_data['gyro_x']), abs(sensor_data['gyro_y']), abs(sensor_data['gyro_z'])))
        }
        
        # Generate human-readable explanation
        explanation = generate_human_explanation(sensor_data, is_accident, confidence, reason, metrics)
        
        # Prepare response
        response = {
            'is_accident': bool(is_accident),
            'confidence': float(confidence),
            'confidence_percent': float(confidence_percent),
            'severity': severity,
            'severity_color': severity_color,
            'reason': reason,
            'model_used': model_used,
            'explanation': explanation,  # New: Easy-to-understand explanation
            'metrics': metrics,
            'thresholds': {
                'acc_severe': 20,
                'acc_high': 15,
                'acc_moderate': 10,
                'gyro_severe': 30,
                'gyro_high': 20,
                'gyro_moderate': 10,
                'total_severe': 60,
                'total_moderate': 40,
                'confidence_threshold': 40,  # Updated to 40%
                'speed_high': 80,
                'speed_moderate': 50
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/batch_test', methods=['POST'])
def batch_test():
    """
    Test multiple scenarios at once.
    
    Expected JSON format:
    {
        "scenarios": [
            {"name": "Test 1", "data": {...}, "expected": true/false},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        scenarios = data.get('scenarios', [])
        
        results = []
        for scenario in scenarios:
            sensor_data = scenario.get('data', {})
            is_accident, confidence, reason = detector.detect_accident(sensor_data)
            
            result = {
                'name': scenario.get('name', 'Unnamed'),
                'is_accident': bool(is_accident),
                'confidence': float(confidence),
                'reason': reason,
                'expected': scenario.get('expected', None),
                'correct': is_accident == scenario.get('expected') if 'expected' in scenario else None
            }
            results.append(result)
        
        # Calculate statistics
        total = len(results)
        with_expectations = [r for r in results if r['expected'] is not None]
        correct = sum(1 for r in with_expectations if r['correct'])
        accuracy = (correct / len(with_expectations)) if with_expectations else None
        
        return jsonify({
            'results': results,
            'statistics': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/system_info')
def system_info():
    """Get information about the detection system."""
    return jsonify({
        'system_type': 'Rule-Based Physics Detection',
        'version': '1.0',
        'features': [
            'Acceleration magnitude detection',
            'Gyroscope rotation analysis',
            'Combined system shock detection',
            'Individual axis extreme detection',
            'Confidence scoring'
        ],
        'thresholds': {
            'acceleration': {
                'severe': '20G+',
                'high': '15-20G',
                'moderate': '10-15G',
                'normal': '<10G'
            },
            'rotation': {
                'severe': '30°/s+',
                'high': '20-30°/s',
                'moderate': '10-20°/s',
                'normal': '<10°/s'
            },
            'confidence': {
                'threshold': '30%',
                'description': 'Minimum confidence to classify as accident'
            }
        }
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 ACCIDENT DETECTION SIMULATION SERVER")
    print("=" * 70)
    print("✅ Rule-based model loaded successfully")
    if ml_detector:
        print("✅ ML model (Random Forest) loaded successfully")
    else:
        print("⚠️  ML model not available (run: python ml_accident_detector.py)")
    print("🌐 Starting web server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
