"""
Test ML Model Sensitivity - Compare Rule-Based vs ML Predictions
This script tests various scenarios to show ML model detects subtle changes
"""

import requests
import json
from colorama import init, Fore, Style

init(autoreset=True)

# Server URL
BASE_URL = "http://localhost:5000"

def test_scenario(name, sensor_data, expected_rule_based, expected_ml):
    """Test a scenario with both models and compare results"""
    print(f"\n{'='*80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{name}{Style.RESET_ALL}")
    print(f"{'='*80}")
    
    # Print sensor values
    print(f"{Fore.YELLOW}📊 Sensor Values:{Style.RESET_ALL}")
    print(f"   Acceleration: X={sensor_data['acc_x']:.1f}G, Y={sensor_data['acc_y']:.1f}G, Z={sensor_data['acc_z']:.1f}G")
    print(f"   Gyroscope: X={sensor_data['gyro_x']:.1f}°/s, Y={sensor_data['gyro_y']:.1f}°/s, Z={sensor_data['gyro_z']:.1f}°/s")
    print(f"   Speed: {sensor_data['speed']:.1f} km/h")
    
    # Test Rule-Based Model
    print(f"\n{Fore.MAGENTA}⚙️  RULE-BASED MODEL:{Style.RESET_ALL}")
    rule_data = sensor_data.copy()
    rule_data['model_type'] = 'rule-based'
    
    try:
        response = requests.post(f"{BASE_URL}/api/detect", json=rule_data)
        result = response.json()
        
        if result['accident_detected']:
            print(f"   {Fore.RED}🚨 ACCIDENT DETECTED{Style.RESET_ALL}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Reasons: {result['reasons'][:100]}...")
        else:
            print(f"   {Fore.GREEN}✅ SAFE{Style.RESET_ALL}")
            print(f"   Confidence: {100 - result['confidence']:.1f}%")
    except Exception as e:
        print(f"   {Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    # Test ML Model
    print(f"\n{Fore.BLUE}🤖 MACHINE LEARNING MODEL:{Style.RESET_ALL}")
    ml_data = sensor_data.copy()
    ml_data['model_type'] = 'ml'
    
    try:
        response = requests.post(f"{BASE_URL}/api/detect", json=ml_data)
        result = response.json()
        
        if result['accident_detected']:
            print(f"   {Fore.RED}🚨 ACCIDENT DETECTED{Style.RESET_ALL}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Reason: {result['reasons']}")
        else:
            print(f"   {Fore.GREEN}✅ SAFE{Style.RESET_ALL}")
            print(f"   Confidence: {100 - result['confidence']:.1f}%")
    except Exception as e:
        print(f"   {Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}📝 Expected Results:{Style.RESET_ALL}")
    print(f"   Rule-Based: {expected_rule_based}")
    print(f"   ML Model: {expected_ml}")

def main():
    print(f"{Fore.GREEN}{Style.BRIGHT}")
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║     ML MODEL SENSITIVITY TEST - COMPARING PREDICTIONS                  ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)
    
    # Test 1: Normal city riding (both should agree)
    test_scenario(
        "TEST 1: Normal City Riding",
        {
            'acc_x': 0.5, 'acc_y': 0.3, 'acc_z': 9.8,
            'gyro_x': 0.2, 'gyro_y': 0.1, 'gyro_z': 0.3,
            'speed': 30
        },
        "SAFE - Normal riding",
        "SAFE - Normal riding"
    )
    
    # Test 2: Minor bump (ML should be more sensitive)
    test_scenario(
        "TEST 2: Minor Bump - Small Pothole",
        {
            'acc_x': 3.0, 'acc_y': 2.0, 'acc_z': 12.0,
            'gyro_x': 1.5, 'gyro_y': 1.0, 'gyro_z': 0.5,
            'speed': 25
        },
        "SAFE - Minor disturbance",
        "May detect accident - More sensitive"
    )
    
    # Test 3: Moderate concern (testing sensitivity)
    test_scenario(
        "TEST 3: Hard Cornering",
        {
            'acc_x': 5.0, 'acc_y': 8.0, 'acc_z': 10.0,
            'gyro_x': 2.5, 'gyro_y': 1.5, 'gyro_z': 3.0,
            'speed': 45
        },
        "Low confidence accident",
        "Should detect accident"
    )
    
    # Test 4: Emergency braking (both should detect)
    test_scenario(
        "TEST 4: Emergency Braking",
        {
            'acc_x': -12.0, 'acc_y': 1.0, 'acc_z': 9.5,
            'gyro_x': 0.5, 'gyro_y': 2.5, 'gyro_z': 0.3,
            'speed': 50
        },
        "ACCIDENT - Hard braking",
        "ACCIDENT - Pattern detected"
    )
    
    # Test 5: Moderate lateral force
    test_scenario(
        "TEST 5: Swerving to Avoid Obstacle",
        {
            'acc_x': 2.0, 'acc_y': 11.0, 'acc_z': 9.8,
            'gyro_x': 2.0, 'gyro_y': 1.0, 'gyro_z': 2.5,
            'speed': 40
        },
        "Medium confidence accident",
        "Should detect accident"
    )
    
    # Test 6: High rotation at low speed (falling over)
    test_scenario(
        "TEST 6: Tipping Over at Stop",
        {
            'acc_x': 1.0, 'acc_y': 3.0, 'acc_z': 8.0,
            'gyro_x': 4.0, 'gyro_y': 2.0, 'gyro_z': 1.0,
            'speed': 5
        },
        "ACCIDENT - Fall detected",
        "ACCIDENT - Pattern detected"
    )
    
    # Test 7: Severe crash (both definitely detect)
    test_scenario(
        "TEST 7: Frontal Collision",
        {
            'acc_x': -28.0, 'acc_y': 5.0, 'acc_z': 15.0,
            'gyro_x': 2.0, 'gyro_y': 7.0, 'gyro_z': 3.0,
            'speed': 60
        },
        "ACCIDENT - Severe crash",
        "ACCIDENT - Severe crash"
    )
    
    # Test 8: Tumbling (both should detect)
    test_scenario(
        "TEST 8: Bike Tumbling/Rolling",
        {
            'acc_x': 8.0, 'acc_y': 12.0, 'acc_z': 18.0,
            'gyro_x': 6.0, 'gyro_y': 5.0, 'gyro_z': 4.0,
            'speed': 35
        },
        "ACCIDENT - High rotation",
        "ACCIDENT - Pattern detected"
    )
    
    # Test 9: Slight instability (testing sensitivity threshold)
    test_scenario(
        "TEST 9: Slight Wobble at Speed",
        {
            'acc_x': 1.5, 'acc_y': 2.0, 'acc_z': 10.5,
            'gyro_x': 1.8, 'gyro_y': 1.2, 'gyro_z': 1.5,
            'speed': 70
        },
        "SAFE - Minor wobble",
        "May detect as concerning"
    )
    
    # Test 10: Combined moderate forces
    test_scenario(
        "TEST 10: Rough Road at Moderate Speed",
        {
            'acc_x': 6.0, 'acc_y': 5.0, 'acc_z': 13.0,
            'gyro_x': 2.5, 'gyro_y': 2.0, 'gyro_z': 1.8,
            'speed': 45
        },
        "Low-Medium confidence",
        "Should detect accident"
    )
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}")
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║                        TEST COMPLETE                                   ║")
    print("║                                                                        ║")
    print("║  The ML model should now detect MORE accidents than before,           ║")
    print("║  especially subtle/moderate scenarios. It's trained with the same     ║")
    print("║  thresholds as the rule-based system but can learn patterns.          ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error running tests: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure the server is running on http://localhost:5000{Style.RESET_ALL}")
