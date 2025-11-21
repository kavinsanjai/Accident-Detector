
#### **Rule 1: Extreme Acceleration (Critical)**
```python
if abs(acc_x) > 25 or abs(acc_y) > 25 or abs(acc_z) > 25:
    return True, 1.0, "Extreme G-force detected"
```
**Condition:** Any axis acceleration > 25G  
**Real-world:** Severe collision, hitting a wall at high speed  
**Confidence:** 100%

---

#### **Rule 2: High Impact Force**
```python
acc_magnitude = sqrt(acc_x² + acc_y² + acc_z²)
if acc_magnitude > 20:
    confidence += 0.5
```
**Condition:** Total acceleration force > 20G  
**Real-world:** Hard crash, hitting another vehicle  
**Confidence:** 50%

---

#### **Rule 3: Severe Rotation (Flipping)**
```python
if abs(gyro_x) > 8 or abs(gyro_y) > 8 or abs(gyro_z) > 8:
    return True, 0.95, "Severe rotation/flipping"
```
**Condition:** Any rotation > 8 rad/s (458°/second)  
**Real-world:** Bike flipping end-over-end  
**Confidence:** 95%

---

#### **Rule 4: High Rotation**
```python
gyro_magnitude = sqrt(gyro_x² + gyro_y² + gyro_z²)
if gyro_magnitude > 5:
    confidence += 0.4
```
**Condition:** Total rotation > 5 rad/s  
**Real-world:** Bike tumbling, loss of control  
**Confidence:** 40%

---

#### **Rule 5: Moderate Rotation**
```python
if gyro_magnitude > 3:
    confidence += 0.3
```
**Condition:** Total rotation 3-5 rad/s  
**Real-world:** Sharp turn crash, skidding  
**Confidence:** 30%

---

#### **Rule 6: Combined Forces (Most Common)**
```python
if acc_magnitude > 12 and gyro_magnitude > 2:
    confidence += 0.35
```
**Condition:** Moderate impact (>12G) + rotation (>2 rad/s)  
**Real-world:** Typical accident - impact causes spinning  
**Confidence:** 35%

---

#### **Rule 7: Stationary Impact**
```python
if speed == 0 and acc_magnitude > 15:
    return True, 1.0, "Stationary impact (hit while stopped)"
```
**Condition:** Not moving but high impact  
**Real-world:** Rear-ended at red light  
**Confidence:** 100%

---

#### **Rule 8: Low-Speed Fall**
```python
if speed < 10 and acc_magnitude > 10 and gyro_magnitude > 2:
    confidence += 0.4
```
**Condition:** Slow speed (<10 km/h) + impact + rotation  
**Real-world:** Tipping over, losing balance  
**Confidence:** 40%

---

#### **Rule 9: High-Speed Amplification**
```python
if speed > 60:
    speed_factor = 1.0 + (speed - 60) / 40  # e.g., 80 km/h = 1.5x
    confidence *= speed_factor
```
**Condition:** Speed > 60 km/h amplifies danger  
**Real-world:** Highway crash - speed makes it worse  
**Effect:** Multiplies confidence by 1.5x at 80 km/h

---

#### **Rule 10: Moderate Speed Impact**
```python
if 40 <= speed <= 60:
    confidence += 0.15
```
**Condition:** City/highway transition speeds  
**Real-world:** Urban accident  
**Confidence:** +15%

---

#### **Rule 11: Endo Detection (Forward Flip)**
```python
if abs(acc_z) > 15 and abs(gyro_y) > 3:
    confidence += 0.4
```
**Condition:** High vertical force + forward rotation  
**Real-world:** Hard front braking, going over handlebars  
**Confidence:** 40%

---

#### **Rule 12: Side Impact Detection**
```python
if abs(acc_x) > 15 and abs(gyro_z) > 2:
    confidence += 0.35
```
**Condition:** Sideways acceleration + yaw rotation  
**Real-world:** T-bone collision, sideswipe  
**Confidence:** 35%

---

#### **Rule 13: Rear Impact Detection**
```python
if abs(acc_y) > 15 and speed < 20:
    confidence += 0.3
```
**Condition:** Backward force while slow  
**Real-world:** Rear-ended  
**Confidence:** 30%

---

#### **Rule 14: Tumbling Pattern**
```python
if gyro_x > 3 and gyro_y > 3:
    confidence += 0.4
```
**Condition:** Rotating in multiple axes  
**Real-world:** Bike rolling/tumbling  
**Confidence:** 40%

---

#### **Rule 15: Minor Impact Detection**
```python
if 10 < acc_magnitude < 15 and gyro_magnitude > 1.5:
    confidence += 0.25
```
**Condition:** Light-moderate impact with some rotation  
**Real-world:** Minor collision, hitting pothole hard  
**Confidence:** 25%