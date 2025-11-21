# 🚴 Bike Accident Detection System - Architecture Flowchart

This flowchart shows the step-by-step data flow in your project, from user input to accident detection result.

---

## 📊 System Flowchart

```
[User Input: Sliders / Presets]
        |
        v
[JavaScript Collects Sensor Values]
        |
        v
[POST Request to Flask API]
        |
        v
[Flask Backend Receives Data]
        |
        v
[Model Selection]
   |                |
   v                v
[Rule-Based]   [Machine Learning]
   |                |
   v                v
[Accident Detection Logic]
        |
        v
[Generate JSON Response]
        |
        v
[Frontend Displays Result]
```

---

## 📝 Explanation

1. **User Input:**
   - User sets sensor values using sliders or selects a preset scenario.
2. **JavaScript:**
   - JS gathers all sensor values and chosen model type.
3. **API Request:**
   - JS sends a POST request to the Flask backend (`/api/detect`).
4. **Backend Processing:**
   - Flask receives the data and checks which model to use (Rule-Based or ML).
5. **Detection:**
   - The selected model processes the sensor data and determines if an accident occurred.
6. **Response:**
   - Flask sends back a JSON response with accident status, confidence, and explanation.
7. **Frontend Display:**
   - The result is shown to the user with clear status and details.

---

**This flowchart is ideal for presentations to show how your system works from start to finish!**
