from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Create FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="🚚 Delivery Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
reg_model = pickle.load(open("reg_model.pkl", "rb"))
clf_model = pickle.load(open("clf_model.pkl", "rb"))

# (Optional but recommended)
try:
    feature_order = pickle.load(open("features.pkl", "rb"))
except:
    feature_order = [
        'Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day',
        'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs',
        'is_peak', 'high_traffic', 'long_distance', 'experienced_driver'
    ]

# Input schema
class DeliveryInput(BaseModel):
    Distance_km: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str
    Preparation_Time_min: float
    Courier_Experience_yrs: float

# Encoding maps (must match your training)
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}

# Home route
@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: DeliveryInput):

    # Encode categorical values
    weather = weather_map.get(data.Weather, 0)
    traffic = traffic_map.get(data.Traffic_Level, 0)
    time = time_map.get(data.Time_of_Day, 0)
    vehicle = vehicle_map.get(data.Vehicle_Type, 0)

    # Feature engineering (same as your notebook)
    is_peak = 1 if data.Time_of_Day == "Evening" else 0
    high_traffic = 1 if data.Traffic_Level == "High" else 0
    long_distance = 1 if data.Distance_km > 10 else 0
    experienced_driver = 1 if data.Courier_Experience_yrs > 3 else 0

    # Create feature dictionary
    feature_dict = {
        'Distance_km': data.Distance_km,
        'Weather': weather,
        'Traffic_Level': traffic,
        'Time_of_Day': time,
        'Vehicle_Type': vehicle,
        'Preparation_Time_min': data.Preparation_Time_min,
        'Courier_Experience_yrs': data.Courier_Experience_yrs,
        'is_peak': is_peak,
        'high_traffic': high_traffic,
        'long_distance': long_distance,
        'experienced_driver': experienced_driver
    }

    # Arrange features in correct order
    features = np.array([[feature_dict[col] for col in feature_order]])

    # Predictions
    delivery_time = reg_model.predict(features)[0]
    delay = clf_model.predict(features)[0]

    return {
        "predicted_delivery_time_min": round(float(delivery_time), 2),
        "delay_prediction": "Yes" if delay == 1 else "No"
    }