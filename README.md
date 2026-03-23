Delivery Time Prediction using Machine Learning
📌 Project Overview

This project predicts delivery time using a machine learning model based on various influencing factors such as distance, weather, traffic level, time of day, vehicle type, preparation time, and courier experience. The system provides real-time predictions through a web interface by integrating a trained machine learning model with a FastAPI backend.

🚀 Features
Real-time delivery time prediction
User-friendly web interface
Machine learning based estimation
FastAPI backend integration
JSON-based communication

🛠️ Tech Stack
Python
Scikit-learn
FastAPI
HTML & CSS
JavaScript (Fetch API)
Google Colab

📂 Project Structure
delivery-time-prediction/
│
├── main.py
├── model.pkl
├── index.html
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone repository
git clone https://github.com/yourusername/delivery-time-prediction.git
2. Navigate to project folder
cd delivery-time-prediction
3. Install dependencies
pip install fastapi uvicorn scikit-learn joblib

▶️ Run the Project
Step 1 — Start FastAPI server
uvicorn main:app --reload

Server will start at:

http://127.0.0.1:8000
Step 2 — Open frontend

Open index.html in browser.

Step 3 — Test
Enter delivery details
Click Submit
View prediction

🔄 Project Workflow

User Input → Frontend → FastAPI → ML Model → Prediction → Display

📊 Input Features
Distance (km)
Weather
Traffic Level
Time of Day
Vehicle Type
Preparation Time
Courier Experience

🎯 Output
Predicted Delivery Time (minutes)
Delay Prediction

🔮 Future Improvements
Add real-time traffic API
Deploy on cloud
Mobile application
Improve model accuracy

👤 Author
Kanish Thakkar
Mohit Bhatia