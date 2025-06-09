# 🌟 Affective Computing Deep Learning Framework
# 🎭 Emotion Recognition and 🎶 Music Suggestion via Convolutional Neural Network (CNN)
## 📌 Project Overview
This project integrates real-time emotion recognition using deep learning (CNN/ResNet50V2) and provides personalized music recommendations based on the detected emotion. The goal is to build an affective computing framework that responds to a user's emotional state using a webcam and recommends appropriate songs from predefined datasets.

## 🧠 Core Features
🎥 **Real-Time Emotion Detection** via webcam using CNN (custom/resnet50v2).

😃 Emotion classes: **Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.**

🎵 **Dynamic Song Recommendations** from emotion-based CSV song datasets.

🧩 **Flask Web App** with live video stream + responsive UI (HTML, CSS, Bootstrap).

🖼️ Facial feature extraction via **Haarcascade + Keras/TensorFlow prediction.**

📊 Fully trained models using the **FER2013 dataset.**

## 💡 Project Motivation
Emotion-aware applications can power the next generation of human-computer interaction. This project combines computer vision, deep learning, and music information retrieval to develop a system that recognizes emotions and responds with music.

## 🛠️ Tech Stack
**Category	      Tools/Libraries**
Backend	            Python, Flask
Frontend	          HTML5, CSS3, Bootstrap
Machine Learning	  TensorFlow 2.15, Keras 3.3.3, OpenCV, NumPy
Model	              Custom CNN & ResNet50V2 (transfer learning)
Dataset	            FER2013 (emotion recognition), custom CSVs for music
Deployment	        Localhost (Flask) / Ready for Docker/Cloud Hosting

## 📂 Project Structure
📁 MY-PROJECT/
│
├── app.py                      # Flask app entry point
├── camera.py                   # Webcam + Emotion detection logic
├── utils.py                    # Threaded webcam class
├── Spotipy.py                  # (Optional) Spotify API handling
├── templates/
│   └── index.html              # Main frontend template
├── static/
│   └── style.css               # UI styling
├── songs/
│   ├── happy.csv
│   ├── sad.csv
│   ├── ...
├── model/
│   └── custom_cnn_model.keras  # or resnet50_model.keras
├── haarcascade_frontalface_default.xml
└── README.md                   # You're reading this!

## 🧪 How to Run
## 1. ✅ Clone this repo:
git clone https://github.com/subhramannil/emotion-music-recommender.git
cd emotion-music-recommender
## 2.✅ Install requirements:
pip install -r requirements.txt
## 3. ✅ Start the Flask server:
python app.py
## 4.✅ Open in browser:
Visit http://127.0.0.1:5000

## 5.📈 Model Training Summary
Dataset: FER2013 (48x48 grayscale facial images)
Preprocessing: Rescaling, augmentation (rotation, flip, zoom)

## Model:
* Custom CNN: 4 Conv layers + Dropout + Dense
* ResNet50V2: Transfer learning with frozen base

**Loss:** Categorical Crossentropy
**Optimizer:** Adam
**Evaluation:** Accuracy, Confusion Matrix, Classification Report

## 🎯 Future Improvements
* Add real Spotify/YouTube API integration.
* Deploy on Streamlit or Dockerized Flask App.
* Emotion-to-mood mapping optimization with user feedback loop.
* Voice assistant integration for emotion query.

## 📄 License
This project is open-source and available under the MIT License.

## 🙌 Acknowledgements
FER2013 Dataset (Kaggle)
TensorFlow/Keras Team
OpenCV Community
