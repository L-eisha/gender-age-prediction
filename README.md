# 👤 Gender & Age Prediction

A Python-based computer vision application that detects faces and predicts gender and age from images. Built with a Streamlit UI supporting multiple image uploads.

---

## 📋 Overview

This app uses face detection and deep learning models to analyze images and predict the **gender** and **age** of detected faces in real time. Upload one or multiple images through the web interface and get instant predictions.

---

## 🗂️ Project Structure

```
gender-age-prediction/
├── app.py                # Streamlit app — UI and image handling
├── face_detection.py     # Face detection logic
├── gender_age.py         # Gender and age prediction model
├── test_images/          # Sample images for testing
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/L-eisha/gender-age-prediction.git
   cd gender-age-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 🎯 Features

- 🔍 **Face Detection** — Automatically detects faces in uploaded images
- 👫 **Gender Prediction** — Classifies detected faces as Male or Female
- 🎂 **Age Prediction** — Estimates the age range of detected faces
- 🖼️ **Multiple Image Support** — Upload and analyze several images at once
- 🌐 **Streamlit UI** — Clean, interactive web interface

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web UI framework
- **OpenCV** — Face detection and image processing
- **Deep Learning Model** — Gender and age prediction

---

## 🧪 Testing

Sample images are available in the `test_images/` folder to try out the app right away.

---

## 👤 Author

**Leisha Choudhary** — [@L-eisha](https://github.com/L-eisha)
