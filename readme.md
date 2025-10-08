# 🧠 AI Project #01: COVID-19 Chest X-ray Detection System

> **An AI-powered web application that detects COVID-19 from chest X-ray images using deep learning and computer vision.**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📄 Resume Project Summary

**Project Title:** COVID-19 Chest X-ray Detection System  
**Type:** Academic + Research Project (Machine Learning | Computer Vision)  
**Institution:** Indian Institute of Technology Jodhpur  
**Student:** *Karan Pratap Singh Rathore (B22CH013)*  

### 🎯 Highlights
- Built a **Streamlit-based web app** that classifies X-ray scans into *COVID-19* or *Normal* in **<2 seconds**.  
- Trained a **DenseNet121 transfer learning model** achieving **96.2% accuracy** and **0.978 AUC-ROC** on a dataset of **40,000+ images**.  
- Designed a **medical-grade preprocessing pipeline** using CLAHE and normalization for improved contrast and detection reliability.  

### 💻 Tech Stack
`Python` · `TensorFlow` · `Keras` · `Streamlit` · `OpenCV` · `NumPy` · `Pandas` · `Plotly` · `Matplotlib`

---

## 🌐 Live Demo  
🔗 [**Streamlit App**](https://covid-deploy-lfvgaacdplv5cd68aafvkb.streamlit.app/)  

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| ✅ Accuracy | 96.2% |
| 🎯 Precision | 95.1% |
| 🔁 Recall | 94.8% |
| 🧮 F1-Score | 94.9% |
| 📈 AUC-ROC | 0.978 |
| ⚡ Inference Time | < 2 seconds |

---

## 🧩 System Architecture

- **Base Model:** DenseNet121 (Transfer Learning)  
- **Input Size:** 224×224×3  
- **Output:** Binary Classification (COVID-19 / Normal)  
- **Dataset:** COVID-19 Radiography Database (40k+ images)  
- **Enhancements:** CLAHE, histogram equalization, normalization, and augmentation  


## 📁 Project Structure
covid-deploy/
├── covid_detection_app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── deployment/ # Trained models
│ ├── best_DenseNet121_TL.h5
│ └── best_Custom_CNN_V3.h5
└── datasets/ # Dataset (not included)



---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/KaranxKP007/covid-deploy.git
cd covid-deploy

# Create virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run covid_detection_app.py

