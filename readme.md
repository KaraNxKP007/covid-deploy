# 🧠 Machine Learning Project: COVID-19 Chest X-ray Detection System

> **An AI-powered web application that detects COVID-19 from chest X-ray images using deep learning and computer vision.**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📄 Resume Project Summary

**Project Title:** COVID-19 Chest X-ray Detection System  
**Type:** Academic Project (Machine Learning | Computer Vision)  
**Institution:** Indian Institute of Technology Jodhpur  
**Student:** *Karan Pratap Singh Rathore (B22CH013)*  

### 🎯 Highlights
- Built a **Streamlit-based web app** that classifies X-ray scans into *COVID-19* or *Normal* in **<2 seconds**.  
- Trained a **DenseNet121 transfer learning model** achieving **96.2% accuracy** and **0.978 AUC-ROC** on a dataset of **40,000+ images**.  
- Designed a **medical-grade preprocessing pipeline** using CLAHE and normalization for improved contrast and detection reliability.  

### 💻 Tech Stack
`Python` · `TensorFlow` · `Keras` · `Streamlit` · `OpenCV` · `NumPy` · `Pandas` · `Plotly` · `Matplotlib`

---

## 🌟 Features

- **AI-Powered Detection**: DenseNet121 with transfer learning for accurate COVID-19 detection  
- **Real-time Analysis**: Provides results in under 2 seconds per image  
- **Medical-grade Preprocessing**: CLAHE enhancement and image normalization  
- **Interactive Dashboard**: Visualization with confidence scores and risk assessment  
- **Multi-page Interface**: Dashboard, detection, insights, and about sections  
- **Responsive Design**: Works seamlessly on desktop and mobile devices  

---

## 🚀 Live Demo

The application is deployed and live at:  
🔗 **[https://covid-deploy-lfvgaacdplv5cd68aafvkb.streamlit.app/](https://covid-deploy-lfvgaacdplv5cd68aafvkb.streamlit.app/)**


## 🛠️ Technology Stack

- **Frontend:** Streamlit, HTML/CSS  
- **Backend:** Python, TensorFlow, Keras  
- **Computer Vision:** OpenCV, PIL/Pillow  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **Deployment:** Streamlit Community Cloud  

---

## 📊 Model Architecture

- **Base Model:** DenseNet121 (Transfer Learning)  
- **Input Size:** 224×224×3  
- **Output:** Binary Classification (COVID-19 / Normal)  
- **Training Data:** 40,000+ chest X-ray images  
- **Performance:** 96.2% accuracy, 0.978 AUC-ROC  

---

## 🎯 Key Metrics

| Metric | Score |
|--------|-------|
| ✅ Accuracy | 96.2% |
| 🎯 Precision | 95.1% |
| 🔁 Recall | 94.8% |
| 🧮 F1-Score | 94.9% |
| 📈 AUC-ROC | 0.978 |
| ⚡ Inference Time | < 2 seconds |

---


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/KaranxKP007/covid-deploy.git
cd covid-deploy

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run covid_detection_app.py
