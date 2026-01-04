# 🌸 Iris Flower Classification App

An advanced **Machine Learning web application** built using **Streamlit** that classifies Iris flowers using a **Random Forest model**.  
The app includes interactive UI controls, real-time predictions, probability confidence, and **2D & 3D visualizations** powered by Plotly.

---

## 🚀 Project Overview

- **Dataset:** Iris Dataset (from `sklearn.datasets`)
- **Model Used:** Random Forest Classifier
- **Framework:** Streamlit
- **Visualization:** Plotly (2D + 3D interactive charts)
- **Model Persistence:** Joblib

This project demonstrates a complete ML pipeline:

> Dataset → Model Training → Model Saving → UI-based Prediction & Visualization

---

## 📊 Dataset Information

- **Name:** Iris Dataset
- **Source:** `sklearn.datasets.load_iris`
- **Total Samples:** 150
- **Features:**
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes:**
  - Setosa
  - Versicolor
  - Virginica

The dataset is used to train a **Random Forest Classifier** for multi-class classification.

---

## 🤖 Model Details

- **Algorithm:** Random Forest Classifier
- **Library:** scikit-learn
- **Task:** Multi-class classification
- **Model File:** `iris_rf_model.joblib`

The trained model is loaded inside the Streamlit app and used for real-time predictions.

---

## 📦 Imports Used

### 🔹 Streamlit App (`app.py`)

```python
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
```
