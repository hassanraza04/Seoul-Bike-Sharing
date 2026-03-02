# Seoul Bike Sharing – Demand Dashboard

A Streamlit app that predicts hourly bike-sharing demand in Seoul using linear regression. Built for a data science course project.

**Live app:** [https://seoul-bike-sharing.streamlit.app](https://seoul-bike-sharing.streamlit.app)

## Overview

The app helps a bike-sharing operator anticipate **hourly demand** using historical data (weather and calendar) so they can rebalance bikes, plan maintenance, and optimize staffing.

## Dataset

- **Source:** [Kaggle – Seoul Bike Sharing Demand Prediction](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction/data?select=SeoulBikeData.csv).
- **Contents:** Hourly records with weather (temperature, humidity, wind, visibility, solar radiation, rainfall, snowfall) and calendar features (hour, season, holiday, functioning day).
- **Target variable:** `Rented Bike Count`.

Place `SeoulBikeData.csv` in the same folder as `app.py`.

## App structure

| Page | Description |
|------|-------------|
| **Introduction** | Business case, dataset source, data preview, missing values, summary statistics, column descriptions. |
| **Visualization** | Guided insights (demand by hour, season, temperature, holiday), interactive explore charts, correlation heatmap. |
| **Prediction** | Linear regression on Rented Bike Count; feature selection, MSE/MAE/R², driving variables (coefficients), actual vs predicted plot. |

## Tech stack

- **App:** Streamlit  
- **Data:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Model:** Scikit-learn (Linear Regression)

## Setup and run

### 1. Clone or download the repo

Ensure the project folder contains `app.py`, `SeoulBikeData.csv`, and `requirements.txt`.

### 2. Create a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python3 -m streamlit run app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

## Project structure

```
ds_proj/
├── README.md           # This file
├── app.py              # Streamlit app
├── requirements.txt    # Python dependencies
├── SeoulBikeData.csv   # Dataset (add this file)
└── bike.png            # Optional: landing image
```

## License

This project was created for educational purposes.
