🌞Solar Energy Predictor

**A Machine Learning Model for Predicting Solar Energy Potential using Weather Data**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/SDG7-Affordable%20&%20Clean%20Energy-brightgreen)

#Project Overview

This project aligns with **UN Sustainable Development Goal (SDG) 7: Affordable and Clean Energy**, which focuses on ensuring access to affordable, reliable, sustainable, and modern energy for all.

The **Solar Energy Predictor** uses **supervised machine learning** to estimate **solar energy potential** based on key weather parameters such as:

* Temperature
* Humidity
* Wind Speed
* Solar Radiation
* Cloud Cover

By leveraging predictive modeling, this tool helps researchers, energy companies, and policymakers **identify optimal locations and times for solar power generation**, promoting cleaner and more sustainable energy production.

# Machine Learning Approach

* **Learning Type:** Supervised Learning
* **Model Used:** Linear Regression / Random Forest / XGBoost (comparative analysis)
* **Target Variable:** Solar Energy Output
* **Input Features:** Weather parameters (temperature, humidity, pressure, etc.)
* 
# Tech Stack

| Component            | Tool          |
| -------------------- | ------------- |
| Programming Language | Python        |
| Data Visualization   | Plotly        |
| Web Framework        | Streamlit     |
| Data Handling        | Pandas, NumPy |
| Machine Learning     | Scikit-learn  |
| Version Control      | Git, GitHub   |

---

# Project Structure

```
SolarEnergyPredictor/
│
├── streamlit_solar_app.py       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── venv/                        # Virtual environment (excluded in .gitignore)
```

---

# Setup & Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/raykiarie123-alt/SolarEnergyPredictor.git
   cd SolarEnergyPredictor
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run streamlit_solar_app.py
   ```

5. *View in Browser**
   Visit `http://localhost:8501`

# Datasets

You can source open weather and solar datasets from:

* [NASA POWER Data API](https://power.larc.nasa.gov/)
(for real-time data integration)*

# Model Evaluation

Performance metrics include:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

# Reflection
Bias Impact:
If weather data is regionally imbalanced (e.g., only urban or dry areas), the model may generalize poorly across diverse climates.

**Fairness & Sustainability:**
The model promotes energy equity by helping identify new solar opportunities in underrepresented regions, fostering sustainable growth.

# Stretch Goals

✅ Integrate **real-time weather data** via APIs
✅ Deploy the app using **Streamlit Cloud or Render**
✅ Compare multiple ML algorithms for better accuracy

# License

This project is licensed under the **MIT License** — free to use and modify with attribution.

# Author

Rachael Kiarie
📍 Data Science & BBIT Student | JKUAT
💡 Passionate about AI for Climate and Energy
🌐 GitHub: [@raykiarie123-alt](https://github.com/raykiarie123-alt)

