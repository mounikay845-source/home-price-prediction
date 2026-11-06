# 🏡 Real Estate Home Price Prediction Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)

A supervised machine learning regression pipeline engineered to predict residential real estate property values based on socio-economic, infrastructural, and neighborhood factors.

---

## 🔬 Model Overview & Methodology

The model employs the gradient boosted trees algorithm via **XGBoost Regressor (`XGBRegressor`)**:

- **Key Predictive Features:**
  - `income`: Neighborhood median household income level.
  - `schools`: Number of quality educational institutions in the vicinity.
  - `hospitals`: Proximity to healthcare facilities.
  - `crime_rate`: Local neighborhood safety and crime statistics.
- **Target Variable:**
  - `price`: Final predicted valuation of the home.

---

## 📊 Pipeline Architecture

1. **Data Preprocessing & Structuring:** Formatted into tabular feature matrices using **Pandas** and **NumPy**.
2. **Train/Test Splitting:** Evaluated using `train_test_split` with an 80/20 train-test ratio and fixed random seed for reproducibility.
3. **Hyperparameter Tuning:** Configured with `n_estimators=100`, `learning_rate=0.1`, and `max_depth=3` to balance bias and variance.
4. **Model Evaluation:** Performance validated using Mean Absolute Error (MAE) and R² Coefficient of Determination.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Framework:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
- **Data Analysis:** Pandas, NumPy

---

## 🚀 Running the Project

```bash
# Clone the repository
git clone https://github.com/mounikay845-source/home-price-prediction.git
cd home-price-prediction

# Install requirements
pip install pandas numpy scikit-learn xgboost

# Run the training and prediction script
python home_price_prediction.py
```

---

## 👩‍💻 Author

**Mounika Yadav**
- GitHub: [@mounikay845-source](https://github.com/mounikay845-source)
- Email: `mounikay845@gmail.com`
