# 🏠 House Price Prediction with Deployment

A machine learning project that predicts house prices from property features and serves real-time predictions through an interactive **Streamlit** web app.

🔗 **Live App:** _add your Streamlit link here_

## 🛠️ Technologies Used
- **Python** – core language
- **Pandas, NumPy** – data handling
- **Matplotlib, Seaborn** – visualisation
- **Scikit-learn** – preprocessing, model training, tuning
- **Joblib** – saving the trained model
- **Streamlit** – web app for real-time predictions
- **Jupyter / Google Colab** – development
- **Git & GitHub** – version control

## 📊 Dataset
3,000 records with 10 features (area, bedrooms, bathrooms, floors, age, garage spaces, pool, location, condition, distance to city) and the target `price`.
The dataset is synthetic, generated using `generate_data.py`.

## 🔄 What I Did
1. Cleaned the data and handled missing values
2. Explored the data with plots (distribution, correlation, price vs. features)
3. Built a preprocessing pipeline (imputation, scaling, one-hot encoding)
4. Compared Linear Regression, Ridge, Random Forest and Gradient Boosting
5. Tuned the best model (Gradient Boosting) with `RandomizedSearchCV`
6. Saved the model and built a Streamlit app around it

## 🏆 Results
| Metric | Score |
|--------|-------|
| R² | ≈ 0.97 |
| MAE | ≈ $18,700 |
| RMSE | ≈ $24,900 |

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model, open `notebooks/house_price_prediction.ipynb` and run all cells.

## 📁 Project Structure
```
├── data/house_prices.csv
├── notebooks/house_price_prediction.ipynb
├── models/house_price_model.pkl
├── app.py
├── generate_data.py
└── requirements.txt
```

## 👤 Author
 Amisha Dei
 student of M.SC. Data science
 GitHub : https://github.com/bainsamisha4-blip
