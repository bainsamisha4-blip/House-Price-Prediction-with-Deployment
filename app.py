"""Streamlit web app for real-time house price predictions.
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

def train_fallback():
    """Train the model from the CSV if the saved .pkl can't be loaded
    (e.g. scikit-learn version mismatch on the hosting server)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv("data/house_prices.csv")
    X, y = df.drop(columns="price"), df["price"]
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42))])
    return pipe.fit(X, y)


@st.cache_resource
def load_model():
    try:
        return joblib.load("models/house_price_model.pkl")
    except Exception:
        return train_fallback()

model = load_model()

st.title("🏠 House Price Predictor")
st.caption("Enter the property details to get an instant price estimate.")

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", 500, 5000, 1800, step=50)
    bedrooms = st.slider("Bedrooms", 1, 7, 3)
    bathrooms = st.slider("Bathrooms", 1, 6, 2)
    floors = st.selectbox("Floors", [1, 2, 3], index=1)
    garage = st.slider("Garage spaces", 0, 3, 2)
with col2:
    age = st.slider("Age of house (years)", 0, 60, 10)
    dist = st.number_input("Distance to city centre (km)", 0.5, 60.0, 10.0, step=0.5)
    location = st.selectbox("Location", ["Downtown", "Suburban", "Rural", "Waterfront"], index=1)
    condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"], index=2)
    pool = st.checkbox("Swimming pool")

if st.button("Predict price", type="primary"):
    x = pd.DataFrame([{
        "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "floors": floors,
        "age_years": age, "garage_spaces": garage, "has_pool": int(pool),
        "location": location, "condition": condition, "distance_to_city_km": dist,
    }])
    price = model.predict(x)[0]
    st.success(f"Estimated price: **${price:,.0f}**")
    st.caption(f"Typical error range: about ±$19,000 (model MAE on test data).")
