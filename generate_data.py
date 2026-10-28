"""Generate a realistic synthetic housing dataset (data/house_prices.csv)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 3000

area = rng.normal(1800, 650, N).clip(500, 5000).round()
bedrooms = np.clip((area / 700 + rng.normal(0, 0.8, N)).round(), 1, 7).astype(int)
bathrooms = np.clip((bedrooms * 0.7 + rng.normal(0.5, 0.6, N)).round(), 1, 6).astype(int)
floors = rng.choice([1, 2, 3], N, p=[0.5, 0.4, 0.1])
age = rng.integers(0, 60, N)
garage = rng.choice([0, 1, 2, 3], N, p=[0.1, 0.35, 0.45, 0.1])
has_pool = rng.choice([0, 1], N, p=[0.85, 0.15])
location = rng.choice(["Downtown", "Suburban", "Rural", "Waterfront"], N, p=[0.25, 0.45, 0.2, 0.1])
condition = rng.choice(["Poor", "Fair", "Good", "Excellent"], N, p=[0.08, 0.25, 0.45, 0.22])
dist = np.where(location == "Downtown", rng.uniform(0.5, 5, N),
        np.where(location == "Suburban", rng.uniform(5, 20, N),
        np.where(location == "Rural", rng.uniform(20, 60, N), rng.uniform(3, 25, N)))).round(1)

loc_mult = pd.Series(location).map({"Downtown": 1.25, "Suburban": 1.0, "Rural": 0.75, "Waterfront": 1.6}).values
cond_mult = pd.Series(condition).map({"Poor": 0.8, "Fair": 0.92, "Good": 1.0, "Excellent": 1.12}).values

base = 50_000 + 130 * area + 8_000 * bedrooms + 12_000 * bathrooms + 9_000 * floors \
       + 7_500 * garage + 25_000 * has_pool - 900 * age - 1_200 * dist
price = base * loc_mult * cond_mult * rng.normal(1, 0.06, N)

df = pd.DataFrame(dict(area_sqft=area.astype(int), bedrooms=bedrooms, bathrooms=bathrooms,
                       floors=floors, age_years=age, garage_spaces=garage, has_pool=has_pool,
                       location=location, condition=condition, distance_to_city_km=dist,
                       price=price.round(-2).astype(int)))
# inject a few missing values to show cleaning
for col in ["age_years", "distance_to_city_km"]:
    df.loc[rng.choice(N, 40, replace=False), col] = np.nan
df.to_csv("data/house_prices.csv", index=False)
print(df.shape); print(df.head())
