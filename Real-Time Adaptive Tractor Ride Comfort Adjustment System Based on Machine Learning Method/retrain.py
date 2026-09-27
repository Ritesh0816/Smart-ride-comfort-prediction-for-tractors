import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

# Load data
df = pd.read_csv("vibration_dataset.csv")

# Encode categorical features
categorical_cols = ['suspension', 'soil_type']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and targets
X = df[['suspension', 'speed_m_s', 'tire_pressure_kPa', 'wheelbase_m', 'soil_type', 'soil_moisture_pct', 'tractive_force_kN', 'tool_depth_m']]
y_seat = df['vibration_seat']
y_backrest = df['vibration_backrest']
y_floor = df['vibration_floor']

# Split data
X_train, X_test, y_train_seat, y_test_seat = train_test_split(X, y_seat, test_size=0.2, random_state=42)
_, _, y_train_backrest, _ = train_test_split(X, y_backrest, test_size=0.2, random_state=42)
_, _, y_train_floor, _ = train_test_split(X, y_floor, test_size=0.2, random_state=42)

# Define models
xgb = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

gbr = GradientBoostingRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)

lgbm = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

cat = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_state=42
)

# Train and save for seat
voting_seat = VotingRegressor(
    estimators=[
        ('xgb', xgb),
        ('gbr', gbr),
        ('lgbm', lgbm),
        ('cat', cat)
    ],
    weights=[3, 1, 3, 2]
)

voting_seat.fit(X_train, y_train_seat)
joblib.dump(voting_seat, "Models/model_seat.sav")

# Train and save for backrest
voting_backrest = VotingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=1200, learning_rate=0.03, max_depth=5, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
        ('cat', CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=6, loss_function='RMSE', verbose=0, random_state=42))
    ],
    weights=[3, 1, 3, 2]
)

voting_backrest.fit(X_train, y_train_backrest)
joblib.dump(voting_backrest, "Models/model_backrest.sav")

# Train and save for floor
voting_floor = VotingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=1200, learning_rate=0.03, max_depth=5, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
        ('cat', CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=6, loss_function='RMSE', verbose=0, random_state=42))
    ],
    weights=[3, 1, 3, 2]
)

voting_floor.fit(X_train, y_train_floor)
joblib.dump(voting_floor, "Models/model_floor.sav")

print("Models retrained and saved.")