import warnings
warnings.filterwarnings('ignore')
import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load models
model_seat = joblib.load(os.path.join(BASE_DIR, "Models", "model_seat.sav"))
model_backrest = joblib.load(os.path.join(BASE_DIR, "Models", "model_backrest.sav"))
model_floor = joblib.load(os.path.join(BASE_DIR, "Models", "model_floor.sav"))

# Encoders
suspension_encoder = LabelEncoder()
suspension_encoder.fit(['soft','medium','hard'])

soil_encoder = LabelEncoder()
soil_encoder.fit(['clay','loam','sand','silt'])

def comfort_level(v):

    if v < 0.5:
        return "Comfortable"

    elif v < 0.8:
        return "Moderate"

    else:
        return "Uncomfortable"


def recommend_settings(speed, depth):

    rec_speed = round(speed * 0.9,2)
    rec_depth = round(depth * 0.85,2)

    return rec_speed, rec_depth


@app.route("/", methods=["GET","POST"])
def index():

    result = None
    speed_curve = None
    surface_data = None
    recommendation = None

    if request.method == "POST":

        data = {
            "suspension": request.form["suspension"],
            "speed_m_s": float(request.form["speed"]),
            "tire_pressure_kPa": float(request.form["tire_pressure"]),
            "wheelbase_m": float(request.form["wheelbase"]),
            "soil_type": request.form["soil"],
            "soil_moisture_pct": float(request.form["moisture"]),
            "tractive_force_kN": float(request.form["traction"]),
            "tool_depth_m": float(request.form["depth"])
        }

        df = pd.DataFrame([data])

        df["suspension"] = suspension_encoder.transform(df["suspension"])
        df["soil_type"] = soil_encoder.transform(df["soil_type"])

        seat = model_seat.predict(df)[0]
        backrest = model_backrest.predict(df)[0]
        floor = model_floor.predict(df)[0]

        result = {
            "seat": round(seat,3),
            "backrest": round(backrest,3),
            "floor": round(floor,3),
            "seat_level": comfort_level(seat),
            "backrest_level": comfort_level(backrest),
            "floor_level": comfort_level(floor)
        }

        # Speed vs vibration curve
        speeds = np.linspace(0.5,3,20)
        vib_values = []

        for s in speeds:

            temp = df.copy()
            temp["speed_m_s"] = s

            vib = model_seat.predict(temp)[0]
            vib_values.append(vib)

        speed_curve = {
            "speed": speeds.tolist(),
            "vibration": vib_values
        }

        # Surface plot data
        speed_range = np.linspace(0.5,3,20)
        depth_range = np.linspace(0.05,0.3,20)

        Z = []

        for s in speed_range:

            row = []

            for d in depth_range:

                temp = df.copy()
                temp["speed_m_s"] = s
                temp["tool_depth_m"] = d

                vib = model_seat.predict(temp)[0]
                row.append(vib)

            Z.append(row)

        surface_data = {
            "speed": speed_range.tolist(),
            "depth": depth_range.tolist(),
            "z": Z
        }

        # Recommendation
        rec_speed, rec_depth = recommend_settings(data["speed_m_s"], data["tool_depth_m"])

        recommendation = {
            "speed": rec_speed,
            "depth": rec_depth
        }

    return render_template(
        "index.html",
        result=result,
        speed_curve=speed_curve,
        surface_data=surface_data,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run()