"""Flask backend — Athlete Recovery Score Prediction."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "athlete_recovery_model.joblib"

app = Flask(__name__, template_folder="templates", static_folder="static")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]


def _compute_engineered(raw: dict) -> dict:
    """Derive engineered features from raw user input."""
    dur = raw.get("Training_Duration_Min", 0) or 0
    intensity = raw.get("Training_Intensity", 0) or 0
    sleep = raw.get("Sleep_Duration_Hours", 7)
    hrv = raw.get("HRV_ms", 60)
    rhr = raw.get("Resting_Heart_Rate", 60)
    dow = raw.get("Day_of_Week", 1)

    raw["Training_Load"] = (dur / 60.0) * intensity
    raw["Sleep_Deficit"] = max(0, 8.0 - (sleep if sleep else 8.0))
    raw["HRV_RHR_Ratio"] = hrv / rhr if rhr else 0
    raw["Is_Weekend"] = 1 if dow >= 6 else 0
    return raw


def _score_to_level(score: float):
    if score >= 75:
        return {"label": "Çok Yüksek", "color": "#10b981", "emoji": "💪"}
    if score >= 55:
        return {"label": "Yüksek", "color": "#3b82f6", "emoji": "✅"}
    if score >= 35:
        return {"label": "Orta", "color": "#f59e0b", "emoji": "⚠️"}
    return {"label": "Düşük", "color": "#ef4444", "emoji": "🔴"}


def _factor_analysis(raw: dict, score: float):
    """Provide simple factor-level impact indicators."""
    factors = []

    load = raw.get("Training_Load", 0)
    if load > 8:
        factors.append({"name": "Antrenman Yükü", "level": "Yüksek", "color": "#ef4444", "icon": "🏋️"})
    elif load > 4:
        factors.append({"name": "Antrenman Yükü", "level": "Orta", "color": "#f59e0b", "icon": "🏋️"})
    else:
        factors.append({"name": "Antrenman Yükü", "level": "Düşük", "color": "#10b981", "icon": "🏋️"})

    soreness = raw.get("Muscle_Soreness", 5)
    if soreness >= 7:
        factors.append({"name": "Kas Ağrısı", "level": "Yüksek", "color": "#ef4444", "icon": "💢"})
    elif soreness >= 4:
        factors.append({"name": "Kas Ağrısı", "level": "Orta", "color": "#f59e0b", "icon": "💢"})
    else:
        factors.append({"name": "Kas Ağrısı", "level": "Düşük", "color": "#10b981", "icon": "💢"})

    sleep = raw.get("Sleep_Duration_Hours", 7)
    if sleep and sleep >= 7.5:
        factors.append({"name": "Uyku Kalitesi", "level": "İyi", "color": "#10b981", "icon": "😴"})
    elif sleep and sleep >= 6:
        factors.append({"name": "Uyku Kalitesi", "level": "Orta", "color": "#f59e0b", "icon": "😴"})
    else:
        factors.append({"name": "Uyku Kalitesi", "level": "Kötü", "color": "#ef4444", "icon": "😴"})

    stress = raw.get("Stress_Level", "Medium")
    stress_map = {"Low": ("Düşük", "#10b981"), "Medium": ("Orta", "#f59e0b"), "High": ("Yüksek", "#ef4444")}
    sl, sc = stress_map.get(stress, ("Orta", "#f59e0b"))
    factors.append({"name": "Stres Seviyesi", "level": sl, "color": sc, "icon": "🧠"})

    energy = raw.get("Energy_Level", 5)
    if energy >= 7:
        factors.append({"name": "Enerji Seviyesi", "level": "Yüksek", "color": "#10b981", "icon": "⚡"})
    elif energy >= 4:
        factors.append({"name": "Enerji Seviyesi", "level": "Orta", "color": "#f59e0b", "icon": "⚡"})
    else:
        factors.append({"name": "Enerji Seviyesi", "level": "Düşük", "color": "#ef4444", "icon": "⚡"})

    return factors


def _recommendations(raw: dict, score: float):
    tips = []
    sleep = raw.get("Sleep_Duration_Hours", 7)
    if sleep and sleep < 7:
        tips.append({"title": "Uyku Süresini Artırın", "desc": "Günlük 7-9 saat kaliteli uyku hedefleyin.", "icon": "🛌"})
    if raw.get("Training_Load", 0) > 8:
        tips.append({"title": "Aktif Dinlenme", "desc": "Düşük yoğunluklu aktivite öncelikli olmalı.", "icon": "🧘"})
    if raw.get("Stress_Level") == "High":
        tips.append({"title": "Stres Yönetimi", "desc": "Meditasyon veya nefes egzersizleri uygulayın.", "icon": "🌿"})
    if raw.get("Muscle_Soreness", 0) >= 6:
        tips.append({"title": "Kas Bakımı", "desc": "Foam roller ve soğuk-sıcak terapi uygulayın.", "icon": "💆"})
    if raw.get("Caffeine_Intake_mg", 0) > 300:
        tips.append({"title": "Kafein Azaltın", "desc": "Günlük 200-300mg altında tutmaya çalışın.", "icon": "☕"})
    if score < 40:
        tips.append({"title": "Tam Dinlenme Günü", "desc": "Vücut ciddi toparlanma gerektiriyor.", "icon": "🏠"})
    if not tips:
        tips.append({"title": "Harika Gidiyorsunuz!", "desc": "Mevcut rutininizi korumaya devam edin.", "icon": "🎯"})
    recovery_hours = max(12, int(60 - score * 0.5 + raw.get("Muscle_Soreness", 5) * 2))
    return tips, recovery_hours


@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/app")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    raw = {
        "Day": int(data.get("day", 1)),
        "Day_of_Week": int(data.get("day_of_week", 1)),
        "Week": int(data.get("week", 1)),
        "Age": int(data.get("age", 25)),
        "Gender": data.get("gender", "Male"),
        "Sport_Type": data.get("sport_type", "Team Sport"),
        "Training_Type": data.get("training_type", "Cardio"),
        "Training_Duration_Min": int(data.get("training_duration", 60)),
        "Training_Intensity": float(data.get("training_intensity", 5)),
        "Sleep_Duration_Hours": float(data.get("sleep_duration", 7)),
        "Caffeine_Intake_mg": int(data.get("caffeine", 150)),
        "Stress_Level": data.get("stress_level", "Medium"),
        "Resting_Heart_Rate": int(data.get("resting_hr", 62)),
        "HRV_ms": int(data.get("hrv", 65)),
        "Mood_Score": float(data.get("mood", 5)),
        "Muscle_Soreness": float(data.get("muscle_soreness", 4)),
        "Energy_Level": float(data.get("energy_level", 5)),
    }

    raw = _compute_engineered(raw)

    row = {col: raw.get(col, 0) for col in feature_columns}
    df = pd.DataFrame([row])
    score = float(np.clip(pipeline.predict(df)[0], 0, 100))
    score = round(score, 1)

    level = _score_to_level(score)
    factors = _factor_analysis(raw, score)
    tips, recovery_hours = _recommendations(raw, score)

    return jsonify({
        "score": score,
        "level": level,
        "factors": factors,
        "recommendations": tips,
        "recovery_hours": f"{recovery_hours} - {recovery_hours + 12} saat",
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
