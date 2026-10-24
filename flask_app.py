"""Heart Disease Prediction (4 inputs) - Flask app for PythonAnywhere."""
import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
art = joblib.load(os.path.join(BASE, "heart_model.pkl"))
model, enc, FEATURES = art["model"], art["encoders"], art["features"]

app = Flask(__name__)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heart Disease Predictor</title>
<style>
 body{font-family:Arial,sans-serif;background:#5c1a1b;margin:0;padding:20px;color:#222}
 .card{max-width:480px;margin:auto;background:#fff;border-radius:12px;padding:24px}
 h2{margin-top:0;color:#5c1a1b} label{display:block;margin-top:12px;font-weight:bold}
 input,select{width:100%;padding:8px;margin-top:4px;box-sizing:border-box}
 button{margin-top:18px;width:100%;padding:12px;background:#a4161a;color:#fff;border:0;border-radius:6px;font-size:16px;cursor:pointer}
 .res{margin-top:18px;padding:14px;border-radius:8px;font-size:18px;text-align:center}
 .ok{background:#d4edda;color:#155724}.no{background:#f8d7da;color:#721c24}
 small.n{display:block;margin-top:14px;color:#777}
</style></head><body><div class="card">
<h2>❤️ Heart Disease Predictor</h2>
<form method="post">
 <label>Sex</label>
 <select name="sex">{% for v in sexes %}<option {{'selected' if f.get('sex')==v}}>{{v}}</option>{% endfor %}</select>
 <label>Chest-pain type</label>
 <select name="cp">{% for v in cps %}<option {{'selected' if f.get('cp')==v}}>{{v}}</option>{% endfor %}</select>
 <label>Maximum heart rate achieved (bpm)</label>
 <input type="number" min="60" max="220" name="maxhr" value="{{f.get('maxhr','150')}}" required>
 <label>ST depression (Oldpeak)</label>
 <input type="number" step="0.1" min="0" max="7" name="oldpeak" value="{{f.get('oldpeak','1.0')}}" required>
 <button type="submit">Predict</button>
</form>
{% if result %}<div class="res {{cls}}">{{result}}<br><small>Estimated risk: {{prob}}%</small></div>{% endif %}
<small class="n">Educational project – not a medical diagnosis tool.</small>
</div></body></html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    ctx = dict(f=request.form, sexes=list(enc["Sex"].classes_), cps=list(enc["ChestPain"].classes_),
               result=None, cls="", prob=None)
    if request.method == "POST":
        row = pd.DataFrame([{
            "Sex": enc["Sex"].transform([request.form["sex"]])[0],
            "ChestPain": enc["ChestPain"].transform([request.form["cp"]])[0],
            "MaxHR": float(request.form["maxhr"]),
            "Oldpeak": float(request.form["oldpeak"]),
        }])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        sick = p >= 0.5
        ctx.update(result="⚠️ Heart disease LIKELY" if sick else "✅ Heart disease UNLIKELY",
                   cls="no" if sick else "ok", prob=round(p * 100, 1))
    return render_template_string(PAGE, **ctx)

if __name__ == "__main__":
    app.run(debug=True)
