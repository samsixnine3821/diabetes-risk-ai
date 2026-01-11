from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
from io import BytesIO
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import green, orange, red
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

app = Flask(__name__)

# Load trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)


def get_float(form, key, default):
    try:
        return float(form.get(key, default))
    except ValueError:
        return default


@app.route("/", methods=["GET", "POST"])
def index():
    result = explanation = advice = ""
    risk_colour = "green"
    risk_percent = 0
    inputs = {}

    if request.method == "POST":
        glucose = get_float(request.form, "glucose", 100)
        blood_pressure = get_float(request.form, "blood_pressure", 70)
        bmi = get_float(request.form, "bmi", 25)
        age = get_float(request.form, "age", 30)

        inputs = {
            "Glucose (mg/dL)": glucose,
            "Blood Pressure (mm Hg)": blood_pressure,
            "BMI": bmi,
            "Age": age
        }

        # Defaults for removed features
        pregnancies = 0
        skin_thickness = 20
        insulin = 80
        dpf = 0.5

        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]])

        probability = model.predict_proba(input_data)[0][1]
        risk_percent = round(probability * 100, 1)

        if probability < 0.3:
            risk_colour = "green"
            result = f"Low diabetes risk ({risk_percent}%)"
        elif probability < 0.6:
            risk_colour = "orange"
            result = f"Moderate diabetes risk ({risk_percent}%)"
        else:
            risk_colour = "red"
            result = f"High diabetes risk ({risk_percent}%)"

        reasons = []
        if glucose > 140:
            reasons.append("elevated glucose")
        if bmi > 30:
            reasons.append("high BMI")
        if age > 45:
            reasons.append("older age")

        explanation = (
            "Main contributing factors: " + ", ".join(reasons)
            if reasons else
            "No major risk factors detected."
        )

        advice = (
            "Reduce sugar intake, maintain healthy weight, and exercise regularly."
            if probability >= 0.3 else
            "Maintain current healthy lifestyle."
        )

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        advice=advice,
        risk_colour=risk_colour,
        risk_percent=risk_percent,
        inputs=inputs
    )


def add_page_number(canvas, doc):
    page_num_text = f"Page {doc.page}"
    canvas.drawRightString(570, 20, page_num_text)


@app.route("/report", methods=["POST"])
def report():
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )

    content = []

    # ===== COVER PAGE =====
    content.append(Spacer(1, 150))
    content.append(Paragraph(
        "<b>AI Diabetes Risk Assessment</b>", styles["Title"]
    ))
    content.append(Spacer(1, 20))
    content.append(Paragraph(
        "Personalised Health Risk Report", styles["Heading2"]
    ))
    content.append(Spacer(1, 40))
    content.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    content.append(Spacer(1, 20))
    content.append(Paragraph(
        "<i>Created by Sam</i>", styles["Italic"]
    ))

    content.append(PageBreak())

    # ===== SUMMARY =====
    content.append(Paragraph("Risk Summary", styles["Heading1"]))
    content.append(Spacer(1, 10))

    risk_text = request.form["result"]
    risk_color = (
        green if "Low" in risk_text else
        orange if "Moderate" in risk_text else
        red
    )

    content.append(Paragraph(
        f"<font color='{risk_color.hexval()}'><b>{risk_text}</b></font>",
        styles["Normal"]
    ))
    content.append(Spacer(1, 10))
    content.append(Paragraph(
        f"Estimated risk probability: {request.form['risk_percent']}%",
        styles["Normal"]
    ))

    # ===== INPUT SUMMARY =====
    content.append(Spacer(1, 20))
    content.append(Paragraph("User Input Summary", styles["Heading2"]))

    for key, value in request.form.items():
        if key not in ["result", "risk_percent", "explanation", "advice"]:
            content.append(Paragraph(
                f"{key.replace('_',' ').title()}: {value}",
                styles["Normal"]
            ))

    # ===== EXPLANATION =====
    content.append(Spacer(1, 20))
    content.append(Paragraph("Explanation", styles["Heading2"]))
    content.append(Paragraph(
        request.form["explanation"],
        styles["Normal"]
    ))

    # ===== ADVICE =====
    content.append(Spacer(1, 20))
    content.append(Paragraph("Risk Reduction Advice", styles["Heading2"]))
    content.append(Paragraph(
        request.form["advice"],
        styles["Normal"]
    ))

    # ===== CHART =====
    content.append(Spacer(1, 30))
    content.append(Paragraph("Risk Visualisation", styles["Heading2"]))

    drawing = Drawing(400, 200)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 30
    chart.height = 150
    chart.width = 300
    chart.data = [[float(request.form["risk_percent"])]]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.categoryAxis.categoryNames = ["Risk %"]

    chart.bars[0].fillColor = risk_color
    drawing.add(chart)
    content.append(drawing)

    doc.build(content, onLaterPages=add_page_number)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="diabetes_risk_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


