"""
export_routes.py
FIX: ExcelWriter with-block indentation was broken — group_summary
     was written OUTSIDE the with-block (file already closed).
FIX: Updated column names to match new schema (bmi, hba1c_baseline).
"""

from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO

export_bp = Blueprint("export", __name__)


@export_bp.route("/export/csv", methods=["POST"])
def export_csv():
    try:
        data     = request.json
        patients = data.get("patients", [])
        filename = data.get("filename", "synthetic_patients.csv")

        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df   = pd.DataFrame(patients)
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        df.to_csv(temp.name, index=False)
        temp.close()

        return send_file(temp.name, as_attachment=True,
                         download_name=filename, mimetype="text/csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@export_bp.route("/export/excel", methods=["POST"])
def export_excel():
    try:
        data     = request.json
        patients = data.get("patients", [])
        stats    = data.get("statistics", {})

        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df   = pd.DataFrame(patients)
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")

        # ── ALL writes must happen INSIDE the with-block ──────
        with pd.ExcelWriter(temp.name, engine="openpyxl") as writer:
            # Sheet 1: Patient data
            df.to_excel(writer, sheet_name="Patients", index=False)

            # Sheet 2: Statistics
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name="Statistics", index=False)

            # Sheet 3: Group summary
            # Detect column names (handles both old and new schema)
            bmi_col     = "bmi"       if "bmi"       in df.columns else "BMI"
            base_col    = "hba1c_baseline" if "hba1c_baseline" in df.columns else "baseline_hba1c"
            h12_col     = "hba1c_12_months" if "hba1c_12_months" in df.columns else "hba1c_12m"
            glucose_col = "fasting_glucose_mg_dl"

            if "treatment_group" in df.columns:
                agg_dict = {"age": "mean"}
                if bmi_col in df.columns:
                    agg_dict[bmi_col] = "mean"
                if base_col in df.columns:
                    agg_dict[base_col] = "mean"
                if h12_col in df.columns:
                    agg_dict[h12_col] = "mean"
                if glucose_col in df.columns:
                    agg_dict[glucose_col] = "mean"

                summary = df.groupby("treatment_group").agg(agg_dict).round(2)
                summary.to_excel(writer, sheet_name="Group_Summary")  # inside with-block

        return send_file(
            temp.name,
            as_attachment=True,
            download_name="clinical_trial_data.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@export_bp.route("/visualize/distribution/<column>", methods=["POST"])
def get_distribution(column):
    try:
        data     = request.json
        patients = data.get("patients", [])
        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df = pd.DataFrame(patients)
        if column not in df.columns:
            return jsonify({"error": f"Column {column} not found"}), 400

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        hist, bins = np.histogram(values, bins=20)

        return jsonify({
            "column":   column,
            "count":    len(values),
            "histogram": hist.tolist(),
            "bins":     bins.tolist(),
            "mean":     float(values.mean()),
            "median":   float(values.median()),
            "std":      float(values.std()),
            "min":      float(values.min()),
            "max":      float(values.max()),
            "q1":       float(values.quantile(0.25)),
            "q3":       float(values.quantile(0.75)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@export_bp.route("/visualize/comparison", methods=["POST"])
def get_comparison():
    try:
        data     = request.json
        patients = data.get("patients", [])
        metric   = data.get("metric", "hba1c_12_months")
        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df = pd.DataFrame(patients)
        if "treatment_group" not in df.columns:
            return jsonify({"error": "No treatment_group column"}), 400
        if metric not in df.columns:
            return jsonify({"error": f"Metric {metric} not found"}), 400

        treatment = pd.to_numeric(
            df[df["treatment_group"] == "Treatment"][metric], errors="coerce").dropna()
        control   = pd.to_numeric(
            df[df["treatment_group"] == "Control"][metric],   errors="coerce").dropna()

        if len(treatment) == 0 or len(control) == 0:
            return jsonify({"error": "Insufficient data for comparison"}), 400

        all_vals = pd.concat([treatment, control])
        bins     = np.histogram_bin_edges(all_vals, bins=15)
        t_hist, _ = np.histogram(treatment, bins=bins)
        c_hist, _ = np.histogram(control,   bins=bins)

        return jsonify({
            "metric": metric,
            "bins":   bins.tolist(),
            "treatment": {
                "histogram": t_hist.tolist(),
                "mean":   float(treatment.mean()),
                "median": float(treatment.median()),
                "std":    float(treatment.std()),
                "count":  len(treatment),
            },
            "control": {
                "histogram": c_hist.tolist(),
                "mean":   float(control.mean()),
                "median": float(control.median()),
                "std":    float(control.std()),
                "count":  len(control),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@export_bp.route("/visualize/timeline", methods=["POST"])
def get_timeline():
    try:
        data     = request.json
        patients = data.get("patients", [])
        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df = pd.DataFrame(patients)
        timepoints   = [3, 6, 12, 24, 36]
        result = {"timepoints": timepoints, "hba1c": {}}

        for m in timepoints:
            col = f"hba1c_{m}_months"
            if col in df.columns:
                trt_mean  = df[df["treatment_group"] == "Treatment"][col].mean()
                ctrl_mean = df[df["treatment_group"] == "Control"][col].mean()
                result["hba1c"][str(m)] = {
                    "treatment": float(trt_mean)  if not np.isnan(trt_mean)  else None,
                    "control":   float(ctrl_mean) if not np.isnan(ctrl_mean) else None,
                }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@export_bp.route("/visualize/chart/<chart_type>", methods=["POST"])
def get_chart_image(chart_type):
    try:
        data     = request.json
        patients = data.get("patients", [])
        if not patients:
            return jsonify({"error": "No patient data provided"}), 400

        df      = pd.DataFrame(patients)
        bmi_col = "bmi" if "bmi" in df.columns else "BMI"
        base_col = "hba1c_baseline" if "hba1c_baseline" in df.columns else "baseline_hba1c"

        plt.figure(figsize=(10, 6))

        if chart_type == "hba1c_distribution":
            if "treatment_group" in df.columns and base_col in df.columns:
                trt_vals  = pd.to_numeric(df[df["treatment_group"] == "Treatment"][base_col], errors="coerce").dropna()
                ctrl_vals = pd.to_numeric(df[df["treatment_group"] == "Control"][base_col],   errors="coerce").dropna()
                plt.hist([trt_vals, ctrl_vals], bins=15,
                         label=["Treatment", "Control"], alpha=0.7)
                plt.xlabel("HbA1c (%)"); plt.ylabel("Patients")
                plt.title("HbA1c Distribution by Treatment Group"); plt.legend()

        elif chart_type == "bmi_vs_hba1c":
            if bmi_col in df.columns and base_col in df.columns:
                colors = {"Treatment": "steelblue", "Control": "tomato"}
                for grp in ["Treatment", "Control"]:
                    sub = df[df["treatment_group"] == grp]
                    plt.scatter(
                        pd.to_numeric(sub[bmi_col], errors="coerce"),
                        pd.to_numeric(sub[base_col], errors="coerce"),
                        label=grp, alpha=0.6, c=colors.get(grp, "gray"), s=20)
                plt.xlabel("BMI (kg/m²)"); plt.ylabel("HbA1c (%)")
                plt.title("BMI vs HbA1c by Treatment Group"); plt.legend()

        elif chart_type == "age_distribution":
            if "age" in df.columns:
                plt.hist(pd.to_numeric(df["age"], errors="coerce").dropna(),
                         bins=20, edgecolor="black", alpha=0.7)
                plt.xlabel("Age (years)"); plt.ylabel("Patients")
                plt.title("Age Distribution of Synthetic Cohort")

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(); buf.seek(0)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({"success": True, "chart_type": chart_type,
                        "image": image_b64, "format": "png"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500