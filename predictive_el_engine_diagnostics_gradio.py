import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
)
from fpdf import FPDF
import tempfile
import os

# Load column names from uploaded CSV
def load_column_names(csv_file):
    df = pd.read_csv(csv_file.name)
    columns = df.columns.tolist()
    return gr.update(choices=columns, value=columns[:3]), gr.update(choices=columns, value=columns[-1])

# Generate predictive report based on selected columns
def generate_report(csv_file, input_cols, output_col):
    df = pd.read_csv(csv_file.name)

    # Convert object-type columns to datetime if possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue

    # Drop datetime columns
    datetime_cols = df.select_dtypes(include='datetime').columns
    df.drop(columns=datetime_cols, inplace=True)

    # Target column
    df['target'] = df[output_col].shift(-1).fillna(0).astype(int)

    # Keep only numeric input columns
    numeric_input_cols = [col for col in input_cols if pd.api.types.is_numeric_dtype(df[col])]
    skipped_cols = list(set(input_cols) - set(numeric_input_cols))

    # Prepare features and target
    X = df[numeric_input_cols]
    y = df['target']

    # Train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Temp folder
    temp_dir = tempfile.mkdtemp()
    cm_path = os.path.join(temp_dir, "confusion_matrix.png")
    roc_path = os.path.join(temp_dir, "roc_curve.png")
    fi_path = os.path.join(temp_dir, "feature_importance.png")
    report_csv_path = os.path.join(temp_dir, "classification_report.csv")
    pdf_path = os.path.join(temp_dir, "predictive_report.pdf")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    pd.Series(importances, index=numeric_input_cols).sort_values(ascending=False).plot(kind='bar', figsize=(8, 6))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(report_csv_path)

    # PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Predictive Maintenance Report", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Accuracy: {accuracy_score(y_test, y_pred):.2%}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Confusion Matrix", ln=True)
    pdf.image(cm_path, w=160)
    pdf.ln(5)
    pdf.cell(0, 10, "ROC Curve", ln=True)
    pdf.image(roc_path, w=160)
    pdf.ln(5)
    pdf.cell(0, 10, "Feature Importance", ln=True)
    pdf.image(fi_path, w=160)
    pdf.output(pdf_path)

    return (
        cm_path, roc_path, fi_path,
        report_csv_path, pdf_path,
        f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}",
        f"✔ Used Inputs: {', '.join(numeric_input_cols)} | Output: {output_col}" +
        (f"\n⚠️ Skipped non-numeric: {', '.join(skipped_cols)}" if skipped_cols else "")
    )

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        file_input = gr.File(label="Upload CSV")
        load_button = gr.Button("Load Column Options")

    input_selector = gr.CheckboxGroup(label="Select Input Columns", choices=[], interactive=True)
    output_selector = gr.Dropdown(label="Select Output Column", choices=[], interactive=True)

    load_button.click(
        load_column_names,
        inputs=file_input,
        outputs=[input_selector, output_selector]
    )

    with gr.Row():
        run_button = gr.Button("Generate Report")

    with gr.Row():
        cm_image = gr.Image(label="Confusion Matrix")
        roc_image = gr.Image(label="ROC Curve")
        fi_image = gr.Image(label="Feature Importance")

    with gr.Row():
        csv_out = gr.File(label="Download Report CSV")
        pdf_out = gr.File(label="Download PDF Report")

    with gr.Row():
        acc_box = gr.Textbox(label="Accuracy")
        summary_box = gr.Textbox(label="Input/Output Summary")

    run_button.click(
        generate_report,
        inputs=[file_input, input_selector, output_selector],
        outputs=[
            cm_image, roc_image, fi_image,
            csv_out, pdf_out,
            acc_box, summary_box
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)


