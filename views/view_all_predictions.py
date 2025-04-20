import streamlit as st
import pandas as pd
import plotly.express as px

# Custom CSS for professional styling
st.markdown("""
    <style>
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1e88e5;
        margin-bottom: 10px;
    }
    .sub-section {
        background-color: white;
        padding: 10px;
        border: 0;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1e88e5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        color: white;
    }
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin-top: 10px;
    }
    .report-table th, .report-table td {
        border: 1px solid #e0e0e0;
        padding: 8px;
        text-align: right;
    }
    .report-table th {
        background-color: #1e88e5;
        color: white;
        font-weight: 600;
    }
    .report-table td:first-child {
        text-align: left;
        font-weight: 500;
    }
    .report-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .report-table tr.accuracy-row {
        background-color: #e3f2fd;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def parse_classification_report(report_str):
    """Parse sklearn classification report string into a DataFrame."""
    lines = report_str.strip().split('\n')
    data = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']

    # Skip header line and process data lines
    for line in lines[2:]:  # Skip first two lines (header and separator)
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                # Handle class labels with spaces or special cases
                support = parts[-1]
                f1 = parts[-2]
                recall = parts[-3]
                precision = parts[-4]
                class_label = ' '.join(parts[:-4]) if len(parts) > 5 else parts[0]
                if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                    if class_label == 'accuracy':
                        # Accuracy row has fewer columns
                        data.append([class_label, '', '', parts[-2], support])
                    else:
                        data.append([class_label, precision, recall, f1, support])
                else:
                    data.append([class_label, precision, recall, f1, support])

    return pd.DataFrame(data, columns=headers)


def predictions_page():
    # Main container
    st.markdown('<h1 class="section-header">Predictions and Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown(
        "Review dropout risk, performance, and engagement predictions, and compare model performance with and without "
        "sentiment data.",
        unsafe_allow_html=True)

    # Check if necessary data is available in session state
    if "integrated_acc" not in st.session_state:
        st.error("Please run the analysis on the Overview page first to generate predictions and comparison results.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Load the prediction CSV files
    try:
        integrated_predictions = pd.read_csv("data/integrated_data_predictions.csv")
        analytics_predictions = pd.read_csv("data/analytics_data_predictions.csv")
    except FileNotFoundError as e:
        st.error(f"Prediction files not found: {e}. Please ensure the analysis has been run on the Overview page.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Retrieve comparison results from session state
    integrated_acc = st.session_state["integrated_acc"]
    analytics_acc = st.session_state["analytics_acc"]
    integrated_report = st.session_state["integrated_report"]
    analytics_report = st.session_state["analytics_report"]
    integrated_cv = st.session_state["integrated_cv"]
    analytics_cv = st.session_state["analytics_cv"]

    # Predictions Section
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Predictions</h2>', unsafe_allow_html=True)

    # Integrated Data Predictions
    st.markdown("#### Integrated Data Predictions (With Sentiment)")
    st.markdown("Dropout risk, performance, and engagement predictions using both analytics and sentiment data.")
    st.dataframe(
        integrated_predictions.head(),
        use_container_width=True,
        column_config={"_index": None}  # Hide index
    )

    # Analytics-Only Data Predictions
    st.markdown("#### Analytics-Only Data Predictions (Without Sentiment)")
    st.markdown("Predictions based solely on analytics data, excluding sentiment.")
    st.dataframe(
        analytics_predictions.head(),
        use_container_width=True,
        column_config={"_index": None}
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Comparison Section
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Model Comparison (Dropout Prediction)</h2>', unsafe_allow_html=True)
    st.markdown("Compare the performance of dropout prediction models with and without sentiment data.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Integrated (With Sentiment)**")
        st.markdown(f'<div class="metric-box"><b>Test Accuracy:</b> {integrated_acc:.4f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Cross-Validation Accuracy:</b> {integrated_cv:.4f}</div>',
                    unsafe_allow_html=True)
        st.markdown("**Classification Report:**")
        try:
            report_df = parse_classification_report(integrated_report)
            report_df[['Precision', 'Recall', 'F1-Score']] = report_df[['Precision', 'Recall', 'F1-Score']].apply(
                pd.to_numeric, errors='coerce')

            # Convert DataFrame to styled HTML
            html = report_df.to_html(
                index=False,
                classes="report-table",
                escape=False,
                formatters={
                    'Precision': lambda x: f"{x:.2f}" if pd.notnull(x) else "",
                    'Recall': lambda x: f"{x:.2f}" if pd.notnull(x) else "",
                    'F1-Score': lambda x: f"{x:.2f}" if pd.notnull(x) else ""
                }
            )
            # Add accuracy row styling
            html = html.replace('<tr>\n<td>accuracy</td>', '<tr class="accuracy-row">\n<td>accuracy</td>')
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not parse Integrated classification report: {str(e)}. Displaying raw text.")
            st.text_area("", integrated_report, height=200, disabled=True)

    with col2:
        st.markdown("**Analytics-Only (Without Sentiment)**")
        st.markdown(f'<div class="metric-box"><b>Test Accuracy:</b> {analytics_acc:.4f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box"><b>Cross-Validation Accuracy:</b> {analytics_cv:.4f}</div>',
                    unsafe_allow_html=True)
        st.markdown("**Classification Report:**")
        try:
            report_df = parse_classification_report(analytics_report)
            report_df[['Precision', 'Recall', 'F1-Score']] = report_df[['Precision', 'Recall', 'F1-Score']].apply(
                pd.to_numeric, errors='coerce')

            html = report_df.to_html(
                index=False,
                classes="report-table",
                escape=False,
                formatters={
                    'Precision': lambda x: f"{x:.2f}" if pd.notnull(x) else "",
                    'Recall': lambda x: f"{x:.2f}" if pd.notnull(x) else "",
                    'F1-Score': lambda x: f"{x:.2f}" if pd.notnull(x) else ""
                }
            )
            html = html.replace('<tr>\n<td>accuracy</td>', '<tr class="accuracy-row">\n<td>accuracy</td>')
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not parse Analytics-Only classification report: {str(e)}. Displaying raw text.")
            st.text_area("", analytics_report, height=200, disabled=True)

    # Visual Comparison
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown("#### Model Performance Comparison")
    st.markdown("Interactive bar chart comparing test accuracies of the two models. Hover to see details.")
    # Create Plotly bar chart
    df = pd.DataFrame({
        "Model": ["Integrated", "Analytics-Only"],
        "Accuracy": [integrated_acc, analytics_acc]
    })
    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        color="Model",
        color_discrete_map={"Integrated": "#1e88e5", "Analytics-Only": "#43a047"},
        title="Dropout Prediction: Integrated vs Analytics-Only",
        text="Accuracy"  # Display values on bars
    )
    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>"
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        showlegend=False,
        title_x=0.5,
        margin=dict(t=50, b=50),
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#e0e0e0"),
        xaxis=dict(tickfont=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

predictions_page()
