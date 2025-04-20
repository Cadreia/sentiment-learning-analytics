# pages/cluster_visualization.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

# Custom CSS for professional styling (consistent with view_all_predictions.py)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
        text-align: center;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
    }
    .sub-header {
        font-size: 22px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-value {
        font-size: 18px;
        color: #1f77b4;
    }
    .stDataFrame, .stTable {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def cluster_visualization_page():
    st.markdown('<div class="main-header">Interactive 3D Cluster Visualization</div>', unsafe_allow_html=True)

    # Check if necessary data is available in session state
    if "reduced_integrated" not in st.session_state or "groups" not in st.session_state or "anomalies" not in st.session_state:
        st.error("Please run the analysis on the Overview page first to generate the clustering results.")
        return

    # Retrieve data from session state
    X_integrated = st.session_state["X_integrated"]
    clusters = st.session_state["groups"]  # Use groups instead of clusters for full dataset
    anomalies = st.session_state["anomalies"]

    # For hover data and cluster summary, we still need merged_df
    if "merged_df" not in st.session_state:
        st.warning("Merged DataFrame not found in session state. Hover data and detailed summary may be limited.")
        hover_data_df = None
    else:
        hover_data_df = st.session_state["merged_df"].copy()

    # Prepare data for visualization
    # Reduce to 3 dimensions for 3D plotting if necessary
    if X_integrated.shape[1] > 3:
        pca_3d = PCA(n_components=3, random_state=42)
        X_3d = pca_3d.fit_transform(X_integrated)
    else:
        X_3d = X_integrated[:, :3]  # Use the first 3 components if already reduced

    # Create a DataFrame for Plotly
    plot_data = pd.DataFrame(X_3d, columns=["PC1", "PC2", "PC3"])
    plot_data["Cluster"] = clusters
    plot_data["Cluster"] = plot_data["Cluster"].astype('category')

    # Add hover data from merged_df if available
    if hover_data_df is not None:
        hover_columns = ["Student_ID", "Total_Score", "Attendance"]
        available_hover_data = [col for col in hover_columns if col in hover_data_df.columns]
        if available_hover_data:
            for col in available_hover_data:
                plot_data[col] = hover_data_df[col].values
        else:
            st.warning(
                "Hover data columns (Student_ID, Total_Score, Attendance) not found in merged_df. Hover information "
                "will be limited.")
            available_hover_data = None
    else:
        available_hover_data = None

    # Section 1: 3D Cluster Visualization
    st.markdown('<div class="section-header">3D Cluster Plot</div>', unsafe_allow_html=True)

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        plot_data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        hover_data=available_hover_data,
        title="Student Clusters in 3D",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
        color_discrete_sequence=px.colors.qualitative.Set1  # Use a distinct color sequence
    )

    # Customize the layout
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
            aspectmode="cube",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3",
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Update traces to increase marker size for better visibility
    fig.update_traces(marker=dict(size=5))

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Section 2: Cluster Summary
    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)

    # Compute summary metrics
    num_clusters = len(np.unique(clusters))
    num_anomalies = sum(anomalies)

    # Display summary metrics in styled boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Status:</div>
                <div class="metric-value">Unsupervised Learning Path (K-Means)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Number of Clusters Found:</div>
                <div class="metric-value">{num_clusters}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Number of Anomalies:</div>
                <div class="metric-value">{num_anomalies}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Display cluster distribution
    st.markdown('<div class="sub-header">Cluster Distribution</div>', unsafe_allow_html=True)
    distribution = pd.Series(clusters).value_counts().sort_index().reset_index()
    distribution.columns = ['Cluster', 'Number of Students']
    # Style the distribution table
    styled_distribution = distribution.style.set_properties(**{
        'background-color': '#f8f9fa',
        'border': '1px solid #ddd',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1f77b4'), ('color', 'white'), ('font-weight', 'bold')]}
    ])
    st.table(styled_distribution)


cluster_visualization_page()