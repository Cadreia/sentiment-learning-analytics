# pages/cluster_visualization.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def page():
    st.title("Cluster Visualization")

    # Check if clusters and data are available
    if 'clusters' not in st.session_state or 'reduced_features' not in st.session_state:
        st.warning("No clustering data available. Please run the analysis on the Overview page first.")
        return

    # Retrieve clusters and data
    clusters = st.session_state['clusters']
    X = st.session_state['reduced_features']

    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

page()