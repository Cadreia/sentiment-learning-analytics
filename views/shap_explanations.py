import streamlit as st
import shap
import streamlit.components.v1 as components
from scripts.shap import get_shap_results
from scripts.sentiment_analysis import load_sentiment_pipeline
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Custom CSS for professional styling
st.markdown("""
    <style>
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
    .plot-selector {
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


def plot_to_base64():
    """Convert current Matplotlib figure to base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str


def display_text_shap(shap_values, texts_subset, title):
    """Display SHAP text plots for individual texts."""
    st.markdown(f"#### {title} - Text Plots", unsafe_allow_html=True)
    # for i, text in enumerate(texts_subset):
    #     st.markdown(f"**Text {i + 1}:** {text[:200]}{'...' if len(text) > 200 else ''}")
    try:
        # Generate the HTML for the entire SHAP Explanation object (all students)
        shap_html = shap.plots.text(shap_values, display=False)  # display=False returns HTML
        print(f"SHAP HTML Output for {title}:", shap_html)
        with open(f"{title.lower().replace(' ', '_')}_shap_output.html", "w") as f:
            f.write(shap_html if shap_html else "<p>No HTML output</p>")
        if shap_html:
            components.html(shap_html, height=600, scrolling=True)  # Increased height for better visibility
        else:
            st.write(f"No valid HTML output from shap.plots.text for {title}. Falling back to raw SHAP values.")
            raise ValueError("No HTML output")
    except Exception as e:
        st.warning(f"Could not render text plot for text {i + 1}: {str(e)}")
    st.markdown("---")


def display_bar_shap(shap_values, title, tokens=None):

    # st.write(f"Type of shap_values: {type(shap_values)}")
    # st.text(shap_values[:2])
    # st.text(shap_values.data[:2])
    # st.text(shap_values.feature_names[:2])
    """Display SHAP summary plot for feature importance."""
    st.markdown(f"#### {title} - Summary Plot", unsafe_allow_html=True)
    st.markdown("This plot shows the most impactful features (e.g., words or tokens) across all samples.")

    try:
        # st.text(shap_values.shape)
        # st.text(shap_values.data)
        # st.text(shap_values.feature_names)

        # Generate SHAP bar plot using absolute sum across samples
        shap.plots.bar(shap_values.abs.sum(0))
        img_str = plot_to_base64()
        st.image(f"data:image/png;base64,{img_str}", caption=f"{title} - Feature Importance")

    except Exception as e:
        st.warning(f"Could not render summary plot: {str(e)}")


def shap_explanations_page():
    # Check for required data
    if "merged_df" not in st.session_state:
        st.error("Please run the analysis on the Overview page first to generate the clustering results.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    else:
        st.header("Sentiment Polarity Preview")
        data = st.session_state["merged_df"].copy()

    try:
        coursecontent_shap_values, labwork_shap_values = get_shap_results(load_sentiment_pipeline())
    except Exception as e:
        st.error(f"Error loading SHAP results: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Course Content SHAP Explanations
    st.markdown('<h2 class="section-header">Course Content Sentiment</h2>', unsafe_allow_html=True)
    columns_to_display = ["Student_ID", "coursecontent_text", "coursecontent_sentiment_score"]
    st.dataframe(data[columns_to_display][:5], use_container_width=True)

    st.markdown("Explore how individual words and overall features contribute to sentiment predictions.",
                unsafe_allow_html=True)
    subset_data = data[["coursecontent_text_cleaned"]].head(5)
    coursecontent_texts_subset = subset_data["coursecontent_text_cleaned"].fillna("").tolist()

    if coursecontent_texts_subset and coursecontent_shap_values is not None:
        # Plot type selector
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Text Plots", "Summary Plot"],
            key="coursecontent_plot_type",
            help="Choose between detailed text explanations or a summary of feature importance.",
            label_visibility="collapsed"
        )

        if plot_type == "Text Plots":
            display_text_shap(coursecontent_shap_values[:5], coursecontent_texts_subset, "Course Content Sentiment")
        else:
            feature_names = getattr(coursecontent_shap_values, "feature_names", None)
            display_bar_shap(coursecontent_shap_values, "Course Content Sentiment", feature_names)
    else:
        st.warning("No SHAP values available for Course Content.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Lab Work SHAP Explanations
    st.markdown('<div class="sub-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Lab Work Sentiment</h2>', unsafe_allow_html=True)
    columns_to_display = ["Student_ID", "labwork_text", "labwork_sentiment_score"]
    st.dataframe(data[columns_to_display][:5], use_container_width=True)

    subset_data = data[["labwork_text"]].head(5)
    labwork_texts_subset = subset_data["labwork_text"].fillna("").tolist()

    if labwork_texts_subset and labwork_shap_values is not None:
        # Plot type selector
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Text Plots", "Summary Plot"],
            key="labwork_plot_type",
            help="Choose between detailed text explanations or a summary of feature importance.",
            label_visibility="collapsed"
        )

        if plot_type == "Text Plots":
            display_text_shap(labwork_shap_values[:5], labwork_texts_subset, "Lab Work Sentiment")
        else:
            feature_names = getattr(labwork_shap_values, "feature_names", None)
            display_bar_shap(labwork_shap_values, "Labwork Sentiment", feature_names)

    else:
        st.warning("No SHAP values available for Lab Work.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    shap_explanations_page()