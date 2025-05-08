import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_data
from segmentation.clustering import perform_kmeans
from segmentation.distance_metrics import compute_cosine_similarity

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧩 Customer Segmentation for E-commerce")

uploaded_file = st.file_uploader("📁 Upload your customer data (CSV)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check if file is empty
        if df.empty:
            st.warning("⚠️ Uploaded file is empty.")
            st.stop()

        # Check required columns
        required_columns = ["Age", "Gender", "Annual Income (k$)", "Spending Score", "Pages Visited", "Time on Site"]
        if not all(col in df.columns for col in required_columns):
            missing_cols = set(required_columns) - set(df.columns)
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Show raw data
        st.subheader("📊 Raw Data")
        st.dataframe(df.head(10))

        # Select number of clusters
        n_clusters = st.slider("🎯 Select Number of Clusters", min_value=2, max_value=5, value=3)

        # Preprocess data
        st.write("🔄 Preprocessing data...")
        processed_df = preprocess_data(df)

        # Perform clustering
        clustered_data, model = perform_kmeans(processed_df, n_clusters)
        df["Cluster"] = clustered_data["Cluster"]

        # Show clustered data
        st.subheader("🧠 Clustered Data")
        st.dataframe(df)

        # Cluster visualization
        st.subheader("📈 Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score", hue="Cluster", palette="viridis", ax=ax)
        st.pyplot(fig)

        # Cosine similarity
        st.subheader("📐 Cosine Similarity Matrix")
        similarity = compute_cosine_similarity(processed_df)
        st.dataframe(similarity)

        # Recommendations
        st.subheader("🎁 Personalized Recommendations by Cluster")
        recommendations = {
            0: ["Budget Products", "Discounted Essentials", "Affordable Electronics"],
            1: ["Premium Fashion", "Luxury Gadgets", "Smart Devices"],
            2: ["Casual Wear", "Daily Use Products", "Eco-Friendly Items"],
            3: ["Office Supplies", "Work From Home Kits", "Productivity Tools"],
            4: ["Fitness Equipment", "Health Supplements", "Yoga Mats"]
        }

        for cluster, items in recommendations.items():
            st.markdown(f"**Cluster {cluster}:** " + ", ".join(items))

        # Download button
        st.download_button("⬇️ Download Clustered Data", df.to_csv(index=False), file_name="clustered_customers.csv")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()
else:
    st.info("📤 Please upload a CSV file to get started.")
