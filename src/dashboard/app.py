import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Brand Sentiment Analyzer", layout="wide")

def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def analyze_single(text, platform=None, brand=None):
    payload = {"text": text}
    if platform: payload["platform"] = platform
    if brand: payload["brand"] = brand
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

def analyze_batch(df):
    comments = []
    for _, row in df.iterrows():
        item = {"text": str(row.get("text", row.get("comment", "")))}
        if "platform" in row:
            item["platform"] = str(row["platform"])
        if "brand" in row:
            item["brand"] = str(row["brand"])
        comments.append(item)
    
    try:
        response = requests.post(f"{API_URL}/predict/batch", json={"comments": comments})
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return []

def fetch_clusters():
    try:
        response = requests.get(f"{API_URL}/clusters")
        response.raise_for_status()
        return response.json().get("clusters", [])
    except:
        return []

def fetch_model_info():
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()

with st.sidebar:
    st.title("Brand Sentiment Analyzer")
    api_online = check_api_status()
    if api_online:
        st.success("API: Connected")
        model_info = fetch_model_info()
        if model_info:
            model_name = model_info.get("sentiment_model_type", "Active")
            st.caption(f"Active Model: **{model_name}**")
    else:
        st.error("API: Disconnected")
    
    st.markdown("### Data Loading")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        if st.button("Process Uploaded CSV"):
            df = pd.read_csv(uploaded_file)
            with st.spinner("Analyzing comments..."):
                results = analyze_batch(df)
                if results:
                    st.session_state.batch_results = pd.DataFrame(results)
                    st.success(f"Processed {len(results)} comments!")
    
    if st.button("Load Sample Data"):
        sample_path = Path("data/synthetic/comments.csv")
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            with st.spinner("Analyzing sample data..."):
                results = analyze_batch(df)
                if results:
                    st.session_state.batch_results = pd.DataFrame(results)
                    st.success(f"Processed {len(results)} comments!")
        else:
            st.error("Sample data not found.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Complaint Analysis", 
    "Platform & Brand", 
    "Single Analysis", 
    "Batch Results"
])

results_df = st.session_state.batch_results

with tab1:
    st.header("Overview")
    if not results_df.empty and "sentiment" in results_df.columns:
        total = len(results_df)
        pos = (results_df["sentiment"] == "positive").sum()
        neg = (results_df["sentiment"] == "negative").sum()
        neu = (results_df["sentiment"] == "neutral").sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Comments", total)
        col2.metric("Positive %", f"{(pos/total*100):.1f}%")
        col3.metric("Negative %", f"{(neg/total*100):.1f}%")
        col4.metric("Neutral %", f"{(neu/total*100):.1f}%")
        
        fig = px.pie(
            values=[pos, neg, neu], 
            names=["Positive", "Negative", "Neutral"],
            title="Sentiment Distribution",
            color_discrete_sequence=["#2ca02c", "#d62728", "#1f77b4"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (neg/total)*100,
            title = {'text': "Negative Feedback %"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "red"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("No data available. Please load data from the sidebar.")

with tab2:
    st.header("Complaint Analysis")
    if not results_df.empty and "complaint_category" in results_df.columns:
        complaints = results_df[results_df["complaint_category"].notna() & (results_df["complaint_category"] != "None")]
        if not complaints.empty:
            cat_counts = complaints["complaint_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_bar = px.bar(cat_counts, x="Category", y="Count", title="Complaint Categories")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("Cluster Details")
            clusters = fetch_clusters()
            if clusters:
                for c in clusters:
                    cid = c.get("cluster_id")
                    c_name = c.get("name", f"Cluster {cid}")
                    keywords = c.get("keywords", "")
                    c_complaints = complaints[complaints["complaint_cluster"] == cid]
                    c_size = len(c_complaints)
                    st.markdown(f"##### Cluster {cid}: {c_name} ({c_size} complaints)")
                    if keywords:
                        st.caption(f"Top terms: {keywords}")
                    if not c_complaints.empty:
                        for ex in c_complaints["original_text"].head(3):
                            st.markdown(f"- {ex}")
            else:
                for cat in cat_counts["Category"]:
                    st.markdown(f"##### {cat}")
                    examples = complaints[complaints["complaint_category"] == cat]["original_text"].head(3)
                    for ex in examples:
                        st.markdown(f"- {ex}")
        else:
            st.info("No complaints found in the dataset.")
    else:
        st.info("No data available.")

with tab3:
    st.header("Platform & Brand")
    if not results_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            if "platform" in results_df.columns:
                plat_counts = results_df["platform"].value_counts().reset_index()
                plat_counts.columns = ["Platform", "Count"]
                fig_plat = px.pie(plat_counts, values="Count", names="Platform", title="Platform Distribution")
                st.plotly_chart(fig_plat, use_container_width=True)
        with col2:
            if "brand" in results_df.columns and "sentiment" in results_df.columns:
                brand_sent = results_df.groupby(["brand", "sentiment"]).size().reset_index(name="Count")
                fig_brand = px.bar(brand_sent, x="brand", y="Count", color="sentiment", title="Brand-wise Sentiment", barmode="group")
                st.plotly_chart(fig_brand, use_container_width=True)
    else:
        st.info("No data available.")

with tab4:
    st.header("Single Analysis")
    with st.form("single_analysis"):
        text_input = st.text_area("Enter comment:")
        platform_input = st.text_input("Platform (optional):")
        brand_input = st.text_input("Brand (optional):")
        submitted = st.form_submit_button("Analyze")
        
        if submitted and text_input:
            with st.spinner("Analyzing..."):
                res = analyze_single(text_input, platform_input, brand_input)
                if res:
                    st.success("Analysis Complete")
                    st.json(res)

with tab5:
    st.header("Batch Results")
    if not results_df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        filtered_df = results_df.copy()
        with col_f1:
            if "sentiment" in results_df.columns:
                sentiments = ["All"] + sorted([str(s) for s in results_df["sentiment"].dropna().unique()])
                selected_sentiment = st.selectbox("Filter by Sentiment", sentiments)
                if selected_sentiment != "All":
                    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
        with col_f2:
            if "platform" in results_df.columns:
                platforms = ["All"] + sorted([str(p) for p in results_df["platform"].dropna().unique()])
                selected_platform = st.selectbox("Filter by Platform", platforms)
                if selected_platform != "All":
                    filtered_df = filtered_df[filtered_df["platform"] == selected_platform]
        with col_f3:
            if "brand" in results_df.columns:
                brands = ["All"] + sorted([str(b) for b in results_df["brand"].dropna().unique()])
                selected_brand = st.selectbox("Filter by Brand", brands)
                if selected_brand != "All":
                    filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

        st.dataframe(filtered_df, use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv',
        )
    else:
        st.info("No data available.")
