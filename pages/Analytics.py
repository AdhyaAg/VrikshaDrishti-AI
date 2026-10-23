import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from PIL import Image
from utils.project_data import load_project_statistics, load_model_summary, load_model_comparison, display_class_name, percentage

st.set_page_config(
    page_title="Impact & Admin Dashboard | Vasundhara Abhiyan",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: 
        linear-gradient(rgba(244, 249, 244, 0.78), rgba(232, 245, 233, 0.82)),
        url("https://vasundharaabhiyan.org/wp-content/uploads/2021/06/Optimized-amla-tree-scaled-2.jpg") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

section[data-testid="stSidebar"] {
    background: #EAF3EC !important;
    border-right: 2px solid #2E7D32 !important;
}

section[data-testid="stSidebar"] * {
    color: #0D3B16 !important;
}

[data-testid="stSidebarNav"] a {
    color: #0D3B16 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarNav"] a:hover, 
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #D8E9A8 !important;
    color: #0D3B16 !important;
}

body, p, span, label, div, h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #0D3B16 !important;
}

.main-title {
    font-size: 2.2rem;
    color: #0D3B16 !important;
    font-weight: 800;
    margin-bottom: 0px;
    border-left: 5px solid #2E7D32;
    padding-left: 12px;
}
.sub-title {
    font-size: 1.1rem;
    color: #2E7D32 !important;
    font-weight: 700;
    margin-bottom: 25px;
}

.stat-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border-left: 5px solid #2E7D32;
    border-top: 2px solid #FBC02D;
    text-align: center;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0D3B16 !important;
}
.stat-label {
    font-size: 0.95rem;
    color: #2E7D32 !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Real Dataset & Impact Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Vasundhara Abhiyan NGO | <a href="https://vasundharaabhiyan.org/" target="_blank">vasundharaabhiyan.org</a></div>', unsafe_allow_html=True)

stats = load_project_statistics()
model_summary = load_model_summary()
model_comparison = load_model_comparison()
category_counts = stats.get("class_counts", {})
total_unique = int(stats.get("source_unique_images", 0))
validated_total = int(stats.get("validated_images", 0))
duplicate_count = int(stats.get("source_valid_images", 0)) - total_unique
st.caption(f"Primary Real Dataset: {stats.get('source_original_images', 0)} readable files, {total_unique} unique photos ({duplicate_count} exact duplicates removed); {validated_total} reviewed labels accepted for training, {stats.get('pending_review', 0)} pending review, {stats.get('excluded', 0)} exclusion decisions including duplicate copies. Species/health labels verified: {stats.get('species_verified_count', 0)}.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total_unique}</div>
        <div class="stat-label">📷 Unique Primary Photos</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card" style="border-left-color: #2E7D32;">
        <div class="stat-number" style="color: #2E7D32 !important;">{validated_total}</div>
        <div class="stat-label"> Training-ready labels ({percentage(validated_total, total_unique)})</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card" style="border-left-color: #1B5E20;">
        <div class="stat-number" style="color: #1B5E20 !important;">{category_counts.get("trees", 0)} Trees</div>
        <div class="stat-label"> Tree-focused images ({percentage(category_counts.get("trees", 0), validated_total)})</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card" style="border-left-color: #FBC02D;">
        <div class="stat-number" style="color: #FBC02D !important;">{category_counts.get("saplings", 0)} Saplings</div>
        <div class="stat-label"> Sapling-focused images ({percentage(category_counts.get("saplings", 0), validated_total)})</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("###  Real Dataset Class Distribution")
    labels = [display_class_name(name) for name in ["trees", "leafs", "saplings"]]
    values = [category_counts.get(name, 0) for name in ["trees", "leafs", "saplings"]]
    colors = ["#1B5E20", "#2E7D32", "#FBC02D"]

    fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.45, marker_colors=colors)])
    fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_chart2:
    st.markdown("### 📈 Real Model Evaluation Confusion Matrix")
    cm_path = os.path.join("reports", "evaluation_v2", "confusion_matrix.png")
    if os.path.exists(cm_path):
        test = model_summary.get("test", {}) if model_summary else {}
        caption = (f"Grouped held-out test: accuracy {test.get('accuracy', 0):.1%}, macro F1 {test.get('macro_f1', 0):.3f}, n={sum(test.get('support', {}).values())}"
                   if test else "Grouped held-out test confusion matrix")
        st.image(cm_path, caption=caption, use_container_width=True)
    else:
        st.info("Confusion matrix plot will be displayed here once training evaluation completes.")

st.divider()

st.markdown("###  Audited Dataset Split (filename variants grouped together)")
split_counts = stats.get("split_class_counts", {})
count_keys = {"train": "train_count", "validation": "validation_count", "test": "test_count"}
split_rows = [
    {"Split": split_name.capitalize(), "Images": stats.get(count_keys[split_name], 0),
     **{display_class_name(category): split_counts.get(split_name, {}).get(category, 0)
        for category in ["leafs", "saplings", "trees"]}}
    for split_name in ["train", "validation", "test"]
]
st.dataframe(pd.DataFrame(split_rows), use_container_width=True)

if model_comparison:
    st.markdown("####  Baseline vs. Training-only Augmentation")
    comparison_rows = []
    for candidate, splits in model_comparison["validation_and_test"].items():
        validation = splits.get("validation", {})
        test = splits.get("test", {})
        comparison_rows.append({
            "Candidate": candidate.capitalize(),
            "Validation accuracy": f"{validation.get('accuracy', 0):.1%}",
            "Validation macro F1": f"{validation.get('macro_f1', 0):.3f}",
            "Grouped test accuracy": f"{test.get('accuracy', 0):.1%}",
            "Grouped test macro F1": f"{test.get('macro_f1', 0):.3f}",
            "Test images": sum(test.get("support", {}).values()),
        })
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)
