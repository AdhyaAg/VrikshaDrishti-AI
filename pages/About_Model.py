import streamlit as st
from utils.project_data import load_active_model_metadata, display_class_name

st.set_page_config(
    page_title="About AI Model & NGO | Vasundhara Abhiyan",
    page_icon="ℹ",
    layout="wide"
)

# Simple & Clean CSS System
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

.title-style {
    font-size: 2.2rem;
    color: #0D3B16 !important;
    font-weight: 800;
    border-left: 5px solid #2E7D32;
    padding-left: 12px;
}
.sub-style {
    font-size: 1.1rem;
    color: #2E7D32 !important;
    font-weight: 700;
    margin-bottom: 25px;
}
.card-box {
    background: #FFFFFF;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #C8E6C9;
    border-top: 4px solid #2E7D32;
    margin-bottom: 20px;
}
.card-box * {
    color: #0D3B16 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-style">ℹ️ Vasundhara Abhiyan Mission & AI Technology</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-style">Preserving Pune\'s Biodiversity | Official Website: <a href="https://vasundharaabhiyan.org/" target="_blank">vasundharaabhiyan.org</a></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card-box">
        <h3> About Vasundhara Abhiyan NGO</h3>
        <p><b>Vasundhara Abhiyan</b> is a registered environmental NGO committed to conserving and enriching Pune’s natural ecosystems.</p>
        <p>Our core focus is the ecological restoration of <b>Baner Hill</b> through native tree plantation, soil conservation, continuous watering drives, and environmental education.</p>
        <h4 style="color:#2E7D32; margin-top:15px;">Key NGO Activities:</h4>
        <ul>
            <li><b>Native Tree Afforestation:</b> Planting indigenous saplings like Saptaparni, Amla, Karanja, Arjun, Mango, Jamun, and Bael.</li>
            <li><b>Soil & Water Conservation:</b> Creating continuous contour trenches (CCT) to recharge Baner’s groundwater table.</li>
            <li><b>Seed Ball Initiatives:</b> Organizing seed-ball work to support vegetation recovery on degraded slopes.</li>
            <li><b>Community Engagement:</b> Weekly volunteer plantation drives and biodiversity monitoring workshops.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card-box">
        <h3> Audited Image Category Model</h3>
        <p><b>VrikshaDrishti AI</b> uses an <b>EfficientNetB0 transfer-learning</b> classifier trained on visually reviewed primary Real Dataset photos. It classifies image framing/content; it does not identify tree species or diagnose plant health.</p>
        <h4 style="color:#2E7D32; margin-top:15px;">Model Technical Specifications:</h4>
        <ul>
            <li><b>Base Network:</b> EfficientNetB0 (ImageNet Pre-trained)</li>
            <li><b>Input Size:</b> 224 × 224 pixels (RGB)</li>
            <li><b>Image Categories:</b> Leaf-focused, Sapling-focused, Tree-focused; low-confidence results are returned as uncertain.</li>
            <li><b>Training:</b> Frozen ImageNet feature extractor, class-balanced loss weighting, validation early stopping, and a mild augmentation candidate applied to training images only.</li>
            <li><b>Evaluation:</b> Filename-variant groups are held together across train, validation, and test splits.</li>
            <li><b>Limits:</b> The source set is small and category framing is subjective; species, health, and disease labels are unverified.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("###  Supported Image Categories")
metadata = load_active_model_metadata()
class_list = metadata.get("class_names", [])
col_a, col_b = st.columns(2)
for idx, class_name in enumerate(class_list):
    target = col_a if idx % 2 == 0 else col_b
    target.markdown(f"- **{display_class_name(class_name)}**")
st.caption(f"Uncertainty threshold: {metadata.get('uncertainty_threshold', 0.45):.0%} maximum class probability. This is a fixed abstention rule; model confidence is not calibrated and is not a health assessment.")
