import streamlit as st
import plotly.express as px
from PIL import Image
from utils.predictor import predict_tree_health
from utils.report_generator import generate_pdf_report

st.set_page_config(
    page_title="AI Image Category Predictor | Vasundhara Abhiyan",
    page_icon="",
    layout="wide"
)

# Simple & Clean CSS
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

.main-header {
    font-size: 2.2rem;
    color: #0D3B16 !important;
    font-weight: 800;
    margin-bottom: 0px;
    border-left: 5px solid #2E7D32;
    padding-left: 12px;
}
.sub-header {
    font-size: 1.1rem;
    color: #2E7D32 !important;
    font-weight: 700;
    margin-bottom: 25px;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #4E9F3D !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #0D3B16 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div, input, textarea {
    background-color: #FFFFFF !important;
    color: #0D3B16 !important;
    border: 1.5px solid #4E9F3D !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stExpander, details, summary {
    background: #FFFFFF !important;
    border: 1px solid #A5D6A7 !important;
    border-radius: 12px !important;
    color: #0D3B16 !important;
    margin-bottom: 10px !important;
}

summary * {
    color: #0D3B16 !important;
    font-weight: 700 !important;
}

.stButton>button, .stDownloadButton>button {
    background: #1E5128 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    height: 48px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(30, 81, 40, 0.2) !important;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background: #4E9F3D !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"> AI Image Category Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Vasundhara Abhiyan | <a href="https://vasundharaabhiyan.org/" target="_blank">vasundharaabhiyan.org</a></div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📷 Select Image Input Method")
    input_mode = st.radio(
        "Choose photo source:",
        ["Option A: File Upload (Drag & Drop / Picker)", "Option B: Live Web Camera Capture"],
        horizontal=True
    )

    image_file = None
    if "Option A" in input_mode:
        uploaded_file = st.file_uploader(
            "Upload a tree or leaf photo:",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_file = uploaded_file
    else:
        camera_file = st.camera_input("Capture a tree or leaf photo:")
        if camera_file is not None:
            image_file = camera_file

    location_tag = st.selectbox(
        " Select Baner Plantation Zone:",
        ["Baner Hill Biodiversity Park", "Baner Hill Trail - Phase 1", "Baner Hill Trail - Phase 2", "Pashan-Baner Link Road Greenbelt", "Mula-Mutha Riverbank Baner Zone"]
    )

    if image_file is not None:
        st.image(image_file, caption="Input Photo", use_container_width=True)

with col_output:
    if image_file is not None:
        with st.spinner(" Running EfficientNetB0 inference..."):
            try:
                results = predict_tree_health(image_file)

                if "scan_history" not in st.session_state:
                    st.session_state["scan_history"] = []
                
                if not any(item.get("top_label") == results["top_label"] and item.get("confidence") == results["confidence"] for item in st.session_state["scan_history"]):
                    scan_entry = results.copy()
                    scan_entry["location"] = location_tag
                    st.session_state["scan_history"].insert(0, scan_entry)

                st.markdown("### 📊 Image Category Prediction")
                
                badge_bg = results["badge_color"]
                st.markdown(
                    f'<div style="background-color:{badge_bg}; color:white !important; padding:10px 20px; border-radius:20px; font-weight:800; font-size:17px; display:inline-block; margin-bottom:14px; border:1.5px solid #FBC02D;">'
                    f'{results["status"].upper()} ({results["confidence"]}% Top-class Score)</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"**Image Category:** <span style='font-size:18px; font-weight:800; color:#0D3B16;'>{results['species']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Result:** <span style='font-size:17px; font-weight:800; color:#2E7D32;'>{results['diagnosis']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Health scope:** {results['symptoms']}")

                st.markdown("#### 📈 Prediction Confidence Breakdown")
                top5_labels = [item[0].capitalize() for item in results["top5"]]
                top5_scores = [item[1] for item in results["top5"]]

                fig = px.bar(
                    x=top5_scores,
                    y=top5_labels,
                    orientation='h',
                    color=top5_scores,
                    color_continuous_scale="Greens"
                )
                fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

                with st.expander(" About this result", expanded=True):
                    st.markdown(f"**Model scope:** {results['organic_remedy']}")
                    st.markdown(f"**Next step:** {results['pune_eco_tip']}")

                pdf_bytes = generate_pdf_report(results, location_tag=location_tag)
                st.download_button(
                    label="📄 Download Image Category Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"VasundharaAbhiyan_Category_{results['species'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.info(" Upload a photo or capture one to view its image category. This model does not diagnose plant health.")
