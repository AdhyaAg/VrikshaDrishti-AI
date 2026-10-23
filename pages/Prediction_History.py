import streamlit as st
from utils.report_generator import generate_pdf_report

st.set_page_config(
    page_title="Prediction History | Vasundhara Abhiyan",
    page_icon="",
    layout="wide"
)

# Unified High-Contrast & Simple Native Sidebar CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* 1. Brightened Background Image */
.stApp {
    background: 
        linear-gradient(rgba(244, 249, 244, 0.78), rgba(232, 245, 233, 0.82)),
        url("https://vasundharaabhiyan.org/wp-content/uploads/2021/06/Optimized-amla-tree-scaled-2.jpg") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* 2. SIMPLE NATIVE SIDEBAR (Clean, No Nested Borders) */
section[data-testid="stSidebar"] {
    background: #EAF3EC !important;
    border-right: 2px solid #2E7D32 !important;
}

section[data-testid="stSidebar"] * {
    color: #0D3B16 !important;
}

/* Streamlit native sidebar nav: simple clean active pill */
[data-testid="stSidebarNav"] a {
    color: #0D3B16 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    box-shadow: none !important;
    background-color: transparent !important;
}

[data-testid="stSidebarNav"] a:hover, 
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #D8E9A8 !important;
    color: #0D3B16 !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] * {
    color: #0D3B16 !important;
}

/* Main Content Text Contrast */
body, p, span, label, div, h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #0D3B16 !important;
}

/* Expander Header Fix */
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
    font-size: 16px !important;
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

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: #1E5128 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    height: 48px !important;
    font-size: 15px !important;
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

st.markdown('<div class="title-style"> Scan History & PDF Reports</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-style">Vasundhara Abhiyan Baner | <a href="https://vasundharaabhiyan.org/" target="_blank">vasundharaabhiyan.org</a></div>', unsafe_allow_html=True)

if "scan_history" not in st.session_state or len(st.session_state["scan_history"]) == 0:
    st.info("ℹ No photo predictions recorded in this session yet. Use the image-category predictor to classify a photo.")
else:
    st.markdown(f"**Total Scans Recorded:** `{len(st.session_state['scan_history'])}`")
    
    for i, item in enumerate(st.session_state["scan_history"]):
        with st.expander(f"Scan #{len(st.session_state['scan_history']) - i}: {item['species']} ({item['status']}) - score {item['confidence']}%", expanded=(i==0)):
            c_img, c_info = st.columns([1, 2])
            with c_img:
                if "pil_image" in item and item["pil_image"] is not None:
                    st.image(item["pil_image"], caption="Input Photo", width=200)
            with c_info:
                st.markdown(f"**Location:** `{item.get('location', 'Baner Biodiversity Park')}`")
                st.markdown(f"**Image-category result:** <span style='font-weight:800; color:#2E7D32;'>{item['diagnosis']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Model scope:** {item['symptoms']}")
                st.markdown(f"**Treatment scope:** {item['organic_remedy']}")
                
                pdf_bytes = generate_pdf_report(item, location_tag=item.get('location', 'Baner Biodiversity Park'))
                st.download_button(
                    label=f"📄 Download PDF Report for Scan #{len(st.session_state['scan_history']) - i}",
                    data=pdf_bytes,
                    file_name=f"Report_{item['species'].replace(' ', '_')}_{i}.pdf",
                    mime="application/pdf",
                    key=f"hist_pdf_{i}"
                )
