import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from utils.predictor import predict_tree_health
from utils.report_generator import generate_pdf_report
from utils.project_data import load_project_statistics, load_model_summary, load_model_comparison, display_class_name, percentage

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="VrikshaDrishti AI | Vasundhara Abhiyan",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom CSS: Rich Glassmorphic Main App UI + Simple Native Sidebar
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* 1. Vibrant Brightened Background Image */
.stApp {
    background: 
        linear-gradient(rgba(244, 249, 244, 0.76), rgba(232, 245, 233, 0.80)),
        url("https://vasundharaabhiyan.org/wp-content/uploads/2021/06/Optimized-amla-tree-scaled-2.jpg") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* Hide default headers */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

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
}

[data-testid="stSidebarNav"] a:hover, 
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #D8E9A8 !important;
    color: #0D3B16 !important;
}

/* 3. RICH GLASSMORPHIC & PREMIUM MAIN BODY STYLING */

/* High Contrast Main Body Typography */
p, span, label, div, h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #0D3B16 !important;
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 800 !important;
}

/* Top Navbar */
.navbar-container {
    background: linear-gradient(135deg, #0D3B16 0%, #1E5128 60%, #2E7D32 100%);
    padding: 18px 32px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 30px rgba(13, 59, 22, 0.3);
    margin-bottom: 25px;
    border-bottom: 4px solid #FBC02D;
}

.brand-title {
    font-size: 28px;
    font-weight: 800;
    color: #FFF176 !important;
}
.brand-sub {
    font-size: 14px;
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Hero Box with Gradient Overlay & Shadow */
.hero-box {
    background: 
        linear-gradient(rgba(9, 40, 14, 0.82), rgba(9, 40, 14, 0.86)),
        url("https://vasundharaabhiyan.org/wp-content/uploads/2021/06/Optimized-amla-tree-scaled-2.jpg");
    background-size: cover;
    background-position: center;
    padding: 55px 45px;
    border-radius: 26px;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    margin-bottom: 35px;
    border-left: 8px solid #FBC02D;
}

.hero-box * {
    color: white !important;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.2;
    color: #FFFFFF !important;
    margin-bottom: 12px;
}

.hero-tagline {
    font-size: 22px;
    color: #FFF176 !important;
    font-weight: 700;
    margin-bottom: 18px;
}

.hero-body {
    font-size: 16px;
    color: #F4F9F4 !important;
    max-width: 800px;
    line-height: 1.6;
    margin-bottom: 25px;
}

/* Glassmorphic Feature Cards with Dynamic Hover Animations */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(12px);
    padding: 26px;
    border-radius: 22px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
    border: 2px solid #C8E6C9;
    border-top: 5px solid #2E7D32;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}

.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 40px rgba(46, 125, 50, 0.22);
    border-top-color: #FBC02D;
}

.card-title {
    font-size: 20px;
    font-weight: 800;
    color: #0D3B16 !important;
    margin-bottom: 8px;
}

.card-desc {
    font-size: 14px;
    color: #1B4D20 !important;
    line-height: 1.5;
    font-weight: 600;
}

/* Stat Badge Pill (Yellow & Green Plantation Theme) */
.stat-pill {
    background: linear-gradient(135deg, #1E5128 0%, #0D3B16 100%);
    color: #FFF176 !important;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: 800;
    font-size: 15px;
    display: inline-block;
    border: 2px solid #FBC02D;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* Expander & Accordion Styling */
.stExpander, details, summary {
    background: #FFFFFF !important;
    border: 1px solid #A5D6A7 !important;
    border-radius: 14px !important;
    color: #0D3B16 !important;
    margin-bottom: 12px !important;
}

summary * {
    color: #0D3B16 !important;
    font-weight: 800 !important;
    font-size: 16px !important;
}

/* File Uploader Container */
[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #4E9F3D !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #0D3B16 !important;
    font-weight: 700 !important;
}

/* Action & Download Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    height: 52px !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    border: 2px solid #FBC02D !important;
    box-shadow: 0 6px 20px rgba(27, 94, 32, 0.3) !important;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%) !important;
    color: #FFFDE7 !important;
    transform: translateY(-2px) !important;
}

.section-header {
    font-size: 32px;
    font-weight: 800;
    color: #0D3B16 !important;
    margin-top: 15px;
    margin-bottom: 20px;
    border-left: 6px solid #2E7D32;
    padding-left: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Navigation Bar
# ---------------------------------------------------
st.markdown("""
<div class="navbar-container">
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="https://vasundharaabhiyan.org/wp-content/uploads/2021/05/cropped-logo-vasundharaabhiyan.png" width="60" style="border-radius: 10px; background: white; padding: 4px;">
        <div>
            <div class="brand-title">🌳 VrikshaDrishti AI</div>
            <div class="brand-sub">Vasundhara Abhiyan | Official Website: <a href="https://vasundharaabhiyan.org/" target="_blank" style="color: #FFF176; text-decoration: underline;">vasundharaabhiyan.org</a></div>
        </div>
    </div>
    <div style="font-size: 15px; font-weight: 700; color: #FFFDE7; background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 12px; border: 1px solid #FBC02D;">
        📍 Baner Hill Biodiversity Park, Pune
    </div>
</div>
""", unsafe_allow_html=True)

# Simple Sidebar Branding
st.sidebar.image("https://vasundharaabhiyan.org/wp-content/uploads/2021/05/cropped-logo-vasundharaabhiyan.png", width=110)
st.sidebar.markdown("<h3 style='color:#0D3B16; margin-bottom:0; font-weight:800;'>Vasundhara Abhiyan</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#1B5E20; font-weight:700; font-size:13px; margin-bottom:10px;'>Registered NGO | Pune, MH</p>", unsafe_allow_html=True)
st.sidebar.markdown("[🌐 Visit Official Website](https://vasundharaabhiyan.org/)")

page = st.sidebar.radio(
    "Select Portal:",
    [
        "🏡 Home & NGO Mission",
        "🔬 Predict Image Category (AI Engine)",
        "📊 Admin & Impact Dashboard",
        "📞 Contact & NGO Enquiry"
    ]
)

# ---------------------------------------------------
# PAGE 1: HOME & NGO MISSION
# ---------------------------------------------------
if page == "🏡 Home & NGO Mission":
    home_stats = load_project_statistics()
    home_categories = home_stats.get("class_counts", {})
    st.markdown("""
    <div class="hero-box">
        <div class="hero-tagline">🌿 Preserving Pune's Biodiversity through Native Tree Plantation & AI.</div>
        <div class="hero-title">Vasundhara Abhiyan Baner - VrikshaDrishti AI</div>
        <div class="hero-body">
            Vasundhara Abhiyan is a registered NGO passionately dedicated to conserving biodiversity across Pune. 
            We work tirelessly to plant and nurture native tree saplings (Amla, Neem, Karanja, Saptaparni, Mango, Jamun, Bael), 
            restore soil & water tables on Baner Hill, and use an image classifier to sort photos into leaf-focused, sapling-focused, or tree-focused views.
        </div>
        <div style="margin-top: 15px;">
            <a href="https://vasundharaabhiyan.org/" target="_blank" style="background: #FBC02D; color: #0D3B16; padding: 12px 24px; border-radius: 12px; font-weight: 800; text-decoration: none; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                🌐 Explore vasundharaabhiyan.org ➔
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Project Dataset Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f'<div class="stat-pill">📷 {home_stats.get("source_unique_images", 0)} Unique Photos</div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-pill">✅ {home_stats.get("validated_images", 0)} Reviewed Category Labels</div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat-pill">🏷️ {len(home_categories)} Image Categories</div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="stat-pill">🔬 {home_stats.get("species_verified_count", 0)} Species / Health Labels Verified</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown('<div class="section-header">Our Environmental Initiatives</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 38px; margin-bottom: 10px;">🌱</div>
            <div class="card-title">Native Plantation</div>
            <div class="card-desc">Planting indigenous species like Saptaparni, Karanja, Amla, and Arjun across Baner Hill slopes.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 38px; margin-bottom: 10px;">🔬</div>
            <div class="card-title">AI Image Categories</div>
            <div class="card-desc">EfficientNetB0 sorts images into leaf-focused, sapling-focused, and tree-focused categories. It does not identify species or diagnose disease.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 38px; margin-bottom: 10px;">💧</div>
            <div class="card-title">Soil & Water Conservation</div>
            <div class="card-desc">Constructing trenching and contour bunds on Baner Hill to replenish Pune's groundwater table.</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 38px; margin-bottom: 10px;">📄</div>
            <div class="card-title">Digital Category Reports</div>
            <div class="card-desc">Generate downloadable image-category reports with a timestamp and selected location tag.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown('<div class="section-header">How VrikshaDrishti AI Works</div>', unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    w1.success("📷 1. Upload or Capture Photo")
    w2.success("🤖 2. EfficientNetB0 AI Analysis")
    w3.success("🌳 3. Review Category and Confidence")
    w4.success("📄 4. Download PDF Report")

# ---------------------------------------------------
# PAGE 2: PREDICT IMAGE CATEGORY (AI ENGINE)
# ---------------------------------------------------
elif page == "🔬 Predict Image Category (AI Engine)":
    st.markdown('<div class="section-header">🔬 Interactive AI Image Category Predictor</div>', unsafe_allow_html=True)
    st.caption("Audited EfficientNetB0 image-category model. Species and plant-health diagnosis are not supported.")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### 📷 Select Photo Input Method")
        input_mode = st.radio(
            "Choose photo source:",
            ["Option A: File Upload (JPG, PNG, WEBP)", "Option B: Live Camera Capture"],
            horizontal=True
        )

        image_file = None
        if "Option A" in input_mode:
            uploaded = st.file_uploader("Upload a tree or leaf photo:", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                image_file = uploaded
        else:
            cam_photo = st.camera_input("Capture a tree or leaf photo:")
            if cam_photo:
                image_file = cam_photo

        location_tag = st.selectbox(
            "📍 Select Baner Plantation Zone:",
            [
                "Baner Hill Biodiversity Park",
                "Baner Hill Trail - Phase 1",
                "Baner Hill Trail - Phase 2",
                "Pashan-Baner Link Road Greenbelt",
                "Mula-Mutha Riverbank Baner Zone"
            ]
        )

        if image_file:
            st.image(image_file, caption="Input Photo", use_container_width=True)

    with col_output:
        if image_file:
            with st.spinner("🤖 Running AI inference..."):
                try:
                    res = predict_tree_health(image_file)

                    if "scan_history" not in st.session_state:
                        st.session_state["scan_history"] = []

                    if not any(item.get("top_label") == res["top_label"] and item.get("confidence") == res["confidence"] for item in st.session_state["scan_history"]):
                        entry = res.copy()
                        entry["location"] = location_tag
                        st.session_state["scan_history"].insert(0, entry)

                    status_color = res["badge_color"]
                    st.markdown(
                        f'<div style="background:{status_color}; color:white !important; padding:12px 24px; border-radius:25px; font-weight:800; font-size:18px; display:inline-block; margin-bottom:15px; border:2px solid #FBC02D;">'
                        f'IMAGE CATEGORY: {res["status"].upper()} ({res["confidence"]}% Top-class Score)</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(f"**Image Category:** <span style='font-size:18px; font-weight:800; color:#0D3B16;'>{res['species']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Result:** <span style='font-size:17px; font-weight:800; color:#1B5E20;'>{res['diagnosis']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Health scope:** {res['symptoms']}")

                    st.markdown("##### 📈 Prediction Probability Breakdown")
                    t5_labels = [item[0].capitalize() for item in res["top5"]]
                    t5_scores = [item[1] for item in res["top5"]]

                    fig = px.bar(x=t5_scores, y=t5_labels, orientation='h', color=t5_scores, color_continuous_scale="Greens", labels={'x': 'Classifier score (%)', 'y': 'Category'})
                    fig.update_layout(height=240, margin=dict(l=5, r=5, t=5, b=5), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("🌿 About this result", expanded=True):
                        st.markdown(f"**Model scope:** {res['organic_remedy']}")
                        st.markdown(f"**Next step:** {res['pune_eco_tip']}")

                    pdf_data = generate_pdf_report(res, location_tag=location_tag)
                    st.download_button(
                        label="📄 Download Image Category Report (PDF)",
                        data=pdf_data,
                        file_name=f"VasundharaAbhiyan_Category_{res['species'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as ex:
                    st.error(f"Inference error: {str(ex)}")
        else:
            st.info("👈 Upload a photo or capture one to view its image category. This model does not diagnose plant health.")

# ---------------------------------------------------
# PAGE 3: ADMIN & IMPACT DASHBOARD
# ---------------------------------------------------
elif page == "📊 Admin & Impact Dashboard":
    stats = load_project_statistics()
    model_summary = load_model_summary()
    category_counts = stats.get("class_counts", {})
    total_unique = int(stats.get("source_unique_images", 0))
    validated_total = int(stats.get("validated_images", 0))
    model_comparison = load_model_comparison()
    st.markdown('<div class="section-header">📊 Real Dataset & Impact Analytics Dashboard</div>', unsafe_allow_html=True)
    duplicate_count = int(stats.get("source_valid_images", 0)) - total_unique
    st.caption(f"Primary Real Dataset: {stats.get('source_original_images', 0)} readable files, {total_unique} unique photos ({duplicate_count} exact duplicates removed); {validated_total} reviewed labels accepted for training, {stats.get('pending_review', 0)} pending review, {stats.get('excluded', 0)} exclusion decisions including duplicate copies. Species/health labels verified: {stats.get('species_verified_count', 0)}.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📷 Unique Primary Photos", f"{total_unique}", "exact duplicates removed")
    m2.metric("✅ Training-ready Labels", f"{validated_total}", f"{percentage(validated_total, total_unique)} of unique photos")
    m3.metric("🌳 Tree-focused", f"{category_counts.get('trees', 0)}", f"{percentage(category_counts.get('trees', 0), validated_total)} of reviewed labels")
    m4.metric("🍃 Leaf-focused", f"{category_counts.get('leafs', 0)}", f"{percentage(category_counts.get('leafs', 0), validated_total)} of reviewed labels")

    st.write("")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### 🍩 Real Dataset Category Breakdown")
        fig_pie = px.pie(
            names=[display_class_name(name) for name in ["trees", "leafs", "saplings"]],
            values=[category_counts.get(name, 0) for name in ["trees", "leafs", "saplings"]],
            color_discrete_sequence=["#1B5E20", "#2E7D32", "#FBC02D"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.markdown("#### 📈 Model Evaluation Confusion Matrix")
        cm_file = os.path.join("reports", "evaluation_v2", "confusion_matrix.png")
        if os.path.exists(cm_file):
            test = model_summary.get("test", {}) if model_summary else {}
            caption = (f"Grouped held-out test: accuracy {test.get('accuracy', 0):.1%}, macro F1 {test.get('macro_f1', 0):.3f}, n={sum(test.get('support', {}).values())}"
                       if test else "Grouped held-out test confusion matrix")
            st.image(cm_file, caption=caption, use_container_width=True)
        else:
            st.info("The grouped test confusion matrix will appear after model evaluation completes.")

    st.divider()

    st.markdown("#### 📑 Audited Dataset Split (filename variants grouped together)")
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
        st.markdown("#### ⚖️ Baseline vs. Training-only Augmentation")
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

# ---------------------------------------------------
# PAGE 4: CONTACT & NGO ENQUIRY
# ---------------------------------------------------
elif page == "📞 Contact & NGO Enquiry":
    st.markdown('<div class="section-header">📞 Contact Vasundhara Abhiyan NGO</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 28px; border-radius: 20px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); border-left: 8px solid #FBC02D; border-top: 1px solid #C8E6C9;">
            <h3 style="color:#0D3B16; margin-top:0;">🌿 Vasundhara Abhiyan NGO</h3>
            <p><b>🌐 Official Website:</b><br>
            <a href="https://vasundharaabhiyan.org/" target="_blank" style="color:#2E7D32; font-weight:800; font-size:16px;">https://vasundharaabhiyan.org/</a></p>

            <p><b>🌱 NGO Focus & Initiatives:</b><br>
            Native Tree Plantation, Baner Hill Eco Restoration, Groundwater Replenishment, Seed Ball Drives, Environmental Workshops.</p>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("### 📝 Volunteer Enquiry Form (Demo)")
        st.caption("This demo form does not store or send entries. Use the official website for real enquiries.")
        with st.form("enquiry_form"):
            v_name = st.text_input("Your Full Name:")
            v_phone = st.text_input("Phone Number:")
            v_email = st.text_input("Email Address:")
            v_interest = st.selectbox("Area of Interest:", ["Tree Plantation Drives on Baner Hill", "AI Leaf Health Scanning Volunteer", "Seed Ball Production", "Watering & Nursery Care"])
            v_msg = st.text_area("Your Message / Query:")
            submitted = st.form_submit_button("Preview Demo Enquiry")

            if submitted:
                if v_name and v_phone:
                    st.info("Thanks. This demo does not store or send your entry. Please use the official website link to contact Vasundhara Abhiyan.")
                else:
                    st.warning("Please fill in your name and phone number.")

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2026 Vasundhara Abhiyan | [vasundharaabhiyan.org](https://vasundharaabhiyan.org/)")
