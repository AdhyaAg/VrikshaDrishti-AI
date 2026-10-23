import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(prediction_data, location_tag="Baner Biodiversity Park, Pune"):
    """
    Generates a PDF image-category report in memory. It does not make a
    botanical species or plant-health diagnosis.
    Returns bytes buffer ready for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#1E5128'),
        alignment=1,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#4E9F3D'),
        alignment=1,
        spaceAfter=20
    )

    heading2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1E5128'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#222222'),
        leading=14
    )

    elements = []

    # Title & Branding
    elements.append(Paragraph("VRIKSHADRISHTI AI - IMAGE CATEGORY REPORT", title_style))
    elements.append(Paragraph("Vasundhara Abhiyan Baner, Pune | Image category only; no health diagnosis", subtitle_style))
    elements.append(Spacer(1, 10))

    # Meta Details Table
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_text = prediction_data["status"].upper()
    status_color = prediction_data.get("badge_color", "#2E7D32")

    meta_data = [
        [Paragraph("<b>Inspection Date:</b>", body_style), Paragraph(now_str, body_style)],
        [Paragraph("<b>Location Tag:</b>", body_style), Paragraph(location_tag, body_style)],
        [Paragraph("<b>Image Category:</b>", body_style), Paragraph(prediction_data["species"], body_style)],
        [Paragraph("<b>Classifier Result:</b>", body_style), Paragraph(prediction_data["diagnosis"], body_style)],
        [Paragraph("<b>Category Status:</b>", body_style), Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", body_style)],
        [Paragraph("<b>Uncalibrated Classifier Score:</b>", body_style), Paragraph(f"{prediction_data['confidence']}%", body_style)]
    ]

    t_meta = Table(meta_data, colWidths=[150, 380])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F9F4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D8E9A8')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 15))

    # Image embedding
    if "pil_image" in prediction_data and prediction_data["pil_image"] is not None:
        img_buffer = io.BytesIO()
        prediction_data["pil_image"].save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        rl_img = RLImage(img_buffer, width=180, height=180)
        elements.append(Paragraph("<b>Input Image:</b>", heading2_style))
        elements.append(rl_img)
        elements.append(Spacer(1, 15))

    # Symptoms Section
    elements.append(Paragraph("Model Scope:", heading2_style))
    elements.append(Paragraph(prediction_data["symptoms"], body_style))
    elements.append(Spacer(1, 10))

    # Recommended Remedies Table
    elements.append(Paragraph("Scope & Suggested Next Step:", heading2_style))
    rec_table_data = [
        [Paragraph("<b>Not a diagnosis:</b>", body_style), Paragraph(prediction_data["organic_remedy"], body_style)],
        [Paragraph("<b>Treatment:</b>", body_style), Paragraph(prediction_data["chemical_remedy"], body_style)],
        [Paragraph("<b>Next step:</b>", body_style), Paragraph(prediction_data["pune_eco_tip"], body_style)],
    ]
    t_rec = Table(rec_table_data, colWidths=[150, 380])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#4E9F3D')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_rec)
    elements.append(Spacer(1, 20))

    # Footer Notes
    elements.append(Paragraph("For organization information, visit vasundharaabhiyan.org.", body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
