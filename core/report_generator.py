import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class PDFReportGenerator:
    @staticmethod
    def compile_pdf(output_path, patient_info, diagnosis, confidence, ai_note, xray_img_path, affected_part="N/A", medication_advice="N/A"):
        doc = SimpleDocTemplate(
            output_path, pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'], fontSize=18, leading=22,
            textColor=colors.HexColor("#1a5276"), spaceAfter=2
        )
        section_heading = ParagraphStyle(
            'SectionHeading', parent=styles['Heading2'], fontSize=11, leading=14,
            textColor=colors.HexColor("#2e4053"), spaceBefore=10, spaceAfter=4, keepWithNext=True
        )
        body_style = ParagraphStyle(
            'Body', parent=styles['Normal'], fontSize=9, leading=14, textColor=colors.HexColor("#2c3e50")
        )
        alert_style = ParagraphStyle(
            'Alert', parent=styles['Normal'], fontSize=11, leading=14, 
            textColor=colors.HexColor("#c0392b") if diagnosis == "PNEUMONIA" else colors.HexColor("#27ae60"),
            fontName="Helvetica-Bold"
        )

        # Header Section
        story.append(Paragraph("METROPOLITAN DIGITAL HEALTH & IMAGING CENTRE", title_style))
        story.append(Paragraph("Automated Diagnostic Screening Report (Chest Radiography)", body_style))
        story.append(Spacer(1, 6))
        
        divider = Table([[""]], colWidths=[530], rowHeights=[2])
        divider.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1a5276"))]))
        story.append(divider)
        story.append(Spacer(1, 8))
        
        # Patient Metadata Table
        story.append(Paragraph("Patient Metadata", section_heading))
        patient_table_data = [
            [Paragraph(f"<b>Patient Name:</b> {patient_info['name']}", body_style), 
             Paragraph(f"<b>Date of Report:</b> {patient_info['date']}", body_style)],
            [Paragraph(f"<b>Age / Gender:</b> {patient_info['age']} / {patient_info['gender']}", body_style), 
             Paragraph(f"<b>Record ID:</b> {patient_info['id']}", body_style)]
        ]
        pt_table = Table(patient_table_data, colWidths=[265, 265])
        pt_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9f9")),
            ('PADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d5dbdb")),
        ]))
        story.append(pt_table)
        story.append(Spacer(1, 8))
        
        # Diagnostic Finding Matrix (X-Ray & Grad-CAM)
        story.append(Paragraph("Diagnostic Finding Matrix", section_heading))
        diagnostic_text_panel = [
            Paragraph("<b>Automated Classification:</b>", body_style),
            Spacer(1, 2),
            Paragraph(f"{diagnosis}", alert_style),
            Spacer(1, 6),
            Paragraph(f"<b>Anatomical Localization:</b>", body_style),
            Spacer(1, 1),
            Paragraph(f"<font color='#2980b9'><b>{affected_part}</b></font>", body_style),
            Spacer(1, 6),
            Paragraph(f"<b>Neural Certainty Index:</b> {confidence}%", body_style),
        ]
        
        try:
            xray_visual = Image(xray_img_path, width=105, height=105)
        except Exception:
            xray_visual = Paragraph("<font color='red'>[Raw Image Source Error]</font>", body_style)

        dir_name, file_name = os.path.split(xray_img_path)
        gradcam_img_path = os.path.join(dir_name, f"gradcam_{file_name}")
        
        try:
            if os.path.exists(gradcam_img_path) and diagnosis == "PNEUMONIA":
                gradcam_visual = Image(gradcam_img_path, width=105, height=105)
            else:
                gradcam_visual = Image(xray_img_path, width=105, height=105)
        except Exception:
            gradcam_visual = Paragraph("<font color='red'>[Grad-CAM Mapping Error]</font>", body_style)

        matrix_table_data = [[
            diagnostic_text_panel, 
            [Paragraph("<b>Input Image</b>", body_style), Spacer(1, 2), xray_visual], 
            [Paragraph("<b>Grad-CAM Focus</b>", body_style), Spacer(1, 2), gradcam_visual]
        ]]
        
        matrix_table = Table(matrix_table_data, colWidths=[260, 135, 135])
        matrix_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#eaeded")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('BACKGROUND', (1,0), (2,0), colors.HexColor("#fafafa"))
        ]))
        story.append(matrix_table)
        story.append(Spacer(1, 8))
        
        # Automated Medication Advisory Container
        story.append(Paragraph("💊 Automated Therapeutic Suggestions (Clinical Advisory)", section_heading))
        med_story_block = [
            Paragraph(f"<b>Suggested Regimen:</b> {medication_advice}", body_style),
            Spacer(1, 4),
            Paragraph("<font size='7' color='#7f8c8d'>*CRITICAL DISCLAIMER: This suggestion is generated automatically based on neural network classification targets. Exact dosages, selection pathways, and contraindications must be checked and signed off by a human physician before administration.</font>", body_style)
        ]
        med_table = Table([[med_story_block]], colWidths=[530])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eaf2f8") if diagnosis == "PNEUMONIA" else colors.HexColor("#e8f8f5")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINELEFT', (0,0), (0,0), 3, colors.HexColor("#2980b9") if diagnosis == "PNEUMONIA" else colors.HexColor("#27ae60")),
        ]))
        story.append(med_table)
        story.append(Spacer(1, 8))
        
        # AI Clinical Narrative Abstract Section
        story.append(Paragraph("AI Clinical Narrative Abstract", section_heading))
        med_advice_fallback = (
            f"For Community-Acquired Pneumonia in a {patient_info['age']}-year-old patient, standard first-line empirical antibiotic options generally include a macrolide (e.g., Azithromycin) or Doxycycline."
            if diagnosis == "PNEUMONIA" else 
            "No active empirical antimicrobial or respiratory therapies are indicated based on current normal diagnostic output."
        )
        
        ai_story_block = [
            Paragraph(f"<b>Clinical Impression:</b> {diagnosis} involving the {affected_part}.", body_style),
            Spacer(1, 2),
            Paragraph(f"<b>Key Observations:</b> Presence of localized airspace consolidation and/or opacities identified within the {affected_part}, consistent with an infectious inflammatory process.", body_style),
            Spacer(1, 2),
            Paragraph(f"<b>Standard Medication Guidelines:</b> {med_advice_fallback}", body_style),
            Spacer(1, 2),
            Paragraph("<b>Suggested Next Clinical Steps & Safety Disclaimer:</b> All drug selections, exact weights, dosages, and administration durations must be explicitly validated and authorized by the licensed attending physician based on comprehensive patient assessment.", body_style)
        ]
        ai_table = Table([[ai_story_block]], colWidths=[530])
        ai_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f4f6f7")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('LINELEFT', (0,0), (0,0), 3, colors.HexColor("#2e4053")),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 8))

        # 🟢 NEW: Integrated Neural Distribution Metrics Panels (Pie Chart & Bar Chart)
        report_file_id = f"Report_{patient_info['id']}.pdf"
        pie_path = os.path.join(dir_name, f"pie_{report_file_id}.png")
        bar_path = os.path.join(dir_name, f"bar_{report_file_id}.png")
        
        if os.path.exists(pie_path) and os.path.exists(bar_path):
            story.append(Paragraph("📊 Statistical Certainty Distribution & Validation Benchmarks (%)", section_heading))
            try:
                pie_img = Image(pie_path, width=150, height=105)
                bar_img = Image(bar_path, width=220, height=105)
                
                chart_table_data = [[pie_img, bar_img]]
                chart_table = Table(chart_table_data, colWidths=[240, 290])
                chart_table.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fafafa")),
                    ('PADDING', (0,0), (-1,-1), 6),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#eaeded"))
                ]))
                story.append(chart_table)
            except Exception:
                pass

        doc.build(story)