import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table as RLTable, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.flowables import Image as RLImage
import tempfile
import pandas as pd
import tabulate
import base64

# Configure API key
api_key = "AIzaSyCMMOPf9e4fPUamtxoX50_LcHECE3L4w80"
genai.configure(api_key=api_key)

# Configure page
st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")

# Title
st.title("📄 PDF Summarizer with Gemini AI")
st.markdown("Upload a PDF to get an AI-powered summary and extract images, tables, equations, and graphs")

# Sidebar info
with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("API Key configured!")
    st.info("Using Gemini 2.0 Flash model")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_equations_from_pdf(pdf_file):
    """Extract potential equations from PDF"""
    equations = []
    pdf_file.seek(0)
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        
        # Look for common equation patterns
        lines = text.split('\n')
        for line in lines:
            # Check if line contains mathematical symbols or operators
            math_symbols = ['=', '+', '-', '×', '÷', '∫', '∑', '√', '∂', '∆', '≈', '≤', '≥', '^', '²', '³']
            if any(symbol in line for symbol in math_symbols) and len(line.strip()) > 3:
                # Filter out very long lines (likely not equations)
                if len(line.strip()) < 200:
                    equations.append({
                        "equation": line.strip(),
                        "page": page_num + 1
                    })
    
    return equations

def extract_tables_from_pdf(pdf_file):
    """Extract tables from PDF using PyMuPDF"""
    tables = []
    pdf_file.seek(0)
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        tabs = page.find_tables()
        
        for tab_index, tab in enumerate(tabs):
            try:
                df = tab.to_pandas()
                if df is not None and not df.empty:
                    tables.append({
                        "data": df,
                        "page": page_num + 1,
                        "index": tab_index + 1
                    })
            except:
                pass
    
    return tables

def extract_images_from_pdf(pdf_file):
    """Extract images and graphs from PDF using PyMuPDF"""
    images = []
    pdf_file.seek(0)
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter out very small images (likely icons or decorations)
                if image.width > 100 and image.height > 100:
                    images.append({
                        "image": image,
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "bytes": image_bytes
                    })
            except:
                pass
    
    return images

def analyze_image_with_gemini(image):
    """Analyze image using Gemini Vision API"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create prompt for image analysis
        prompt = """Analyze this image and provide a concise summary (2-3 sentences). 
        If it's a graph or chart, describe what data it shows, the axes, and key trends. 
        If it's a diagram, describe the main components and their relationships.
        If it's a photo or illustration, describe what it depicts."""
        
        # Upload image and generate content
        response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        return response.text
    except Exception as e:
        return f"Could not analyze image: {str(e)}"

def summarize_with_gemini(text):
    """Summarize text using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Please provide a comprehensive summary of the following document. 
        Include key points, main themes, and important details:
        
        {text[:30000]}"""  # Limit text to avoid token limits
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def create_pdf_with_summary(summary, images, tables, equations, image_summaries):
    """Create a downloadable PDF with summary, images, tables, and equations"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    temp_image_files = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    equation_style = ParagraphStyle(
        'EquationStyle',
        parent=styles['Code'],
        fontSize=11,
        textColor=colors.HexColor('#2c5282'),
        leftIndent=20,
        spaceAfter=6
    )
    
    # Title
    story.append(Paragraph("PDF Summary Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Summary section
    story.append(Paragraph("Summary", heading_style))
    summary_paragraphs = summary.split('\n')
    for para in summary_paragraphs:
        if para.strip():
            story.append(Paragraph(para, styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Equations section
    if equations:
        story.append(PageBreak())
        story.append(Paragraph(f"Extracted Equations ({len(equations)} total)", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        for idx, eq_data in enumerate(equations[:50]):  # Limit to 50 equations
            story.append(Paragraph(f"Equation {idx + 1} (from Page {eq_data['page']})", styles['Heading3']))
            story.append(Spacer(1, 0.05*inch))
            story.append(Paragraph(eq_data['equation'], equation_style))
            story.append(Spacer(1, 0.15*inch))
    
    # Tables section
    if tables:
        story.append(PageBreak())
        story.append(Paragraph(f"Extracted Tables ({len(tables)} total)", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        for idx, table_data in enumerate(tables):
            story.append(Paragraph(f"Table {idx + 1} (from Page {table_data['page']})", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            df = table_data['data']
            df_display = df.head(20).iloc[:, :8]
            
            table_list = [df_display.columns.tolist()] + df_display.values.tolist()
            table_list = [[str(cell)[:30] for cell in row] for row in table_list]
            
            t = RLTable(table_list, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            
            story.append(t)
            story.append(Spacer(1, 0.3*inch))
    
    # Images section with AI summaries
    if images:
        story.append(PageBreak())
        story.append(Paragraph(f"Extracted Images and Graphs ({len(images)} total)", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        for idx, img_data in enumerate(images):
            story.append(Paragraph(f"Image {idx + 1} (from Page {img_data['page']})", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Add AI summary of the image
            if idx < len(image_summaries):
                story.append(Paragraph(f"<b>AI Analysis:</b> {image_summaries[idx]}", styles['BodyText']))
                story.append(Spacer(1, 0.1*inch))
            
            # Save image to temporary file
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_data['image'].save(img_temp.name, format="PNG")
            temp_image_files.append(img_temp.name)
            
            # Add image to PDF with proper sizing
            img = img_data['image']
            aspect = img.height / float(img.width)
            
            max_width = 6 * inch
            img_width = min(img.width, max_width)
            img_height = img_width * aspect
            
            if img_height > 7 * inch:
                img_height = 7 * inch
                img_width = img_height / aspect
            
            rl_image = RLImage(img_temp.name, width=img_width, height=img_height)
            story.append(rl_image)
            story.append(Spacer(1, 0.3*inch))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temp image files
    for temp_file in temp_image_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return tmp_file.name

# Main processing
if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text
        st.subheader("📖 Extracting Text...")
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        if pdf_text:
            st.success(f"Extracted {len(pdf_text)} characters")
            
            # Generate summary
            st.subheader("🤖 Generating Summary with Gemini AI...")
            summary = summarize_with_gemini(pdf_text)
            
            # Display summary
            st.subheader("📝 Summary")
            st.markdown(summary)
            
            # Extract equations
            st.subheader("🔢 Extracting Equations...")
            uploaded_file.seek(0)
            equations = extract_equations_from_pdf(uploaded_file)
            
            if equations:
                st.success(f"Found {len(equations)} equations")
                
                with st.expander("📐 View All Equations"):
                    for idx, eq_data in enumerate(equations[:30]):  # Show first 30
                        st.markdown(f"**Page {eq_data['page']}:** `{eq_data['equation']}`")
            else:
                st.info("No equations found in the PDF")
            
            # Extract tables
            st.subheader("📊 Extracting Tables...")
            uploaded_file.seek(0)
            tables = extract_tables_from_pdf(uploaded_file)
            
            if tables:
                st.success(f"Found {len(tables)} tables")
                
                for idx, table_data in enumerate(tables):
                    with st.expander(f"📋 Table {idx + 1} (Page {table_data['page']})"):
                        st.dataframe(table_data['data'])
                        
                        csv = table_data['data'].to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"table_{idx+1}_page_{table_data['page']}.csv",
                            mime="text/csv",
                            key=f"table_{idx}"
                        )
            else:
                st.info("No tables found in the PDF")
            
            # Extract images
            st.subheader("🖼️ Extracting Images and Graphs...")
            uploaded_file.seek(0)
            images = extract_images_from_pdf(uploaded_file)
            
            image_summaries = []
            
            if images:
                st.success(f"Found {len(images)} images/graphs")
                
                # Analyze images with Gemini Vision
                st.subheader("🔍 Analyzing Images with AI...")
                progress_bar = st.progress(0)
                
                for idx, img_data in enumerate(images):
                    with st.spinner(f"Analyzing image {idx + 1}/{len(images)}..."):
                        img_summary = analyze_image_with_gemini(img_data['image'])
                        image_summaries.append(img_summary)
                    progress_bar.progress((idx + 1) / len(images))
                
                progress_bar.empty()
                
                # Display images with AI summaries
                cols = st.columns(2)
                for idx, img_data in enumerate(images):
                    with cols[idx % 2]:
                        st.image(img_data["image"], 
                                caption=f"Page {img_data['page']}, Image {img_data['index']}",
                                use_container_width=True)
                        
                        # Show AI analysis
                        st.markdown(f"**🤖 AI Analysis:** {image_summaries[idx]}")
                        
                        # Download individual image
                        buf = io.BytesIO()
                        img_data["image"].save(buf, format="PNG")
                        st.download_button(
                            label="Download Image",
                            data=buf.getvalue(),
                            file_name=f"image_p{img_data['page']}_{img_data['index']}.png",
                            mime="image/png",
                            key=f"img_{idx}"
                        )
                        st.markdown("---")
            else:
                st.info("No images found in the PDF")
            
            # Create downloadable summary PDF
            st.subheader("💾 Download Complete Report")
            with st.spinner("Creating PDF report with all content..."):
                pdf_path = create_pdf_with_summary(summary, images, tables, equations, image_summaries)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Complete Summary PDF (Summary + Equations + Tables + Images with AI Analysis)",
                    data=f.read(),
                    file_name="pdf_summary_report.pdf",
                    mime="application/pdf"
                )
            
            # Clean up temp file
            os.unlink(pdf_path)
            
        else:
            st.error("Could not extract text from PDF")

else:
    st.info("👆 Upload a PDF file to get started")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini AI")