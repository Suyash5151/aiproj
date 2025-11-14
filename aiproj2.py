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
import re

# Configure API key
api_key = "AIzaSyBHySBPk3W9fcpuXujmGMc1C5F2BsPgIO8"
genai.configure(api_key=api_key)

# Configure page
st.set_page_config(page_title="Multi-PDF Synthesizer", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .section-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    .pdf-tag {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .equation-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📚 Multi-PDF Research Synthesizer with AI")
st.markdown("Upload multiple PDFs to generate a condensed research paper with proper equation extraction")

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    st.success("API Key configured!")
    st.info("Using Gemini 2.0 Flash")
    st.markdown("---")
    st.markdown("**🎯 Features**")
    st.markdown("""
    - 📄 Multi-PDF processing
    - 🔢 Advanced equation detection
    - 📊 Table extraction
    - 🖼️ Inline image placement
    - 🔬 Cross-document analysis
    - 📝 Concise synthesis (NO copy-paste)
    """)
    st.markdown("---")
    max_papers = st.slider("Target synthesis length (pages)", 4, 15, 8)
    include_images = st.checkbox("Include images", value=True)
    include_tables = st.checkbox("Include tables", value=True)
    synthesis_mode = st.radio(
        "Synthesis Mode",
        ["Concise Summary", "Detailed Analysis"],
        help="Concise mode creates shorter, focused summaries"
    )

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF files (select multiple)",
    type="pdf",
    accept_multiple_files=True,
    help="You can select 2-10 research papers for synthesis"
)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_equations_advanced(pdf_file):
    """Advanced equation extraction using pattern matching and context analysis"""
    equations = []
    pdf_file.seek(0)
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        
                        # Advanced equation detection criteria
                        has_math_ops = bool(re.search(r'[=+\-×÷∫∑√∂∆≈≤≥]', text))
                        has_numbers = bool(re.search(r'\d', text))
                        has_variables = bool(re.search(r'[a-zA-Z]', text))
                        has_subscript_superscript = bool(re.search(r'[₀-₉⁰-⁹]', text))
                        has_greek = bool(re.search(r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', text))
                        has_parentheses = '(' in text and ')' in text
                        
                        confidence_score = sum([
                            has_math_ops * 3,
                            has_numbers * 1,
                            has_variables * 2,
                            has_subscript_superscript * 2,
                            has_greek * 2,
                            has_parentheses * 1
                        ])
                        
                        is_short_enough = 5 < len(text) < 300
                        is_centered = block.get("bbox", [0, 0, 0, 0])[0] > 100
                        
                        if confidence_score >= 5 and is_short_enough:
                            context = ""
                            try:
                                full_text = page.get_text()
                                text_index = full_text.find(text)
                                if text_index != -1:
                                    context_start = max(0, text_index - 100)
                                    context_end = min(len(full_text), text_index + len(text) + 100)
                                    context = full_text[context_start:context_end].replace('\n', ' ')
                            except:
                                pass
                            
                            equations.append({
                                "equation": text,
                                "page": page_num + 1,
                                "confidence": confidence_score,
                                "context": context,
                                "position": "centered" if is_centered else "inline"
                            })
    
    # Remove duplicates
    seen = set()
    unique_equations = []
    for eq in sorted(equations, key=lambda x: x['confidence'], reverse=True):
        eq_text = eq['equation'].strip()
        if eq_text not in seen and len(eq_text) > 5:
            seen.add(eq_text)
            unique_equations.append(eq)
    
    return unique_equations

def extract_tables_from_pdf(pdf_file):
    """Extract tables from PDF"""
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
    """Extract meaningful images from PDF"""
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
                image = Image.open(io.BytesIO(image_bytes))
                
                if image.width > 150 and image.height > 150:
                    if is_meaningful_image(image):
                        images.append({
                            "image": image,
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "bytes": image_bytes
                        })
            except:
                pass
    
    return images

def is_meaningful_image(image):
    """Check if image is meaningful"""
    import numpy as np
    
    img_array = np.array(image.convert('RGB'))
    color_variance = np.var(img_array)
    
    from PIL import ImageFilter
    edges = image.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_density = np.count_nonzero(edge_array > 30) / edge_array.size
    
    is_complex = color_variance > 100
    has_content = edge_density > 0.05
    is_large_enough = image.width > 150 and image.height > 150
    
    return is_complex and has_content and is_large_enough

def analyze_images_batch(images):
    """Analyze multiple images in ONE API call"""
    if not images:
        return []
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare all images for batch processing
        image_parts = []
        for idx, img_data in enumerate(images):
            img_byte_arr = io.BytesIO()
            img_data['image'].save(img_byte_arr, format='PNG')
            image_parts.append({
                "mime_type": "image/png",
                "data": img_byte_arr.getvalue()
            })
        
        # Single API call for all images
        prompt = f"""Analyze these {len(images)} images from research papers. For EACH image, provide a 1-2 sentence description.

Format your response as:
IMAGE 1: [description]
IMAGE 2: [description]
...

For graphs/charts: describe key trends and data.
For diagrams: describe main components.
For photos: describe what it depicts.
If decorative/simple icon: write "DECORATIVE_ELEMENT"

Keep each description concise and factual."""
        
        content = [prompt] + image_parts
        response = model.generate_content(content)
        
        # Parse response
        descriptions = []
        response_text = response.text
        for i in range(len(images)):
            pattern = f"IMAGE {i+1}:(.+?)(?=IMAGE {i+2}:|$)"
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                desc = match.group(1).strip()
                descriptions.append(desc)
            else:
                descriptions.append("Description unavailable")
        
        return descriptions
    
    except Exception as e:
        return ["Description unavailable"] * len(images)

def synthesize_multiple_pdfs(all_texts, all_equations, all_tables, all_images, target_pages, mode):
    """Synthesize multiple PDFs - CRITICAL: Tell Gemini NOT to copy-paste"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        num_pdfs = len(all_texts)
        
        # Create condensed input
        condensed_input = ""
        for idx, text in enumerate(all_texts):
            sample = text[:15000]
            condensed_input += f"\n\n=== DOCUMENT {idx + 1} ===\n{sample}"
        
        # Add equation context
        eq_summary = f"\n\n=== KEY EQUATIONS ({len(all_equations)} total) ===\n"
        for eq in all_equations[:20]:
            eq_summary += f"- {eq['equation']} (Page {eq['page']})\n"
        
        target_words = target_pages * 400 if mode == "Concise Summary" else target_pages * 500
        
        prompt = f"""You are creating a CONCISE RESEARCH SYNTHESIS of {num_pdfs} papers (~{target_words} words, {target_pages} pages).

🚨 CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. DO NOT copy-paste sentences from the original papers
2. DO NOT reproduce long excerpts or paragraphs verbatim
3. WRITE IN YOUR OWN WORDS - synthesize and paraphrase
4. Focus on KEY INSIGHTS, not detailed reproductions
5. Create an INFOGRAPHIC-STYLE summary (visual, concise, impactful)

FORMAT (use plain text headers, NO ** markers):

TITLE: [Comprehensive title covering all papers]

ABSTRACT:
[150-200 words: What papers collectively address, methods, key findings]

1. INTRODUCTION
[300-400 words: Background, research gap, objectives]

2. METHODOLOGY
[300-400 words: Synthesized approaches, key techniques]

3. KEY FINDINGS
[400-600 words: Main results, agreements/contradictions, unified conclusions]
[MENTION: "See Figure 1", "Table 1 shows", etc. where images/tables would fit]

4. DISCUSSION
[300-400 words: Implications, limitations, how papers complement each other]

5. CONCLUSIONS
[200-300 words: Unified conclusions, future directions]

6. KEY EQUATIONS
[List 5-10 most important equations with brief explanations]

STYLE REQUIREMENTS:
✅ Paraphrase everything in your own words
✅ Use phrases like: "The research demonstrates...", "Findings indicate...", "Analysis reveals..."
✅ Be concise and impactful
✅ Mention where figures/tables should appear (e.g., "Figure 1 illustrates...")
✅ Academic but readable tone

❌ DO NOT copy-paste original sentences
❌ DO NOT use ** for emphasis
❌ DO NOT reproduce lengthy excerpts
❌ DO NOT just summarize - SYNTHESIZE across papers

Documents:
{condensed_input}

{eq_summary}

Create the synthesis now:"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating synthesis: {str(e)}"

def format_summary_for_display(summary):
    """Format summary with HTML"""
    summary = summary.replace('**', '')
    
    # Format TITLE
    summary = re.sub(r'TITLE:\s*(.*?)(\n|$)', r'<div class="section-header">\1</div>\n', summary)
    
    # Format numbered sections
    summary = re.sub(r'(\d+\.\s+[A-Z][A-Z\s&/]+):?\s*(\n|$)', r'<div class="section-header">\1</div>\n', summary)
    
    # Format main headers
    summary = re.sub(
        r'^(ABSTRACT|INTRODUCTION|METHODOLOGY|KEY FINDINGS|DISCUSSION|CONCLUSIONS|KEY EQUATIONS):?\s*(\n|$)',
        r'<div class="section-header">\1</div>\n',
        summary,
        flags=re.MULTILINE
    )
    
    # Format equations
    summary = re.sub(r'`([^`]+)`', r'<div class="equation-box">\1</div>', summary)
    
    return summary

def create_synthesis_pdf_with_inline_images(synthesis, all_equations, all_tables, all_images, image_summaries, pdf_names):
    """Create PDF with images INLINE with text"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_file.name, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    temp_image_files = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.black,
        spaceAfter=20,
        alignment=1,
        fontName='Helvetica-Bold'
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=18,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=8,
        leading=14,
        alignment=4
    )
    
    # Add header
    story.append(Paragraph("RESEARCH SYNTHESIS", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Synthesized from {len(pdf_names)} papers", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Parse synthesis and insert images inline
    lines = synthesis.split('\n')
    image_index = 0
    table_index = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        clean_line = line.replace('**', '')
        
        # Check for figure mentions and insert image
        if re.search(r'(Figure|Fig\.?)\s*\d+', clean_line, re.IGNORECASE):
            story.append(Paragraph(clean_line, body_style))
            
            # Insert corresponding image
            if image_index < len(all_images):
                img_data = all_images[image_index]
                
                # Save temp image
                img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img_data['image'].save(img_temp.name, format="PNG")
                temp_image_files.append(img_temp.name)
                
                # Calculate size
                img = img_data['image']
                aspect = img.height / float(img.width)
                img_width = min(img.width, 4.5*inch)
                img_height = img_width * aspect
                if img_height > 5*inch:
                    img_height = 5*inch
                    img_width = img_height / aspect
                
                # Add image
                story.append(Spacer(1, 0.1*inch))
                rl_image = RLImage(img_temp.name, width=img_width, height=img_height)
                story.append(rl_image)
                
                # Add caption
                if image_index < len(image_summaries):
                    caption = f"Figure {image_index + 1}: {image_summaries[image_index]}"
                    story.append(Paragraph(caption, styles['Italic']))
                
                story.append(Spacer(1, 0.15*inch))
                image_index += 1
        
        # Check for table mentions
        elif re.search(r'Table\s*\d+', clean_line, re.IGNORECASE):
            story.append(Paragraph(clean_line, body_style))
            
            # Insert corresponding table
            if table_index < len(all_tables):
                table_data = all_tables[table_index]
                df = table_data['data'].head(10).iloc[:, :5]
                
                table_list = [df.columns.tolist()] + df.values.tolist()
                table_list = [[str(cell)[:20] for cell in row] for row in table_list]
                
                t = RLTable(table_list, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                ]))
                
                story.append(Spacer(1, 0.1*inch))
                story.append(t)
                story.append(Spacer(1, 0.15*inch))
                table_index += 1
        
        # Section headers
        elif (clean_line.startswith('TITLE:') or 
              re.match(r'\d+\.\s+[A-Z]', clean_line) or
              clean_line in ['ABSTRACT', 'INTRODUCTION', 'METHODOLOGY', 'KEY FINDINGS', 
                           'RESULTS', 'DISCUSSION', 'CONCLUSIONS', 'KEY EQUATIONS']):
            section_text = clean_line.replace('TITLE:', '').strip().rstrip(':')
            story.append(Paragraph(section_text, section_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Regular text
        else:
            story.append(Paragraph(clean_line, body_style))
    
    # Build PDF
    doc.build(story)
    
    # Cleanup
    for temp_file in temp_image_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return tmp_file.name

# Main processing
if uploaded_files and len(uploaded_files) > 0:
    st.markdown('<div class="section-header">📚 Processing PDFs</div>', unsafe_allow_html=True)
    
    # Storage
    all_texts = []
    all_equations = []
    all_tables = []
    all_images = []
    pdf_names = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each PDF
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        pdf_names.append(uploaded_file.name)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f'<span class="pdf-tag">📄 {uploaded_file.name}</span>', unsafe_allow_html=True)
        
        # Extract text
        pdf_text = extract_text_from_pdf(uploaded_file)
        all_texts.append(pdf_text)
        
        # Extract equations
        uploaded_file.seek(0)
        equations = extract_equations_advanced(uploaded_file)
        all_equations.extend(equations)
        
        # Extract tables
        if include_tables:
            uploaded_file.seek(0)
            tables = extract_tables_from_pdf(uploaded_file)
            all_tables.extend(tables)
        
        # Extract images
        if include_images:
            uploaded_file.seek(0)
            images = extract_images_from_pdf(uploaded_file)
            all_images.extend(images)
        
        with col2:
            st.caption(f"✅ {len(equations)} equations")
            if include_tables:
                st.caption(f"📊 {len(tables)} tables")
            if include_images:
                st.caption(f"🖼️ {len(images)} images")
    
    progress_bar.empty()
    status_text.empty()
    
    # Batch analyze ALL images in ONE API call
    if all_images:
        st.info("🖼️ Analyzing images (single batch API call)...")
        all_image_summaries = analyze_images_batch(all_images)
        
        # Filter out decorative images
        meaningful_images = []
        meaningful_summaries = []
        for img_data, summary in zip(all_images, all_image_summaries):
            if "DECORATIVE_ELEMENT" not in summary:
                meaningful_images.append(img_data)
                meaningful_summaries.append(summary)
        
        all_images = meaningful_images
        all_image_summaries = meaningful_summaries
        
        st.success(f"✅ Kept {len(all_images)} meaningful images")
    else:
        all_image_summaries = []
    
    st.markdown("---")
    
    # Generate synthesis
    st.markdown('<div class="section-header">🔬 Generating Research Synthesis</div>', unsafe_allow_html=True)
    
    with st.spinner("Creating intelligent synthesis... (this will NOT copy-paste from papers)"):
        synthesis = synthesize_multiple_pdfs(
            all_texts,
            all_equations,
            all_tables,
            all_images,
            max_papers,
            synthesis_mode
        )
    
    # Display synthesis
    st.markdown('<div class="section-header">📑 Synthesized Research Paper</div>', unsafe_allow_html=True)
    
    # Show source PDFs
    st.markdown("**Source Documents:**")
    for name in pdf_names:
        st.markdown(f'<span class="pdf-tag">{name}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    formatted_synthesis = format_summary_for_display(synthesis)
    st.markdown(formatted_synthesis, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PDFs", len(uploaded_files))
    with col2:
        st.metric("Equations", len(all_equations))
    with col3:
        st.metric("Tables", len(all_tables))
    with col4:
        st.metric("Images", len(all_images))
    
    # Download PDF
    st.markdown("---")
    st.markdown('<div class="section-header">💾 Download Complete Report</div>', unsafe_allow_html=True)
    
    with st.spinner("Creating professional PDF with inline images..."):
        pdf_path = create_synthesis_pdf_with_inline_images(
            synthesis,
            all_equations,
            all_tables,
            all_images,
            all_image_summaries,
            pdf_names
        )
    
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Synthesis PDF",
            data=f.read(),
            file_name="research_synthesis.pdf",
            mime="application/pdf",
            help="Professional synthesis with inline figures and tables"
        )
    
    os.unlink(pdf_path)

else:
    st.markdown('<div class="section-header">👋 Welcome!</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload **2-10 research papers** to generate an intelligent synthesis that:
    
    ✨ **Paraphrases and synthesizes** (no copy-paste!)  
    🖼️ **Places images inline** with relevant text  
    🔢 **Extracts key equations**  
    📊 **Includes tables**  
    🔬 **Cross-analyzes findings**
    
    ### How to use:
    1. Upload multiple PDF research papers
    2. Configure settings in the sidebar
    3. Wait for AI synthesis
    4. Download professional PDF report
    """)
    
    with st.expander("ℹ️ Technical Details"):
        st.markdown("""
        **Optimizations:**
        - Single batch API call for all image analysis (reduces API calls)
        - Inline image and table placement in PDF
        - Intelligent paraphrasing (no copy-paste from sources)
        - Advanced equation detection with confidence scoring
        
        **Features:**
        - Multi-document synthesis
        - Equation extraction with context
        - Table extraction and formatting
        - Image filtering (removes decorative elements)
        - Professional PDF generation
        """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Gemini AI")