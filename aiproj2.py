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
api_key = "AIzaSyCMMOPf9e4fPUamtxoX50_LcHECE3L4w80"
genai.configure(api_key=api_key)

# Configure page
st.set_page_config(page_title="Multi-PDF Synthesizer", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 28px !important;
    font-weight: bold !important;
    color: #000000 !important;
    margin-bottom: 10px !important;
}
.section-header {
    font-size: 22px !important;
    font-weight: bold !important;
    color: #000000 !important;
    margin-top: 20px !important;
    margin-bottom: 10px !important;
}
.subsection-header {
    font-size: 18px !important;
    font-weight: bold !important;
    color: #000000 !important;
    margin-top: 15px !important;
}
.equation-box {
    background-color: #f7fafc;
    border-left: 4px solid #000000;
    padding: 15px;
    margin: 10px 0;
    font-family: 'Courier New', monospace;
    font-size: 14px;
}
.pdf-badge {
    display: inline-block;
    background-color: #e6e6e6;
    color: #000000;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    margin: 2px;
    font-weight: 600;
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
    st.markdown("**Features:**")
    st.markdown("✅ Multi-PDF processing")
    st.markdown("✅ Advanced equation detection")
    st.markdown("✅ Condensed synthesis")
    st.markdown("✅ Research paper format")
    st.markdown("✅ Cross-document analysis")

    st.markdown("---")
    max_papers = st.slider("Target paper length (pages)", 4, 15, 8)
    include_images = st.checkbox("Include images in synthesis", value=True)
    include_tables = st.checkbox("Include tables in synthesis", value=True)

# File uploader - MULTIPLE FILES
uploaded_files = st.file_uploader(
    "Choose PDF files (you can select multiple)",
    type="pdf",
    accept_multiple_files=True
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

        # Get text blocks with positioning
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

                        # Check if it looks like an equation
                        confidence_score = sum([
                            has_math_ops * 3,
                            has_numbers * 1,
                            has_variables * 2,
                            has_subscript_superscript * 2,
                            has_greek * 2,
                            has_parentheses * 1
                        ])

                        # Filter criteria
                        is_short_enough = 5 < len(text) < 300
                        is_not_sentence = not text.endswith('.')
                        is_centered = block.get("bbox", [0, 0, 0, 0])[0] > 100  # Not at left margin

                        if confidence_score >= 5 and is_short_enough:
                            # Extract surrounding context
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

    # Remove duplicates and sort by confidence
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
    """Extract meaningful images from PDF (filters out decorative elements)"""
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

                # More strict filtering for meaningful images
                if image.width > 150 and image.height > 150:
                    # Check if image is meaningful
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
    """Check if image is meaningful (not just decorative shapes or icons)"""
    import numpy as np

    # Convert to numpy array
    img_array = np.array(image.convert('RGB'))

    # Calculate color variance
    color_variance = np.var(img_array)

    # Calculate edge density (complexity)
    from PIL import ImageFilter
    edges = image.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_density = np.count_nonzero(edge_array > 30) / edge_array.size

    # Meaningful images have:
    # - Higher color variance (not just solid colors)
    # - Reasonable edge density (actual content, not simple shapes)
    # - Sufficient size
    is_complex = color_variance > 100
    has_content = edge_density > 0.05
    is_large_enough = image.width > 150 and image.height > 150

    return is_complex and has_content and is_large_enough


def analyze_image_with_gemini(image):
    """Analyze image using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        prompt = """Analyze this image concisely (2-3 sentences). 
        For graphs/charts: describe data, axes, and key trends.
        For diagrams: describe components and relationships.
        For photos: describe what it depicts.
        If this is just a simple shape, icon, or decorative element, respond with: "DECORATIVE_ELEMENT"."""

        response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_arr}])
        return response.text
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"


def synthesize_multiple_pdfs(all_texts, all_equations, all_tables, all_images, target_pages):
    """Synthesize multiple PDFs into a condensed research paper"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Prepare content summary
        total_chars = sum(len(text) for text in all_texts)
        num_pdfs = len(all_texts)

        # Create condensed input (sample from each document)
        condensed_input = ""
        for idx, text in enumerate(all_texts):
            # Take first 15000 chars from each PDF
            sample = text[:15000]
            condensed_input += f"\n\n=== DOCUMENT {idx + 1} ===\n{sample}"

        # Add equation context
        eq_summary = f"\n\n=== EQUATIONS FOUND ({len(all_equations)} total) ===\n"
        for eq in all_equations[:20]:  # Top 20 equations
            eq_summary += f"- {eq['equation']} (confidence: {eq['confidence']})\n"

        prompt = f"""You are synthesizing {num_pdfs} research papers into ONE condensed research paper of approximately {target_pages} pages.

**CRITICAL INSTRUCTIONS:**
1. Create a COMPREHENSIVE SYNTHESIS, not separate summaries
2. Identify COMMON THEMES across all documents
3. Highlight DIFFERENCES and COMPLEMENTARY findings
4. Create a COHESIVE NARRATIVE that integrates all sources
5. Target length: ~{target_pages * 500} words (about {target_pages} pages)

**FORMAT YOUR SYNTHESIS AS:**

TITLE: [Create a comprehensive title covering all papers]

ABSTRACT:
[Synthesized overview of all papers - what they collectively address, methods, and findings. 150-200 words]

1. INTRODUCTION
[Background and motivation from all papers. Research gap identified across papers. Collective objectives. Structure of this synthesis. 400-600 words]

2. LITERATURE REVIEW & RELATED WORK
[Common theoretical foundations. Related work mentioned across papers. Gap analysis. 300-500 words]

3. METHODOLOGY
[Synthesize approaches used across papers. Compare and contrast methods. Data sources and collection methods. Analysis techniques. 400-600 words]

4. KEY FINDINGS & RESULTS
[Synthesize main results from all papers. Highlight agreements and contradictions. Present unified conclusions where possible. Note unique contributions of each paper. 600-800 words]

5. DISCUSSION
[Implications of combined findings. Limitations across papers. How papers complement each other. Gaps remaining. 400-600 words]

6. CONCLUSIONS & FUTURE WORK
[Unified conclusions. Recommendations based on all papers. Future research directions. 300-400 words]

7. KEY EQUATIONS
[List the most important equations with explanations. Show relationships between equations from different papers]

**IMPORTANT:**
- DO NOT use ** markers for section headers
- Section headers should be plain text like: "1. INTRODUCTION" or "ABSTRACT:"
- Use "we observe across the literature" or "the papers collectively show"
- Compare findings: "Paper A found X while Paper B demonstrated Y"
- Synthesize: "Combining these approaches suggests..."
- Keep academic tone
- Use transition phrases to connect ideas
- Be concise but comprehensive

Documents to synthesize:
{condensed_input}

{eq_summary}

Now create the synthesis:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating synthesis: {str(e)}"


def format_summary_for_display(summary):
    """Format summary with HTML - removes ** markers and applies styling"""
    # Remove all ** markers first
    summary = summary.replace('**', '')

    # Format TITLE: pattern
    summary = re.sub(r'TITLE:\s*(.*?)(\n|$)',
                     r'<div class="big-title">\1</div>\n', summary)

    # Format numbered sections like "1. INTRODUCTION:"
    summary = re.sub(r'(\d+\.\s+[A-Z][A-Z\s&/]+):?\s*(\n|$)',
                     r'<div class="section-header">\1</div>\n', summary)

    # Format ABSTRACT, METHODOLOGY, etc.
    summary = re.sub(
        r'^(ABSTRACT|INTRODUCTION|LITERATURE REVIEW|RELATED WORK|METHODOLOGY|KEY FINDINGS|RESULTS|FINDINGS|DISCUSSION|CONCLUSIONS|CONCLUSION|FUTURE WORK|KEY EQUATIONS|KEY FORMULAS):?\s*(\n|$)',
        r'<div class="section-header">\1</div>\n', summary, flags=re.MULTILINE)

    # Format subsection headers (Title Case followed by colon)
    summary = re.sub(r'^([A-Z][a-zA-Z\s]+):(\s*\n)',
                     r'<div class="subsection-header">\1</div>\2', summary, flags=re.MULTILINE)

    return summary


def create_synthesis_pdf(synthesis, all_equations, all_tables, all_images, image_summaries, pdf_names):
    """Create professional research synthesis PDF"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_file.name, pagesize=letter,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)
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

    subsection_style = ParagraphStyle(
        'SubsectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    equation_style = ParagraphStyle(
        'EquationStyle',
        parent=styles['Code'],
        fontSize=10,
        textColor=colors.black,
        leftIndent=30,
        rightIndent=30,
        spaceAfter=10,
        spaceBefore=5,
        backColor=colors.HexColor('#f7fafc'),
        borderWidth=1,
        borderColor=colors.black,
        borderPadding=8,
        fontName='Courier'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=8,
        leading=14,
        alignment=4  # Justify
    )

    # Add source information
    story.append(Paragraph("RESEARCH SYNTHESIS", title_style))
    story.append(Spacer(1, 0.2 * inch))

    source_text = f"<i>Synthesized from {len(pdf_names)} research papers</i>"
    story.append(Paragraph(source_text, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # Parse synthesis
    lines = synthesis.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1 * inch))
            continue

        # Remove ** markers
        clean_line = line.replace('**', '')

        # Check for TITLE:
        if clean_line.startswith('TITLE:'):
            title_text = clean_line.replace('TITLE:', '').strip()
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 0.3 * inch))

        # Check for numbered sections or main headers
        elif (re.match(r'\d+\.\s+[A-Z]', clean_line) or
              clean_line in ['ABSTRACT', 'ABSTRACT:', 'INTRODUCTION', 'INTRODUCTION:',
                             'LITERATURE REVIEW', 'RELATED WORK', 'METHODOLOGY', 'METHODOLOGY:',
                             'KEY FINDINGS', 'RESULTS', 'FINDINGS', 'RESULTS/FINDINGS',
                             'DISCUSSION', 'DISCUSSION:', 'CONCLUSION', 'CONCLUSIONS',
                             'CONCLUSIONS:', 'FUTURE WORK', 'KEY EQUATIONS', 'KEY FORMULAS']):
            section_text = clean_line.rstrip(':')
            story.append(Paragraph(section_text, section_style))
            story.append(Spacer(1, 0.15 * inch))

        # Check for subsection headers (starts with capital, contains colon)
        elif re.match(r'^[A-Z][a-zA-Z\s]+:', clean_line):
            subsection_text = clean_line.rstrip(':')
            story.append(Paragraph(subsection_text, subsection_style))

        else:
            story.append(Paragraph(clean_line, body_style))

    # Add equations section
    if all_equations:
        story.append(PageBreak())
        story.append(Paragraph("APPENDIX A: KEY EQUATIONS", section_style))
        story.append(Spacer(1, 0.2 * inch))

        # Show top equations by confidence
        top_equations = sorted(all_equations, key=lambda x: x['confidence'], reverse=True)[:30]

        for idx, eq_data in enumerate(top_equations):
            story.append(
                Paragraph(f"Equation A.{idx + 1} <i>(Page {eq_data['page']}, Confidence: {eq_data['confidence']})</i>",
                          subsection_style))
            story.append(Paragraph(eq_data['equation'], equation_style))
            if eq_data.get('context'):
                context_text = f"<i>Context: {eq_data['context'][:200]}...</i>"
                story.append(Paragraph(context_text, styles['Italic']))
            story.append(Spacer(1, 0.15 * inch))

    # Add tables
    if all_tables:
        story.append(PageBreak())
        story.append(Paragraph(f"APPENDIX B: EXTRACTED TABLES ({len(all_tables)} total)", section_style))
        story.append(Spacer(1, 0.2 * inch))

        for idx, table_data in enumerate(all_tables[:10]):  # Limit to 10 tables
            story.append(Paragraph(f"Table B.{idx + 1} <i>(Page {table_data['page']})</i>",
                                   subsection_style))

            df = table_data['data'].head(15).iloc[:, :6]
            table_list = [df.columns.tolist()] + df.values.tolist()
            table_list = [[str(cell)[:25] for cell in row] for row in table_list]

            t = RLTable(table_list, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
            ]))

            story.append(t)
            story.append(Spacer(1, 0.3 * inch))

    # Add images
    if all_images:
        story.append(PageBreak())
        story.append(Paragraph(f"APPENDIX C: FIGURES ({len(all_images)} total)", section_style))
        story.append(Spacer(1, 0.2 * inch))

        for idx, img_data in enumerate(all_images[:15]):  # Limit to 15 images
            story.append(Paragraph(f"Figure C.{idx + 1} <i>(Page {img_data['page']})</i>",
                                   subsection_style))

            if idx < len(image_summaries):
                story.append(Paragraph(f"<b>Description:</b> {image_summaries[idx]}", body_style))
                story.append(Spacer(1, 0.1 * inch))

            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_data['image'].save(img_temp.name, format="PNG")
            temp_image_files.append(img_temp.name)

            img = img_data['image']
            aspect = img.height / float(img.width)
            max_width = 5 * inch
            img_width = min(img.width, max_width)
            img_height = img_width * aspect

            if img_height > 6 * inch:
                img_height = 6 * inch
                img_width = img_height / aspect

            rl_image = RLImage(img_temp.name, width=img_width, height=img_height)
            story.append(rl_image)
            story.append(Spacer(1, 0.2 * inch))

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
    st.info(f"📚 Processing {len(uploaded_files)} PDF(s)...")

    # Storage for all documents
    all_texts = []
    all_equations = []
    all_tables = []
    all_images = []
    all_image_summaries = []
    pdf_names = []

    # Process each PDF
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 📄 Processing: {uploaded_file.name}")
        pdf_names.append(uploaded_file.name)

        with st.spinner(f"Extracting from PDF {idx + 1}..."):
            # Extract text
            pdf_text = extract_text_from_pdf(uploaded_file)
            all_texts.append(pdf_text)
            st.success(f"✅ Extracted {len(pdf_text):,} characters")

            # Extract equations with advanced detection
            uploaded_file.seek(0)
            equations = extract_equations_advanced(uploaded_file)
            all_equations.extend(equations)
            st.success(f"✅ Found {len(equations)} equations (confidence-scored)")

            # Show top equations from this PDF
            if equations:
                with st.expander(f"🔢 Top Equations from {uploaded_file.name}"):
                    top_eqs = sorted(equations, key=lambda x: x['confidence'], reverse=True)[:10]
                    for eq in top_eqs:
                        st.markdown(f"""
                        <div class="equation-box">
                        <b>Page {eq['page']}</b> | Confidence: {eq['confidence']} | Position: {eq['position']}<br/>
                        <code>{eq['equation']}</code>
                        </div>
                        """, unsafe_allow_html=True)

            # Extract tables
            if include_tables:
                uploaded_file.seek(0)
                tables = extract_tables_from_pdf(uploaded_file)
                all_tables.extend(tables)
                st.success(f"✅ Found {len(tables)} tables")

            # Extract images
            if include_images:
                uploaded_file.seek(0)
                images = extract_images_from_pdf(uploaded_file)

                if images:
                    st.success(f"✅ Found {len(images)} meaningful images")
                    # Analyze images and filter decorative ones
                    meaningful_images = []
                    for img_data in images:
                        img_summary = analyze_image_with_gemini(img_data['image'])
                        # Skip decorative elements
                        if "DECORATIVE_ELEMENT" not in img_summary:
                            all_image_summaries.append(img_summary)
                            all_images.append(img_data)
                            meaningful_images.append(img_data)

                    if len(meaningful_images) < len(images):
                        st.info(f"ℹ️ Filtered out {len(images) - len(meaningful_images)} decorative images")

        st.markdown("---")

    # Generate synthesis
    st.header("🔬 Generating Research Synthesis")
    st.info(f"Creating a condensed {max_papers}-page synthesis from {len(uploaded_files)} papers...")

    with st.spinner("AI is synthesizing all documents... This may take a minute..."):
        synthesis = synthesize_multiple_pdfs(
            all_texts,
            all_equations,
            all_tables,
            all_images,
            max_papers
        )

    # Display synthesis
    st.subheader("📑 Synthesized Research Paper")

    # Show source PDFs
    st.markdown("**Source Documents:**")
    for name in pdf_names:
        st.markdown(f'<span class="pdf-badge">{name}</span>', unsafe_allow_html=True)
    st.markdown("---")

    formatted_synthesis = format_summary_for_display(synthesis)
    st.markdown(formatted_synthesis, unsafe_allow_html=True)

    # Statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PDFs Processed", len(uploaded_files))
    with col2:
        st.metric("Equations Found", len(all_equations))
    with col3:
        st.metric("Tables Extracted", len(all_tables))
    with col4:
        st.metric("Images Analyzed", len(all_images))

    # Create downloadable PDF
    st.subheader("💾 Download Synthesis Report")
    with st.spinner("Creating professional PDF..."):
        pdf_path = create_synthesis_pdf(
            synthesis,
            all_equations,
            all_tables,
            all_images,
            all_image_summaries,
            pdf_names
        )

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Complete Synthesis PDF",
            data=f.read(),
            file_name="research_synthesis.pdf",
            mime="application/pdf",
            help="Professional synthesis with equations, tables, and figures"
        )

    os.unlink(pdf_path)

else:
    st.info("👆 Upload one or more PDF files to generate a research synthesis")

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Multi-PDF Research Synthesis

        **Upload Multiple Papers:**
        - Select 2-10 related research papers
        - System extracts text, equations, tables, and figures from all

        **Advanced Equation Detection:**
        - Confidence scoring based on mathematical symbols
        - Context extraction around equations
        - Filters out non-mathematical text
        - Identifies Greek letters, subscripts, operators

        **Intelligent Synthesis:**
        - Identifies common themes across papers
        - Compares methodologies and findings
        - Creates cohesive narrative
        - Highlights agreements and contradictions
        - Generates unified conclusions

        **Output:**
        - Single condensed research paper (4-15 pages)
        - Structured sections (Abstract, Intro, Methods, Results, Discussion, Conclusion)
        - Appendices with equations, tables, and figures
        - Professional PDF format
        """)

st.markdown("---")
st.markdown("Built with using Streamlit and Google Gemini AI")