# chatbot.py
# Run:
# pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy reportlab
# streamlit run chatbot.py

import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
import numpy as np
from docx import Document
from io import BytesIO
from datetime import datetime
import time
import base64
import os

# Optional dependencies detection for exports
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------------------------
# Page config & Theme CSS
# -------------------------
st.set_page_config(page_title="Data Analytics Helper", layout="wide", initial_sidebar_state="expanded")

# ChatGPT-like colors (green accent + dark theme)
BG = "#0b1020"        # page background
PANEL = "#0f1724"     # panel background
TEXT = "#e6edf3"      # main text
MUTED = "#97a0b3"     # muted text
ACCENT = "#10a37f"    # green accent

st.markdown(f"""
<style>
/* Page backgrounds */
[data-testid="stAppViewContainer"] {{ background-color: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background-color: {PANEL}; color: {TEXT}; }}

/* General text */
h1, h2, h3, h4, h5, p, label, div {{ color: {TEXT} !important; }}
a {{ color: {ACCENT} !important; }}

/* Upload box */
.upload-box {{ background-color: #0f1620; border-radius: 12px; padding: 12px; border: 1px solid #17212a; }}
.file-entry {{ padding: 8px; border-radius: 8px; margin-bottom: 8px; background-color: #09101a; border: 1px solid #16202a; }}

/* Chat area */
.chat-window {{ background-color: #071021; border-radius: 12px; padding: 16px; height: 520px; overflow-y: auto; }}
.msg-bubble {{ padding: 10px 14px; border-radius: 12px; display: inline-block; max-width:80%; margin-bottom:10px; }}
.msg-user {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.04); color: {TEXT}; float: right; clear: both; }}
.msg-bot {{ background: linear-gradient(90deg, rgba(16,163,127,0.12), rgba(16,163,127,0.06)); border: 1px solid rgba(16,163,127,0.18); color: {TEXT}; float: left; clear: both; }}
.clearfix::after {{ content: ""; clear: both; display: table; }}

/* Input area */
.input-area {{ background-color: #0c1118; padding: 10px; border-radius: 12px; border: 1px solid #202631; }}

/* Buttons */
.stButton>button {{
    background-color: {ACCENT};
    color: #022016;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 12px;
    border: none;
}}
.stButton>button:hover {{ filter: brightness(0.95); }}

/* Keyboard hint */
.kb-hint {{ color: {MUTED}; font-size:12px; }}

/* Dataframe container tweak */
.css-1lcbmhc .stFileUploader {{ width: 100%; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Init
# -------------------------
if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = ""

if "uploads" not in st.session_state:
    st.session_state["uploads"] = []  # each: dict{name, content(bytes), uploaded_at, type}

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # chat messages [{"role":"user/assistant","content": "...", "ts": "..."}]

# -------------------------
# Utility functions
# -------------------------
def set_gemini_api_key(api_key: str):
    st.session_state["gemini_api_key"] = api_key
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

def add_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})

def analyze_dataset_statistics(df):
    """Perform comprehensive statistical analysis on dataset"""
    analysis = {
        'basic_stats': {},
        'distributions': {},
        'correlations': None,
        'categorical': {},
    }

    analysis['basic_stats'] = {
        'shape': df.shape,
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        analysis['distributions'] = df[numerical_cols].describe().to_dict()
        if len(numerical_cols) > 1:
            analysis['correlations'] = df[numerical_cols].corr()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    analysis['categorical'] = {col: df[col].value_counts().head(10).to_dict() for col in categorical_cols[:10]}

    return analysis, numerical_cols, categorical_cols

def extract_text_from_file(uploaded_file):
    """Extract text or dataset from uploaded file. Returns tuple depending on content:
       (file_content:str, file_type:str, dataframe or None, stats_analysis or None)
    """
    try:
        # text file
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            return content, "txt", None, None

        # pdf
        if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
            text_pages = []
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text()
                    if txt:
                        text_pages.append(f"Page {i+1}:\n{txt}")
            return "\n\n".join(text_pages), "pdf", None, None

        # docx
        if uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"] or uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs), "docx", None, None

        # csv
        if uploaded_file.name.endswith('.csv'):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            stats_analysis, num_cols, cat_cols = analyze_dataset_statistics(df)
            content = build_dataset_overview_string(df, stats_analysis, num_cols, cat_cols)
            return content, "csv", df, stats_analysis

        # excel
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            stats_analysis, num_cols, cat_cols = analyze_dataset_statistics(df)
            content = build_dataset_overview_string(df, stats_analysis, num_cols, cat_cols)
            return content, "xlsx", df, stats_analysis

        return "", "", None, None

    except Exception as e:
        st.error(f"Error extracting file: {e}")
        return None, None, None, None

def build_dataset_overview_string(df, stats_analysis, num_cols, cat_cols):
    """Helper to create a compact textual overview for datasets"""
    overview = []
    overview.append(f"Dataset Overview:\n- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    overview.append(f"- Memory Usage: {stats_analysis['basic_stats']['memory_usage']:.2f} MB")
    overview.append(f"- Numerical Features ({len(num_cols)}): {', '.join(num_cols[:10]) if num_cols else 'None'}")
    overview.append(f"- Categorical Features ({len(cat_cols)}): {', '.join(cat_cols[:10]) if cat_cols else 'None'}")
    overview.append("\nMissing Values (percent):")
    for k, v in stats_analysis['basic_stats']['missing_pct'].items():
        overview.append(f"- {k}: {v:.2f}%")
    if num_cols:
        overview.append("\nStatistical Summary (numerical features):")
        overview.append(str(pd.DataFrame(stats_analysis['distributions']).transpose().head(10)))
    if stats_analysis.get('correlations') is not None:
        overview.append("\nCorrelation matrix (top features):")
        overview.append(str(stats_analysis['correlations'].round(3).head(10)))
    overview.append("\nSample Data (first 5 rows):")
    overview.append(df.head(5).to_string())
    return "\n\n".join(overview)

# Exports
def export_chat_as_docx(chat_history):
    if not DOCX_AVAILABLE:
        st.warning("python-docx not installed; DOCX export not available.")
        return None
    doc = DocxDocument()
    doc.add_heading("Chat Export - Data Analytics Helper", level=1)
    for msg in chat_history:
        when = msg['ts']
        sender = msg['role'].capitalize()
        p = doc.add_paragraph()
        p.add_run(f"{sender} ({when}): ").bold = True
        p.add_run(msg['content'])
    out = BytesIO()
    doc.save(out)
    out.seek(0)
    return out

def export_chat_as_pdf(chat_history):
    if not PDF_AVAILABLE:
        st.warning("reportlab not installed; PDF export not available.")
        return None
    buffer = BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "Chat Export - Data Analytics Helper")
    y -= 28
    c.setFont("Helvetica", 10)
    for msg in chat_history:
        if y < 72:
            c.showPage()
            y = height - 72
        sender = msg['role'].capitalize()
        timestamp = msg['ts']
        text = f"{sender} ({timestamp}): {msg['content']}"
        # naive wrap
        while len(text) > 100:
            c.drawString(72, y, text[:100])
            text = text[100:]
            y -= 14
        c.drawString(72, y, text)
        y -= 18
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# API key block
# -------------------------
if not st.session_state["gemini_api_key"]:
    st.sidebar.success("💰 100% FREE - Powered by Google Gemini")
    st.sidebar.info("📊 1,500 requests/day available")
    st.header("Set up your Gemini API Key")
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input("Enter your Gemini API Key:", type="password", key="api_input")
    with col2:
        if st.button("Set API Key"):
            if api_key:
                set_gemini_api_key(api_key)
                st.success("✅ API Key set successfully!")
                st.experimental_rerun()
            else:
                st.error("Please enter a valid API key.")
    with st.expander("📖 How to get your FREE Gemini API Key"):
        st.markdown("""
        1. Go to Google AI Studio (https://aistudio.google.com/app/apikey)
        2. Sign in with Google
        3. Create API Key and paste it above
        """)
    st.stop()
else:
    # Configure library
    try:
        genai.configure(api_key=st.session_state["gemini_api_key"])
    except Exception:
        pass

# -------------------------
# Sidebar (Navigation + Recent uploads)
# -------------------------
with st.sidebar:
    st.title("🎯 Reference & Resource Hub")
    nav = st.radio("Navigate to", ["Chat", "Uploads", "Reference", "Settings"], index=0)
    st.divider()
    st.header("Recent uploads")
    if not st.session_state["uploads"]:
        st.markdown("<div class='file-entry'>No recent uploads</div>", unsafe_allow_html=True)
    else:
        for f in reversed(st.session_state["uploads"][-10:]):
            uploaded_at = f.get("uploaded_at", "unknown")
            size_kb = len(f["content"]) / 1024
            st.markdown(
                f"<div class='file-entry'><strong>{f['name']}</strong><br><span style='color:{MUTED}; font-size:12px;'>{size_kb:.1f} KB — {uploaded_at}</span></div>",
                unsafe_allow_html=True
            )
    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Clear recent uploads"):
        st.session_state["uploads"] = []
        st.experimental_rerun()
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.experimental_rerun()
    st.divider()
    st.markdown("### Shortcuts")
    st.markdown("<div class='kb-hint'>Press Enter to send (Shift+Enter for newline). Drag & drop files to upload.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Reference")
    st.markdown("- Linear/Logistic Regression\n- Decision Trees & Random Forests\n- SVM, KNN, Naive Bayes\n- Neural Networks & Deep Learning\n- Clustering (K-means, DBSCAN)\n- PCA, t-SNE")
    st.divider()
    st.markdown("### Resources")
    st.markdown("- Scikit-learn\n- Papers with Code\n- Kaggle Learn")

# -------------------------
# Main UI
# -------------------------
st.title("Data Analytics Chatbot 🤖")
st.caption("Upload lecture notes, datasets, or statistical analysis documents, then ask targeted questions.")

# NAV: Uploads page
if nav == "Uploads":
    st.header("📁 Upload Materials")
    st.markdown("Drop a file to upload. Supported: txt, pdf, docx, csv, xlsx")
    with st.form("upload_form", clear_on_submit=False):
        files = st.file_uploader("Choose files to analyze", accept_multiple_files=True, type=['txt','pdf','docx','csv','xlsx'])
        auto_analyze = st.checkbox("Auto-analyze after upload", value=False)
        submitted = st.form_submit_button("Upload")
        if submitted and files:
            prog = st.progress(0)
            total = len(files)
            for i, uf in enumerate(files):
                prog.progress(int((i+1)/total*100))
                # read bytes
                content = uf.read()
                # store in session
                st.session_state["uploads"].append({
                    "name": uf.name,
                    "content": content,
                    "uploaded_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "type": uf.type or uf.name.split('.')[-1]
                })
                # if auto, run extract + one-line summary into chat
                if auto_analyze:
                    st.spinner(f"Extracting {uf.name} ...")
                    # For the stub we call extract_text_from_file on an in-memory BytesIO
                    from io import BytesIO
                    buf = BytesIO(content)
                    buf.name = uf.name
                    file_content, file_type, df, stats = extract_text_from_file(buf)
                    if file_content:
                        add_message("assistant", f"Uploaded {uf.name}: {file_type.upper()} detected. Brief overview available.")
            prog.progress(100)
            st.success("Upload complete")
    st.divider()
    st.header("Contextual suggestions")
    if st.session_state["uploads"]:
        last = st.session_state["uploads"][-1]
        st.markdown(f"**Latest file:** {last['name']}")
        st.markdown("- Try: `Summarize the dataset`")
        st.markdown("- Try: `Give me 5 practice problems based on this file`")
        st.markdown("- Try: `Suggest cleaning steps and feature engineering ideas`")
        if st.button("Analyze latest file (quick)"):
            with st.spinner("Running quick extract..."):
                from io import BytesIO
                buf = BytesIO(last["content"])
                buf.name = last["name"]
                file_content, file_type, df, stats = extract_text_from_file(buf)
                if file_content:
                    add_message("assistant", f"Quick analysis ready for {last['name']}. Try asking: 'Summarize the dataset' or 'Which features are most important?'.")
                    st.success("Quick analysis added to chat")
    else:
        st.info("Upload a file to get contextual suggestions.")

# NAV: Reference
elif nav == "Reference":
    st.header("📖 Quick Reference")
    st.markdown("""
    **Core ML Algorithms**
    - Linear/Logistic Regression
    - Decision Trees & Random Forests
    - SVM, KNN, Naive Bayes
    - Neural Networks & Deep Learning
    - Clustering (K-means, DBSCAN)
    - Dimensionality Reduction (PCA, t-SNE)
    """)
    st.divider()
    st.markdown("**Statistical Methods**\n- Hypothesis Testing\n- Regression Analysis\n- Time Series Analysis\n- Experimental Design")
    st.divider()
    st.markdown("**Tips**\nUse specific prompts. Upload files for contextualized answers.")

# NAV: Settings
elif nav == "Settings":
    st.header("⚙️ Settings & Preferences")
    verbose = st.checkbox("Enable verbose responses", value=False, key="verbose")
    autosummary = st.checkbox("Auto-summaries on upload", value=False, key="autosummary")
    st.markdown("Theme: ChatGPT-like (dark + green accent).")
    if st.button("Reset all except API key"):
        gemkey = st.session_state.get("gemini_api_key", "")
        st.session_state.clear()
        st.session_state["gemini_api_key"] = gemkey
        st.experimental_rerun()

# NAV: Chat (default)
else:
    st.header("💬 Ask Questions")
    st.markdown("Ask about uploaded materials or general ML/Stats concepts. Be specific for best results.")

    # Chat display + actions on right
    left_col, right_col = st.columns([0.72, 0.28])

    with left_col:
        # Chat window
        st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            role = msg["role"]
            text = msg["content"]
            ts = msg["ts"]
            # Render bubble
            if role == "user":
                st.markdown(f"<div class='clearfix'><div class='msg-bubble msg-user'>{st.markdown(text, unsafe_allow_html=True) or ''}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='clearfix'><div class='msg-bubble msg-bot'>{st.markdown(text, unsafe_allow_html=True) or ''}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Input: using st.chat_input for native feel. Provide fallback to text_area if necessary.
        prompt = st.chat_input("Ask about algorithms, statistics, math, or implementation...")
        if prompt:
            # Save user message
            add_message("user", prompt)

            # Display user message immediately (so UI shows instantly)
            st.experimental_rerun()  # rerun to render user message and then assistant will generate below

    with right_col:
        st.markdown("### Actions")
        if st.button("Export chat as DOCX"):
            out = export_chat_as_docx(st.session_state["messages"])
            if out:
                st.download_button("Download DOCX", out, file_name="chat_export.docx")
        if st.button("Export chat as PDF"):
            out = export_chat_as_pdf(st.session_state["messages"])
            if out:
                st.download_button("Download PDF", out, file_name="chat_export.pdf")
        st.divider()
        st.markdown("### Quick file view")
        if st.session_state["uploads"]:
            for f in reversed(st.session_state["uploads"][-5:]):
                st.markdown(f"- **{f['name']}** — {f['uploaded_at']}")
        else:
            st.write("No files uploaded yet.")
        st.divider()
        st.markdown("### Suggested prompts")
        if st.session_state["uploads"]:
            last = st.session_state["uploads"][-1]
            if st.button(f"Summarize {last['name']}"):
                add_message("user", f"Please summarize the file {last['name']} and suggest next steps.")
                st.experimental_rerun()
            if st.button(f"Suggest visualizations for {last['name']}"):
                add_message("user", f"Give 3 visualization ideas for {last['name']}.")
                st.experimental_rerun()
        else:
            if st.button("Try: Explain linear regression like I'm 5"):
                add_message("user", "Explain linear regression like I'm 5 years old")
                st.experimental_rerun()

    # If new user messages exist with no assistant reply, generate assistant reply.
    # We'll check last two messages: if last role is 'user', generate an assistant response.
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
        user_prompt = st.session_state["messages"][-1]["content"]

        # Show assistant typing indicator and produce a reply (using Gemini)
        add_message("assistant", "⏳ Generating response...")  # placeholder to show typing
        st.experimental_rerun()

    # If placeholder is present, replace with model response (separate rerun)
    # Find the first assistant placeholder that equals the exact typing placeholder
    for i, m in enumerate(st.session_state["messages"]):
        if m["role"] == "assistant" and m["content"] == "⏳ Generating response...":
            # Build context and call Gemini
            try:
                with st.spinner("🤖 Assistant is typing..."):
                    # Compose system message and context
                    system_context = """You are a helpful, expert Machine Learning and Statistics assistant. 
Provide concise explanations, include formulas in LaTeX when useful, and practical implementation tips."""
                    # Add uploaded file context if available (trimmed)
                    file_context = ""
                    if st.session_state["uploads"]:
                        last = st.session_state["uploads"][-1]
                        snippet = (last["content"][:3000].decode("utf-8", errors="ignore")
                                   if isinstance(last["content"], (bytes, bytearray)) else str(last["content"])[:3000])
                        file_context = f"\n\nContext from uploaded file ({last['name']}):\n{snippet}"

                    # Build prompt from conversation (last n messages)
                    conv_text = system_context + file_context + "\n\nConversation:\n"
                    for msg in st.session_state["messages"][-8:]:
                        conv_text += f"{msg['role']}: {msg['content']}\n"

                    # Call Gemini
                    model = genai.GenerativeModel(
                        'gemini-2.5-flash',
                        generation_config={
                            'temperature': 0.2 if st.session_state.get("verbose", False) is False else 0.7,
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 2048,
                        }
                    )
                    # It's possible generate_content raises -- we handle exceptions
                    response = model.generate_content(conv_text)
                    assistant_text = response.text if response and hasattr(response, "text") else "Sorry, I could not produce an answer."

                    # Replace the placeholder assistant message with the real content
                    st.session_state["messages"][i]["content"] = assistant_text
                    st.experimental_rerun()

            except Exception as e:
                st.session_state["messages"][i]["content"] = f"⚠️ Error generating response: {e}"
                st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("<div style='color:#98a0b6; font-size:12px;'>Tip: Upload files and ask specific questions. This demo contains stubbed and limited analysis — replace or extend with your own model/processing where needed.</div>", unsafe_allow_html=True)

