# run 'pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy' in terminal to install dependencies
# run streamlit run chatbot.py in terminal to start the app

# chatbot.py
# Run:
# pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy reportlab

import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
import numpy as np
from docx import Document
from io import BytesIO
from datetime import datetime
import os

# Optional dependencies detection
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------------------------
# Safe rerun helper
# -------------------------
def safe_rerun():
    """Safe rerun for Streamlit."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
    except Exception:
        pass

# -------------------------
# Page config & Theme CSS
# -------------------------
st.set_page_config(page_title="Data Analytics Helper", layout="wide", initial_sidebar_state="expanded")

BG = "#0b1020"
PANEL = "#0f1724"
TEXT = "#e6edf3"
MUTED = "#97a0b3"
ACCENT = "#10a37f"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background-color: {PANEL}; color: {TEXT}; }}
h1,h2,h3,h4,h5,p,label,div {{ color: {TEXT} !important; }}
a {{ color: {ACCENT} !important; }}
.upload-box {{ background-color: #0f1620; border-radius: 12px; padding: 12px; border: 1px solid #17212a; }}
.file-entry {{ padding: 8px; border-radius: 8px; margin-bottom: 8px; background-color: #09101a; border: 1px solid #16202a; }}
.chat-window {{ background-color: #071021; border-radius: 12px; padding: 16px; height: 520px; overflow-y: auto; }}
.msg-bubble {{ padding: 10px 14px; border-radius: 12px; display: inline-block; max-width:80%; margin-bottom:10px; }}
.msg-user {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.04); color: {TEXT}; float: right; clear: both; }}
.msg-bot {{ background: linear-gradient(90deg, rgba(16,163,127,0.12), rgba(16,163,127,0.06)); border: 1px solid rgba(16,163,127,0.18); color: {TEXT}; float: left; clear: both; }}
.clearfix::after {{ content: ""; clear: both; display: table; }}
.input-area {{ background-color: #0c1118; padding: 10px; border-radius: 12px; border: 1px solid #202631; }}
.stButton>button {{ background-color: {ACCENT}; color: #022016; font-weight: 600; border-radius: 8px; padding: 8px 12px; border: none; }}
.stButton>button:hover {{ filter: brightness(0.95); }}
.kb-hint {{ color: {MUTED}; font-size:12px; }}
body {{ background-color: {BG}; color: {TEXT}; }}
.stSidebar {{ background-color: {PANEL}; color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Init
# -------------------------
if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = ""
if "uploads" not in st.session_state:
    st.session_state["uploads"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# New state for chat session and model
if "chat" not in st.session_state:
    st.session_state["chat"] = None
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gemini-2.5-flash"
if "model_temperature" not in st.session_state:
    st.session_state["model_temperature"] = 0.5


# -------------------------
# Utility functions
# -------------------------
def set_gemini_api_key(api_key: str):
    st.session_state["gemini_api_key"] = api_key
    try:
        genai.configure(api_key=api_key)
        # Reset chat session when key changes
        st.session_state["chat"] = None 
    except Exception:
        pass

def add_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})

def extract_text_from_file(uploaded_file):
    # This is a bit tricky with Streamlit's file_uploader objects losing state on rerun.
    # The 'uploads' state stores the raw content, so we need to process it from there.
    try:
        file_content = uploaded_file["content"]
        file_name = uploaded_file["name"]
        file_type = uploaded_file["type"]

        # Use BytesIO to simulate an uploaded file object for libraries
        file_stream = BytesIO(file_content)

        if file_type == "text/plain" or file_name.endswith(".txt"):
            content = file_stream.read().decode("utf-8", errors="ignore")
            return content, "txt", None, None
        
        if file_type == "application/pdf" or file_name.endswith(".pdf"):
            text_pages = []
            # pdfplumber expects a file-like object, which BytesIO is
            with pdfplumber.open(file_stream) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text()
                    if txt:
                        text_pages.append(f"Page {i+1}:\n{txt}")
            return "\n\n".join(text_pages), "pdf", None, None
        
        if file_name.endswith(".docx"):
            doc = Document(file_stream)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs), "docx", None, None
        
        # Data files: CSV/XLSX
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_stream)
            stats, num_cols, cat_cols = analyze_dataset_statistics(df)
            content = build_dataset_overview_string(df, stats, num_cols, cat_cols)
            return content, "csv", df, stats
        
        if file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_stream)
            stats, num_cols, cat_cols = analyze_dataset_statistics(df)
            content = build_dataset_overview_string(df, stats, num_cols, cat_cols)
            return content, "xlsx", df, stats
        
        return "", "", None, None
    except Exception as e:
        # st.error(f"Error extracting file: {e}") # Don't show in utility func
        return "", "", None, None


def analyze_dataset_statistics(df):
    analysis = {
        'basic_stats': {},
        'distributions': {},
        'correlations': None,
        'categorical': {},
    }
    analysis['basic_stats'] = {
        'shape': df.shape,
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.apply(str).to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        analysis['distributions'] = df[numerical_cols].describe().to_dict()
        if len(numerical_cols) > 1:
            analysis['correlations'] = df[numerical_cols].corr()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    analysis['categorical'] = {col: df[col].value_counts().head(10).to_dict() for col in categorical_cols[:10]}
    return analysis, numerical_cols, categorical_cols

def build_dataset_overview_string(df, stats_analysis, num_cols, cat_cols):
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

def export_chat_as_docx(chat_history):
    doc = Document()
    doc.add_heading("Chat Export - Data Analytics Helper", level=1)
    for msg in chat_history:
        p = doc.add_paragraph()
        # Clean timestamp for display
        ts_clean = datetime.fromisoformat(msg['ts']).strftime("%Y-%m-%d %H:%M:%S")
        p.add_run(f"{msg['role'].capitalize()} ({ts_clean}): ").bold = True
        p.add_run(msg['content'])
    out = BytesIO()
    doc.save(out)
    out.seek(0)
    return out

def export_chat_as_pdf(chat_history):
    buffer = BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "Chat Export - Data Analytics Helper")
    y -= 28
    c.setFont("Helvetica", 10)
    for msg in chat_history:
        # Clean timestamp for display
        ts_clean = datetime.fromisoformat(msg['ts']).strftime("%Y-%m-%d %H:%M:%S")
        text_lines = f"{msg['role'].capitalize()} ({ts_clean}): {msg['content']}".split('\n')
        for line in text_lines:
            if y < 72:
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 14
    c.save()
    buffer.seek(0)
    return buffer

# --- GEMINI CHAT LOGIC ---
def create_chat_session():
    """Initializes and returns the Gemini Chat object with context."""
    if not st.session_state["gemini_api_key"]:
        st.error("API Key is not set. Cannot initialize model.")
        return None

    # 1. Build the System Instruction (The core prompt)
    system_instruction = (
        "You are a highly capable Data Analytics Helper. Your primary role is to assist the user "
        "in understanding and analyzing the data and documents provided. "
        "You have access to the full text and/or statistical overview of all uploaded files. "
        "You must refer to the provided context when answering questions related to the uploaded materials. "
        "For dataset analysis (CSV, XLSX), focus on interpreting the statistical summaries, correlations, "
        "missing values, and sample data. Do not generate code unless specifically asked to. "
        "If the user asks a general question, answer it concisely. If the information is not in the uploaded context, "
        "state that clearly."
    )

    # 2. Add the context from uploaded files to the chat history
    # The Chat API manages history, so we'll push the context as the first message
    context_parts = []
    
    if st.session_state["uploads"]:
        # Iterate over uploads and prepare text/data overview for the model
        for f in st.session_state["uploads"]:
            # Re-read the file content from session state (BytesIO can only be read once)
            file_data_bytes = f["content"]
            file_stream = BytesIO(file_data_bytes)
            
            # Use the existing extract logic to get the text representation
            # We pass a dictionary that simulates the file_uploader object structure
            text_context, ftype, df_data, stats = extract_text_from_file(f)

            if text_context:
                context_parts.append(
                    f"\n--- FILE CONTEXT: {f['name']} ({ftype.upper()}) ---\n"
                    f"{text_context}\n"
                    "---------------------------------------------------\n"
                )
    
    # Prepend the context to the first chat turn
    initial_context_message = "I have loaded the following documents and data for your analysis:\n\n" + "\n".join(context_parts)
    
    if not initial_context_message.strip():
        # Fallback if no files were uploaded or extracted
        initial_context_message = "No files have been loaded yet. I am ready for general data analytics questions."

    # 3. Create the model instance and chat session
    model = genai.GenerativeModel(
        model_name=st.session_state["model_name"],
        system_instruction=system_instruction,
        config=genai.types.GenerateContentConfig(
            temperature=st.session_state["model_temperature"]
        )
    )
    
    # Start the chat session
    chat = model.start_chat()
    
    # Send the initial context to the model so it 'knows' about the files
    # This is a one-time setup for the current context (uploads)
    chat.send_message(initial_context_message)
    
    return chat

def get_gemini_response(user_prompt: str):
    """Sends user prompt to the Gemini Chat session and returns the response."""
    
    # Check if chat session exists, if not, create it
    if st.session_state["chat"] is None:
        st.session_state["chat"] = create_chat_session()
    
    if st.session_state["chat"] is None:
        return "Error: Could not start the chat session. Please check your API key."

    try:
        # Send the user's message
        response = st.session_state["chat"].send_message(user_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with Gemini: {e}"

# -------------------------
# API Key Input
# -------------------------
if not st.session_state["gemini_api_key"]:
    st.sidebar.success("💰 100% FREE - Powered by Google Gemini")
    st.sidebar.info("📊 1,500 requests/day available")
    st.header("Set up your Gemini API Key")
    api_key_col1, api_key_col2 = st.columns([4,1])
    with api_key_col1:
        api_key = st.text_input("Enter your Gemini API Key:", type="password", key="api_input")
    with api_key_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Set API Key"):
            if api_key:
                set_gemini_api_key(api_key)
                st.success("API Key set successfully!")
                safe_rerun()
            else:
                st.error("Please enter a valid API key.")
    with st.expander("How to get your FREE Gemini API Key"):
        st.markdown("""
            1. Go to Google AI Studio (https://aistudio.google.com/app/apikey)
            2. Sign in with Google
            3. Create API Key and paste it above
            """)
    st.stop()
else:
    try:
        genai.configure(api_key=st.session_state["gemini_api_key"])
    except Exception:
        pass


# -------------------------
# Sidebar + Navigation
# -------------------------
with st.sidebar:
    st.title("Navigation Hub")
    nav = st.radio("Navigate to", ["Chat", "Uploads", "Reference", "Settings"], index=0)
    st.divider()
    st.header("Recent uploads")
    if not st.session_state["uploads"]:
        st.markdown("<div class='file-entry'>No recent uploads</div>", unsafe_allow_html=True)
    else:
        for f in reversed(st.session_state["uploads"][-10:]):
            uploaded_at = datetime.fromisoformat(f.get("uploaded_at", datetime.utcnow().isoformat())).strftime("%Y-%m-%d %H:%M:%S")
            size_kb = len(f["content"]) / 1024
            st.markdown(f"<div class='file-entry'><strong>{f['name']}</strong><br><span style='color:{MUTED}; font-size:12px;'>{size_kb:.1f} KB — {uploaded_at}</span></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Clear recent uploads"):
        st.session_state["uploads"] = []
        st.session_state["chat"] = None # Reset chat session
        safe_rerun()
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.session_state["chat"] = None # Reset chat session
        safe_rerun()
    st.divider()
    st.markdown("<div class='kb-hint'>Press Enter to send (Shift+Enter for newline). Drag & drop files to upload.</div>", unsafe_allow_html=True)

# -------------------------
# Main Pages
# -------------------------
st.title("Data Analytics Chatbot 🤖")
st.caption("Upload lecture notes, datasets, or statistical analysis documents, then ask targeted questions.")

# NAV Pages
if nav == "Uploads":
    st.header("Upload Materials")
    st.markdown("Drop a file to upload. Supported: txt, pdf, docx, csv, xlsx")
    
    # A Streamlit file_uploader object must be processed *before* a rerun, 
    # as the object itself is temporary. We'll store its content in session state.
    uploaded_files_objects = st.file_uploader("Choose files to analyze", accept_multiple_files=True, type=['txt','pdf','docx','csv','xlsx'])

    # Use a different form approach that processes the *temporary* file objects immediately
    if uploaded_files_objects:
        with st.form("upload_process_form", clear_on_submit=True):
            auto_analyze = st.checkbox("Auto-analyze after upload", value=False)
            submitted = st.form_submit_button("Process Uploads")
            
            if submitted:
                prog = st.progress(0)
                total = len(uploaded_files_objects)
                
                # IMPORTANT: Clear existing chat session to force new context
                st.session_state["chat"] = None
                
                for i, uf in enumerate(uploaded_files_objects):
                    prog.progress(int((i+1)/total*100))
                    
                    # Read the content once and store it
                    content = uf.read()
                    file_dict = {
                        "name": uf.name,
                        "content": content,
                        "uploaded_at": datetime.utcnow().isoformat(), # Use isoformat for consistency
                        "type": uf.type
                    }
                    st.session_state["uploads"].append(file_dict)
                    
                    if auto_analyze:
                        # For auto-analyze, we must *temporarily* recreate the file object 
                        # or pass the file_dict structure for extraction. We'll use the dict.
                        text, ftype, df, stats = extract_text_from_file(file_dict) 
                        if text:
                             st.text_area(f"Preview: {uf.name} ({ftype.upper()})", text, height=200)

                prog.progress(100)
                st.success(f"{total} files uploaded and processed successfully! Chat context reset.")
                # We do not safe_rerun() here to allow the success message to be seen, 
                # but the sidebar will update on the next interaction.

elif nav == "Reference":
    st.header("Reference Materials")
    st.markdown("This page can display uploaded documents, sample datasets, or links to tutorials.")
    st.info("Currently stubbed. You can add reference display logic here.")

elif nav == "Settings":
    st.header("Settings")
    st.markdown("Adjust chatbot parameters or reset data.")
    
    # Model Selection
    st.subheader("Model Configuration")
    st.session_state["model_name"] = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=["gemini-2.5-flash", "gemini-2.5-pro"].index(st.session_state["model_name"]),
        help="Flash is faster and cheaper, Pro is more capable for complex reasoning."
    )
    st.session_state["model_temperature"] = st.slider(
        "Creativity (Temperature)",
        min_value=0.0, max_value=1.0, value=st.session_state["model_temperature"], step=0.05,
        help="Lower values (closer to 0) for factual, deterministic responses. Higher values for more creative, diverse responses."
    )
    
    if st.button("Apply Model Settings"):
        st.session_state["chat"] = None # Force chat reset with new settings
        st.success("Settings applied! The chat session has been reset.")
        safe_rerun()

    st.subheader("Data Reset")
    if st.button("Reset all session data"):
        st.session_state.clear()
        safe_rerun()


else:
    # CHAT PAGE
    st.header("Chat")
    
    # Check if a chat session is initialized
    if st.session_state["chat"] is None and st.session_state["gemini_api_key"]:
        with st.spinner("Initializing Gemini Chat with uploaded context..."):
            st.session_state["chat"] = create_chat_session()
            
            # The initial context message is added to the model's history, but not the UI history.
            # To give the user a starting point, we can add a custom welcome message.
            if not st.session_state["messages"]:
                 initial_msg = "Hello! I'm your Data Analytics Helper. I've processed your uploaded files (if any). How can I help you analyze your data or notes today? You can ask me to summarize a document, explain a statistic, or find correlations."
                 add_message("bot", initial_msg)
                 safe_rerun()


    # Display chat messages
    chat_box = st.container() 
    with chat_box:
        for msg in st.session_state["messages"]:
            role = msg['role']
            content = msg['content']
            if role == "user":
                st.markdown(f"<div class='msg-bubble msg-user clearfix'>{content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-bubble msg-bot clearfix'>{content}</div>", unsafe_allow_html=True)
        # Scroll to bottom logic is hard in Streamlit, but this is the right place to display.
    
    # Input box and Send button
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.container():
        user_input = st.text_area("Your message:", "", key="chat_input", height=80)
        
        # Streamlit forms or key logic is better for handling ENTER submission in text_area, 
        # but for simplicity, we'll keep the button.
        if st.button("Send", key="send_button"):
            user_prompt = user_input.strip()
            if user_prompt:
                # 1. Add user message to state
                add_message("user", user_prompt)
                
                # 2. Get bot response
                with st.spinner("Gemini is analyzing..."):
                    bot_response = get_gemini_response(user_prompt)
                
                # 3. Add bot message to state
                add_message("bot", bot_response)
                
                # 4. Clear input and rerun
                st.session_state["chat_input"] = "" # Clear the input box
                safe_rerun()


    # Export
    st.divider()
    st.markdown("### Export Chat")
    col1, col2 = st.columns(2)
    with col1:
        docx_file = export_chat_as_docx(st.session_state["messages"])
        st.download_button("Download DOCX", docx_file, file_name="chat_export.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with col2:
        if PDF_AVAILABLE:
            pdf_file = export_chat_as_pdf(st.session_state["messages"])
            st.download_button("Download PDF", pdf_file, file_name="chat_export.pdf", mime="application/pdf")
        else:
            st.info("PDF export unavailable (reportlab missing)")

st.markdown("---")
st.markdown("<div style='color:#98a0b6; font-size:12px;'>Tip: Upload files and ask specific questions. This demo is powered by Google Gemini.</div>", unsafe_allow_html=True)
