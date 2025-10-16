# chatbot.py
# to run locally:
# pip install streamlit google-generativeai pdfplumber pandas python-docx numpy reportlab time base64 os
# streamlit run chatbot.py

import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
import numpy as np
from docx import Document
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import time
import base64
import os

# Optional dependencies detection
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
.upload-box {{ background-color: #0f1620; border-radius:12px; padding:12px; border:1px solid #17212a; }}
.file-entry {{ padding:8px; border-radius:8px; margin-bottom:8px; background-color:#09101a; border:1px solid #16202a; }}
.chat-window {{ background-color:#071021; border-radius:12px; padding:16px; height:520px; overflow-y:auto; }}
.msg-bubble {{ padding:10px 14px; border-radius:12px; display:inline-block; max-width:80%; margin-bottom:10px; }}
.msg-user {{ background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.04); color:{TEXT}; float:right; clear:both; }}
.msg-bot {{ background: linear-gradient(90deg, rgba(16,163,127,0.12), rgba(16,163,127,0.06)); border:1px solid rgba(16,163,127,0.18); color:{TEXT}; float:left; clear:both; }}
.clearfix::after {{ content:""; clear:both; display:table; }}
.input-area {{ background-color:#0c1118; padding:10px; border-radius:12px; border:1px solid #202631; }}
.stButton>button {{ background-color:{ACCENT}; color:#022016; font-weight:600; border-radius:8px; padding:8px 12px; border:none; }}
.stButton>button:hover {{ filter: brightness(0.95); }}
.kb-hint {{ color:{MUTED}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Init
# -------------------------
if "gemini_api_key" not in st.session_state: st.session_state["gemini_api_key"] = ""
if "uploads" not in st.session_state: st.session_state["uploads"] = []
if "messages" not in st.session_state: st.session_state["messages"] = []

# -------------------------
# Utility functions
# -------------------------
def safe_rerun():
    try:
        if hasattr(st, "rerun"): st.rerun()
    except Exception: pass

def set_gemini_api_key(api_key: str):
    st.session_state["gemini_api_key"] = api_key
    try: genai.configure(api_key=api_key)
    except Exception: pass

def add_message(role, content):
    st.session_state["messages"].append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})

def extract_text_from_file(uploaded_file):
    """Extract text/dataset, return (content:str, type:str, df, stats_analysis)"""
    try:
        if uploaded_file is None: return "","",None,None
        fname, ftype = getattr(uploaded_file,"name",""), getattr(uploaded_file,"type","")
        if ftype=="text/plain" or fname.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore"), "txt", None, None
        if ftype=="application/pdf" or fname.endswith(".pdf"):
            pages=[]
            with pdfplumber.open(uploaded_file) as pdf:
                for i,p in enumerate(pdf.pages):
                    txt=p.extract_text()
                    if txt: pages.append(f"Page {i+1}:\n{txt}")
            return "\n\n".join(pages),"pdf",None,None
        if ftype in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document","application/msword"] or fname.endswith(".docx"):
            doc=Document(uploaded_file)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()]),"docx",None,None
        if fname.endswith(".csv"):
            uploaded_file.seek(0); df=pd.read_csv(uploaded_file)
            stats,num_cols,cat_cols=analyze_dataset_statistics(df)
            return build_dataset_overview_string(df, stats, num_cols, cat_cols), "csv", df, stats
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            uploaded_file.seek(0); df=pd.read_excel(uploaded_file)
            stats,num_cols,cat_cols=analyze_dataset_statistics(df)
            return build_dataset_overview_string(df, stats, num_cols, cat_cols), "xlsx", df, stats
        st.warning("Unsupported file type. Use txt,pdf,docx,csv,xlsx."); return "","",None,None
    except Exception as e:
        st.error(f"File extraction error: {e}"); return None,None,None,None

def export_chat_as_docx(chat_history):
    doc=Document()
    doc.add_heading("Chat Export - Data Analytics Helper",level=1)
    for msg in chat_history:
        p=doc.add_paragraph(); p.add_run(f"{msg['role'].capitalize()} ({msg['ts']}): ").bold=True; p.add_run(msg['content'])
    out=BytesIO(); doc.save(out); out.seek(0); return out

def export_chat_as_pdf(chat_history):
    buffer=BytesIO(); c=pdf_canvas.Canvas(buffer,pagesize=letter)
    width,height=letter; y=height-72
    c.setFont("Helvetica-Bold",14); c.drawString(72,y,"Chat Export - Data Analytics Helper"); y-=28
    c.setFont("Helvetica",10)
    for msg in chat_history:
        if y<72: c.showPage(); y=height-72
        text=f"{msg['role'].capitalize()} ({msg['ts']}): {msg['content']}"
        c.drawString(72,y,text[:100]); y-=14
    c.save(); buffer.seek(0); return buffer

def analyze_dataset_statistics(df):
    analysis={'basic_stats':{},'distributions':{},'correlations':None,'categorical':{}}
    analysis['basic_stats']={'shape':df.shape,'missing_pct':(df.isnull().sum()/len(df)*100).to_dict(),'dtypes':df.dtypes.apply(lambda x:str(x)).to_dict(),'memory_usage':df.memory_usage(deep=True).sum()/1024**2}
    numerical_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        analysis['distributions']=df[numerical_cols].describe().to_dict()
        if len(numerical_cols)>1: analysis['correlations']=df[numerical_cols].corr()
    categorical_cols=df.select_dtypes(include=['object','category']).columns.tolist()
    analysis['categorical']={col:df[col].value_counts().head(10).to_dict() for col in categorical_cols[:10]}
    return analysis,numerical_cols,categorical_cols

def build_dataset_overview_string(df, stats_analysis, num_cols, cat_cols):
    try:
        overview=[f"Dataset Overview:\n- Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
        overview.append(f"- Memory Usage: {stats_analysis['basic_stats']['memory_usage']:.2f} MB")
        overview.append(f"- Numerical Features ({len(num_cols)}): {', '.join(num_cols[:10]) if num_cols else 'None'}")
        overview.append(f"- Categorical Features ({len(cat_cols)}): {', '.join(cat_cols[:10]) if cat_cols else 'None'}")
        overview.append("\nMissing Values (percent):")
        for k,v in stats_analysis['basic_stats']['missing_pct'].items(): overview.append(f"- {k}: {v:.2f}%")
        if num_cols: overview.append("\nStatistical Summary (numerical features):\n"+str(pd.DataFrame(stats_analysis['distributions']).transpose().head(10)))
        if stats_analysis.get('correlations') is not None: overview.append("\nCorrelation matrix (top features):\n"+str(stats_analysis['correlations'].round(3).head(10)))
        overview.append("\nSample Data (first 5 rows):\n"+df.head(5).to_string())
        return "\n\n".join(overview)
    except Exception as e: st.error(f"Error building dataset summary: {e}"); return "Error generating overview"

# -------------------------
# Gemini ML/Stats analysis integration
# -------------------------
def generate_ml_stats_analysis(content, file_type, stats_analysis=None):
    """Generate ML/Stats-focused analysis using Gemini (structured as requested)"""
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={'temperature':0.7,'top_p':0.8,'top_k':40,'max_output_tokens':4096}
        )
        # Build prompt
        if file_type in ['csv','xlsx'] and stats_analysis:
            prompt = f"""You are a world-class Machine Learning and Statistics professor. Analyze this dataset and provide a focused analysis with these sections (be concise but insightful):

1. EXECUTIVE SUMMARY (150 words):
   - Dataset overview and potential use cases
   - Key statistical insights
   - Recommended ML approaches

2. STATISTICAL ANALYSIS:
   - Distribution patterns and outliers
   - Feature correlations (highlight top 3-5)
   - Data quality and preprocessing needs

3. MACHINE LEARNING INSIGHTS:
   - Best ML algorithms for this data (top 3)
   - Feature engineering suggestions
   - Potential target variables
   - Expected challenges

4. PRACTICE PROBLEMS (10 questions):
   - Statistical fundamentals (2)
   - ML algorithm selection (3)
   - Model evaluation (3)
   - Advanced concepts (2)

5. QUICK IMPLEMENTATION GUIDE:
   - Python code starter template
   - Key libraries to use
   - Common pitfalls

Dataset snippet (truncated to 20000 chars):
{content[:20000]}"""
        else:
            prompt = f"""You are a world-class Machine Learning and Statistics professor. Analyze this material and create a focused study guide with the same sections:

Content snippet (truncated):
{content[:20000]}"""

        response = model.generate_content(prompt)
        return response.text if response and hasattr(response,"text") else "⚠️ Unable to generate response"
    except Exception as e:
        return f"⚠️ Error generating analysis: {e}"

# -------------------------
# API key input
# -------------------------
if not st.session_state["gemini_api_key"]:
    st.sidebar.success("💰 100% FREE - Powered by Google Gemini")
    st.sidebar.info("📊 1,500 requests/day available")
    st.header("Set up your Gemini API Key")
    api_key = st.text_input("Enter your Gemini API Key:", type="password", key="api_input")
    if st.button("Set API Key") and api_key:
        set_gemini_api_key(api_key)
        st.success("API Key set successfully!")
        safe_rerun()
    st.stop()
else:
    try: genai.configure(api_key=st.session_state["gemini_api_key"])
    except Exception: pass

# -------------------------
# Sidebar navigation & recent uploads
# -------------------------
with st.sidebar:
    st.title("Navigation Hub")
    nav = st.radio("Navigate to", ["Chat","Uploads","Reference","Settings"], index=0)
    st.divider()
    st.header("Recent uploads")
    if not st.session_state["uploads"]: st.markdown("<div class='file-entry'>No recent uploads</div>", unsafe_allow_html=True)
    else:
        for f in reversed(st.session_state["uploads"][-10:]):
            size_kb=len(f["content"])/1024; uploaded_at=f.get("uploaded_at","unknown")
            st.markdown(f"<div class='file-entry'><strong>{f['name']}</strong><br><span style='color:{MUTED}; font-size:12px;'>{size_kb:.1f} KB — {uploaded_at}</span></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Clear recent uploads"): st.session_state["uploads"]=[]; safe_rerun()
    if st.button("Clear chat"): st.session_state["messages"]=[]; safe_rerun()
    st.divider()
    st.markdown("### Shortcuts")
    st.markdown("<div class='kb-hint'>Press Enter to send (Shift+Enter for newline). Drag & drop files to upload.</div>", unsafe_allow_html=True)

# -------------------------
# Main pages
# -------------------------
if nav=="Uploads":
    st.header("Upload Materials")
    files=st.file_uploader("Choose files to analyze",accept_multiple_files=True,type=['txt','pdf','docx','csv','xlsx'])
    if files:
        for uf in files:
            content=uf.read()
            st.session_state["uploads"].append({"name":uf.name,"content":content,"uploaded_at":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),"type":uf.type or uf.name.split('.')[-1]})
            add_message("assistant", f"Uploaded {uf.name}: {uf.type.upper()} detected.")
    st.info("Uploaded files will appear in the sidebar and can be analyzed in chat.")

elif nav=="Reference":
    st.header("Quick Reference")
    st.markdown("- ML Algorithms: Linear/Logistic Regression, Decision Trees, SVM, KNN, Neural Networks, Clustering, PCA/t-SNE\n- Statistical Methods: Hypothesis Testing, Regression, Time Series, Experimental Design")

elif nav=="Settings":
    st.header("Settings & Preferences")
    if st.button("Reset all except API key"): key=st.session_state.get("gemini_api_key",""); st.session_state.clear(); st.session_state["gemini_api_key"]=key; safe_rerun()

else:
    # CHAT PAGE
    st.header("Chat")

    # Display chat messages
    chat_box = st.empty()
    def render_chat():
        for msg in st.session_state["messages"]:
            role = msg['role']
            content = msg['content']
            role_class = "msg-user" if role=="user" else "msg-bot"
            chat_box.markdown(f"<div class='msg-bubble {role_class} clearfix'>{content}</div>", unsafe_allow_html=True)
    render_chat()

    # Input box
    user_input = st.text_area("Your message:", "", key="chat_input", height=80)
    if st.button("Send"):
        if user_input.strip():
            add_message("user", user_input)
            render_chat()

            # Generate assistant response
            try:
                with st.spinner("🤖 Assistant is typing..."):
                    system_context = "You are a helpful, expert ML and Stats assistant. Provide concise explanations with LaTeX when useful."
                    # Add uploaded file snippet
                    file_context = ""
                    if st.session_state["uploads"]:
                        last_file = st.session_state["uploads"][-1]
                        snippet = (last_file["content"][:3000].decode("utf-8", errors="ignore")
                                   if isinstance(last_file["content"], (bytes, bytearray)) else str(last_file["content"])[:3000])
                        file_context = f"\n\nContext from uploaded file ({last_file['name']}):\n{snippet}"

                    # Build prompt
                    conv_text = system_context + file_context + "\n\nConversation:\n"
                    for msg in st.session_state["messages"][-8:]:
                        conv_text += f"{msg['role']}: {msg['content']}\n"

                    # Call Gemini
                    model = genai.GenerativeModel(
                        'gemini-2.5-flash',
                        generation_config={
                            'temperature': 0.2 if not st.session_state.get("verbose", False) else 0.7,
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 2048,
                        }
                    )
                    response = model.generate_content(conv_text)
                    assistant_text = response.text if hasattr(response, "text") else "Sorry, could not generate a response."

                    add_message("assistant", assistant_text)
                    render_chat()
            except Exception as e:
                add_message("assistant", f"⚠️ Error generating response: {e}")
                render_chat()

    # Export
    st.divider()
    st.markdown("### Export Chat")
    col1, col2 = st.columns(2)
    with col1:
        docx_file = export_chat_as_docx(st.session_state["messages"])
        st.download_button("Download DOCX", docx_file, file_name="chat_export.docx")
    with col2:
        if PDF_AVAILABLE:
            pdf_file = export_chat_as_pdf(st.session_state["messages"])
            st.download_button("Download PDF", pdf_file, file_name="chat_export.pdf")
        else:
            st.info("PDF export unavailable (reportlab missing)")

st.markdown("---")
st.markdown("<div style='color:#98a0b6; font-size:12px;'>Tip: Upload files and ask specific questions. This demo contains stubbed and limited analysis.</div>", unsafe_allow_html=True)
