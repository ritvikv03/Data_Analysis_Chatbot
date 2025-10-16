# run 'pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy' in terminal to install dependencies
# run streamlit run chatbot.py in terminal to start the app

import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
from docx import Document
import numpy as np
import json
from datetime import datetime
import io


st.set_page_config(
   page_title="Data Analytics Helper", 
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items={
      'Get Help': 'https://github.com/yourusername/yourrepo',
      'Report a bug': 'https://github.com/yourusername/yourrepo/issues',
      'About': '# Data Analytics Helper\nPowered by Google Gemini'
   }
)

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Better spacing and max width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Prettier chat messages */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Better chat input */
    .stChatInput {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
    }
    
    /* Prettier buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        padding: 0.6rem 1.5rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Better file uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 25px;
        background: #f8f9fa;
    }
    
    /* Nicer tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Better metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Prettier dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Better expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("Data Analytics Chatbot 🤖")
st.markdown("AI-Assistant to help simplify complex topics about data! Upload lecture notes, datasets, or statistical analysis documents to get simplified insights, practice problems, and implementation tips!")

def save_chat_history():
    """Save chat history to browser storage via session state"""
    if "messages" in st.session_state and st.session_state.messages:
        # Store in session state with timestamp
        st.session_state["last_saved"] = datetime.now().isoformat()

def load_chat_history():
    """Load chat history from session state"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    return st.session_state["messages"]

def export_chat_history():
    """Export chat history as JSON"""
    history = {
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(st.session_state.messages),
        "conversations": st.session_state.messages
    }
    return json.dumps(history, indent=2)

# Session state initialization
def get_gemini_api_key():
   return st.session_state.get("gemini_api_key", "")

def set_gemini_api_key(api_key):
    st.session_state["gemini_api_key"] = api_key
    genai.configure(api_key=api_key)

# Advanced ML/Stats analysis functions
def analyze_dataset_statistics(df):
    """Perform comprehensive statistical analysis on dataset"""
    analysis = {
        'basic_stats': {},
        'distributions': {},
        'correlations': None,
        'recommendations': []
    }
    
    # Basic statistics
    analysis['basic_stats'] = {
        'shape': df.shape,
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        analysis['distributions'] = df[numerical_cols].describe().to_dict()
        
        # Correlation matrix for numerical features
        if len(numerical_cols) > 1:
            analysis['correlations'] = df[numerical_cols].corr()
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    analysis['categorical'] = {
        col: df[col].value_counts().head(10).to_dict() 
        for col in categorical_cols[:5]
    }
    
    return analysis, numerical_cols, categorical_cols

def generate_ml_stats_analysis(content, file_type, stats_analysis=None):
    """Generate ML/Stats-focused comprehensive analysis using Gemini"""
    try:
        # Use faster model configuration
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 4096,  # Increased for more complete analysis
            }
        )
        
        # Customize prompt based on file type
        if file_type in ['csv', 'xlsx'] and stats_analysis:
            prompt = f"""You are a world-class Machine Learning and Statistics professor. Analyze this dataset and provide expert guidance.

Dataset Information:
{content[:20000]}

Provide a focused analysis with these sections (be concise but insightful):

1. **EXECUTIVE SUMMARY** (150 words):
   - Dataset overview and potential use cases
   - Key statistical insights
   - Recommended ML approaches

2. **STATISTICAL ANALYSIS**:
   - Distribution patterns and outliers
   - Feature correlations (highlight top 3-5)
   - Data quality and preprocessing needs

3. **MACHINE LEARNING INSIGHTS**:
   - Best ML algorithms for this data (top 3)
   - Feature engineering suggestions
   - Potential target variables
   - Expected challenges

4. **PRACTICE PROBLEMS** (10 questions):
   - Statistical fundamentals (2)
   - ML algorithm selection (3)
   - Model evaluation (3)
   - Advanced concepts (2)

5. **QUICK IMPLEMENTATION GUIDE**:
   - Python code starter template
   - Key libraries to use
   - Common pitfalls"""

        else:
            # For documents (lecture notes, papers, etc.)
            prompt = f"""You are a world-class Machine Learning and Statistics professor. Analyze this academic material and create a focused study guide.

Content to analyze:
{content[:40000]}

Provide a concise analysis with these sections:

1. **EXECUTIVE SUMMARY** (150 words):
   - Main topics and their importance
   - Key concepts covered
   - Real-world applications

2. **CONCEPT BREAKDOWN** (Top 3-5 concepts):
   For each concept:
   - Intuitive explanation
   - Key formulas
   - Common misconceptions

3. **MATHEMATICAL ESSENTIALS**:
   - Core equations with explanations
   - Key assumptions
   - Computational complexity

4. **PRACTICE QUESTIONS** (12 questions):
   - Conceptual (4)
   - Mathematical (4)
   - Application (4)

5. **IMPLEMENTATION TIPS**:
   - Python code patterns
   - Best practices
   - Debugging advice

6. **NEXT STEPS**:
   - Advanced topics to explore
   - Recommended resources"""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return None

def extract_text_from_file(uploaded_file):
    """Extract text from various file types with ML/Stats focus"""
    file_content = ""
    file_type = ""
    
    try:
        if uploaded_file.type == "text/plain":
            file_content = str(uploaded_file.read(), "utf-8")
            file_type = "txt"
            
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                file_content = '\n\n'.join([
                    f"Page {i+1}:\n{page.extract_text()}" 
                    for i, page in enumerate(pdf.pages) 
                    if page.extract_text()
                ])
            file_type = "pdf"
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            file_content = '\n\n'.join([
                paragraph.text for paragraph in doc.paragraphs 
                if paragraph.text.strip()
            ])
            file_type = "docx"
            
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            file_type = "csv"
            
            # Comprehensive statistical analysis
            stats_analysis, num_cols, cat_cols = analyze_dataset_statistics(df)
            
            file_content = f"""Dataset Overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Memory Usage: {stats_analysis['basic_stats']['memory_usage']:.2f} MB
- Numerical Features: {len(num_cols)} ({', '.join(num_cols[:10])})
- Categorical Features: {len(cat_cols)} ({', '.join(cat_cols[:10])})

Missing Values Analysis:
{pd.Series(stats_analysis['basic_stats']['missing_pct']).to_string()}

Statistical Summary (Numerical Features):
{df[num_cols].describe().to_string() if num_cols else 'No numerical features'}

Correlation Analysis:
{stats_analysis['correlations'].to_string() if stats_analysis['correlations'] is not None else 'Not applicable'}

Sample Data (First 10 Rows):
{df.head(10).to_string()}

Categorical Feature Distribution (Top Categories):
{chr(10).join([f"{col}: {list(dist.keys())[:5]}" for col, dist in stats_analysis['categorical'].items()])}"""
            
            return file_content, file_type, df, stats_analysis
            
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            file_type = "xlsx"
            
            stats_analysis, num_cols, cat_cols = analyze_dataset_statistics(df)
            
            file_content = f"""Dataset Overview:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Memory Usage: {stats_analysis['basic_stats']['memory_usage']:.2f} MB
- Numerical Features: {len(num_cols)} ({', '.join(num_cols[:10])})
- Categorical Features: {len(cat_cols)} ({', '.join(cat_cols[:10])})

Missing Values Analysis:
{pd.Series(stats_analysis['basic_stats']['missing_pct']).to_string()}

Statistical Summary (Numerical Features):
{df[num_cols].describe().to_string() if num_cols else 'No numerical features'}

Correlation Analysis:
{stats_analysis['correlations'].to_string() if stats_analysis['correlations'] is not None else 'Not applicable'}

Sample Data (First 10 Rows):
{df.head(10).to_string()}

Categorical Feature Distribution (Top Categories):
{chr(10).join([f"{col}: {list(dist.keys())[:5]}" for col, dist in stats_analysis['categorical'].items()])}"""
            
            return file_content, file_type, df, stats_analysis
            
        return file_content, file_type, None, None
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None

# Load chat history
load_chat_history()

# Header with gradient
st.markdown("""
<div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; margin-bottom: 2rem; color: white;'>
    <h1 style='margin: 0; font-size: 2.5rem;'>🤖 ML & Stats Study Expert</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>
        Your AI-powered learning companion • Powered by Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)

# API Key Setup
if not get_gemini_api_key():
    st.info("🎉 **100% FREE!** No payment required - just get a free API key")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input(
           "Enter your FREE Gemini API Key:", 
           type="password", 
           placeholder="Paste your API key here...",
           help="Get yours free at aistudio.google.com"
        )
       if st.button("🚀 Start Learning", type="primary", use_container_width=True):
          if api_key and api_key.startswith("AI") and len(api_key) > 30:
             set_gemini_api_key(api_key)
             st.success("✅ API Key activated!")
             st.balloons()
             st.rerun()
          else:
             st.error("Please enter a valid API key")
        
    
    with st.expander("📖 How to get your FREE Gemini API Key (takes 30 seconds)"):
        st.markdown("""
        ### Step-by-Step:
        1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click **"Create API Key"**
        4. Copy the key and paste it above
        
        ### Free Tier Limits:
        - ✅ **1,500 requests per day** (more than enough!)
        - ✅ **15 requests per minute**
        - ✅ **1 million tokens per minute**
        - ✅ **No credit card required**
        - ✅ **Never expires**
        
        Perfect for us!
        """)
    
    st.stop()

# Configure Gemini
genai.configure(api_key=get_gemini_api_key())

# Main Application
st.divider()

# File Upload Section
st.markdown("### 📁 Upload Study Materials")

uploaded_file = st.file_uploader(
    "Drop your file here or click to browse",
    type=['txt', 'pdf', 'docx', 'csv', 'xlsx'],
    help="Supports: PDFs, Word docs, Excel, CSV, and text files"
)

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing your file..."):
        result = extract_text_from_file(uploaded_file)
        
        if len(result) == 4:
            file_content, file_type, dataframe, stats_analysis = result
        else:
            file_content, file_type, dataframe = result
            stats_analysis = None
        
        if file_content:
            st.success(f"✅ Loaded: **{uploaded_file.name}** ({file_type.upper()})")
            
            # Enhanced data preview for datasets
            if dataframe is not None:
                tab1, tab2, tab3 = st.tabs(["📊 Preview", "📈 Statistics", "💡 Insights"])
                
                with tab1:
                    st.dataframe(dataframe.head(50), use_container_width=True)
                    
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Rows", f"{len(dataframe):,}")
                    col2.metric("Columns", len(dataframe.columns))
                    col3.metric("Missing", f"{dataframe.isnull().sum().sum():,}")
                    col4.metric("Size", f"{stats_analysis['basic_stats']['memory_usage']:.1f} MB")
                    
                    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
                    if numerical_cols:
                        st.markdown("#### Numerical Features")
                        st.dataframe(dataframe[numerical_cols].describe(), use_container_width=True)
                
                with tab3:
                    st.markdown("#### 🎯 Quick Insights")
                    if len(numerical_cols) > 0:
                        st.info("**Suggested Tasks**: Regression, Clustering, Time Series")
                    
                    categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        for col in categorical_cols[:3]:
                            value_counts = dataframe[col].value_counts()
                            if len(value_counts) < 10:
                                st.success(f"**{col}**: Good classification target ({len(value_counts)} classes)")
            
            # Generate analysis
            st.divider()
            st.markdown("### 🎓 AI-Generated Study Guide")
            
            with st.spinner("⚡ Generating comprehensive analysis... (10-30 seconds)"):
                analysis = generate_ml_stats_analysis(file_content, file_type, stats_analysis)
            
            if analysis:
                st.markdown(analysis)
                
                # Download button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        "📥 Download Study Guide",
                        data=analysis,
                        file_name=f"{uploaded_file.name}_study_guide.md",
                        mime="pdf/markdown",
                        use_container_width=True
                    )

# Advanced ML/Stats Chatbot
st.divider()
st.header("💬 Ask Questions!")

# Display chat history
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
      st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💭 Ask about ML algorithms, statistics, or your uploaded files..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with response_placeholder.container():
            st.markdown("### 🤔 Thinking...")
            st.progress(0.5)
            st.caption("⏳ Processing your question...")
        
        try:
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 4096,
                },
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE_SPEECH': 'block_none',
                    'SEXUALLY_EXPLICIT': 'block_none',
                    'DANGEROUS_CONTENT': 'block_none'
                }
            )
            # Enhanced system context for ML/Stats expertise
            context_message = """You are a world-class Machine Learning and Statistics expert with PhD-level knowledge. Provide clear, accurate, and insightful answers.

When answering:
1. Be concise and thorough but simple
2. Use examples and analogies
3. Include relevant formulas when helpful in LaTeX format
4. Give practical implementation advice with business context
5. Suggest further resources when appropriate"""
            
            # Add file context if available (limit to prevent slowdown)
            if uploaded_file and file_content:
                context += f"\n\nFile context: {file_content[:3000]}"
            
            full_prompt = f"{context}\n\n"
            for msg in st.session_state.messages:
                full_prompt += f"{msg['role']}: {msg['content']}\n"
            
            response = model.generate_content(full_prompt)
            
            if not response.candidates or not response.candidates[0].content.parts:
                assistant_response = "Try rephrasing your question or breaking it into smaller parts."
            else:
                assistant_response = response.text
            
            response_placeholder.empty()
            response_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Auto-save after each exchange
            save_chat_history()
            
        except Exception as e:
            response_placeholder.empty()
            response_placeholder.error(f"❌ Error: {str(e)}")
    
    # Force rerun to show the new message properly
    st.rerun()

# Clear chat button (moved to bottom for better UX)
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.messages:
        if st.button("🗑️ Clear Entire Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat cleared!")
            st.rerun()

# Enhanced sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Actions")
    
    # Chat history management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_json = export_chat_history()
                st.download_button(
                    "📥 Download",
                    chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No chat history yet!")
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
    
    st.divider()
    
    # Stats
    st.markdown("### 📊 Session Stats")
    total_messages = len(st.session_state.messages)
    user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
    
    col1, col2 = st.columns(2)
    col1.metric("Messages", total_messages)
    col2.metric("Questions", user_messages)
    
    if "last_saved" in st.session_state:
        st.caption(f"Last active: {st.session_state['last_saved'][:19]}")
    
    st.divider()
    
    # Quick Reference
    st.markdown("### 📚 Quick Reference")
    with st.expander("🤖 ML Algorithms"):
        st.markdown("""
        - Linear/Logistic Regression
        - Decision Trees & Random Forest
        - SVM, KNN, Naive Bayes
        - Neural Networks
        - K-means, DBSCAN
        - PCA, t-SNE
        """)
    
    with st.expander("📈 Statistics"):
        st.markdown("""
        - t-test, ANOVA, χ²
        - Regression Analysis
        - Bayesian Inference
        - Time Series
        - A/B Testing
        """)
    
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        - Start with simple questions
        - Upload one file at a time
        - Break complex topics down
        - Ask for examples
        - Request practice problems
        """)

    st.divider()
    
    st.markdown("""
    ### 🔗 Recommended Resources
    
    - [Scikit-learn Docs](https://scikit-learn.org)
    - [Towards Data Science](https://towardsdatascience.com)
    - [Papers with Code](https://paperswithcode.com)
    - [Kaggle Learn](https://www.kaggle.com/learn)
    - [3Blue1Brown](https://www.3blue1brown.com)
    """)
    
    st.divider()
    
    if st.button("🔄 Reset Everything", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'gemini_api_key':
                del st.session_state[key]
        st.rerun()

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Made with ❤️ for ML & Stats students • Powered by Google Gemini 2.5 Flash</p>
    <p style='font-size: 0.9rem;'>💡 Tip: Your chat history persists during this session. Export it before closing!</p>
</div>
""", unsafe_allow_html=True)
