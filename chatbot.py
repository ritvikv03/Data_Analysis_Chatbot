# run 'pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy' in terminal to install dependencies
# run streamlit run chatbot.py in terminal to start the app

import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
from docx import Document
import numpy as np
import io


st.set_page_config(
   page_title="Data Analytics Helper", 
   layout="wide"
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
    
    /* Better spacing */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Prettier chat input */
    .stChatInput {
        border-radius: 25px;
    }
    
    /* Better buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Nicer file uploader */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Better expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Data Analytics Chatbot 🤖")
st.markdown("AI-Assistant to help simplify complex topics about data! Upload lecture notes, datasets, or statistical analysis documents to get simplified insights, practice problems, and implementation tips!")

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

# API Key Setup
if not get_gemini_api_key():
    st.success("🎉 **100% FREE!** No payment required - just get a free API key")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input("Enter your FREE Gemini API Key:", type="password", key="api_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Set API Key", type="primary"):
            if api_key:
                set_gemini_api_key(api_key)
                st.success("✅ API Key set successfully!")
                st.rerun()
            else:
                st.error("Please enter a valid API key.")
    
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

# Main Application
st.divider()

# File Upload Section
st.header("📁 Upload Materials Below")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['txt', 'pdf', 'docx', 'csv', 'xlsx'],
        help="Upload ML research papers, lecture notes, datasets, or statistical analysis documents"
    )

with col2:
    st.markdown("### 📚 Material Types")
    st.markdown("""
    - 📊 **Datasets**: Get analysis recommendations
    - 📄 **Research Papers**: Extract key insights
    - 📝 **Lecture Notes**: Master concepts
    - 📈 **Statistical Reports**: Deep analysis
    """)

# Process uploaded file
if uploaded_file is not None:
    with st.spinner("🔍 Performing advanced analysis..."):
        result = extract_text_from_file(uploaded_file)
        
        if len(result) == 4:
            file_content, file_type, dataframe, stats_analysis = result
        else:
            file_content, file_type, dataframe = result
            stats_analysis = None
        
        if file_content:
            st.success(f"✅ Successfully loaded: **{uploaded_file.name}** ({file_type.upper()})")
            
            # Enhanced data preview for datasets
            if dataframe is not None:
                tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Quick Insights"])
                
                with tab1:
                    st.dataframe(dataframe.head(50), use_container_width=True)
                    
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Rows", f"{len(dataframe):,}")
                    col2.metric("Total Columns", len(dataframe.columns))
                    col3.metric("Missing Values", f"{dataframe.isnull().sum().sum():,}")
                    col4.metric("Memory", f"{stats_analysis['basic_stats']['memory_usage']:.1f} MB")
                    
                    # Show detailed statistics
                    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
                    if numerical_cols:
                        st.subheader("Numerical Features")
                        st.dataframe(dataframe[numerical_cols].describe(), use_container_width=True)
                        
                        # Correlation heatmap info
                        if len(numerical_cols) > 1:
                            st.subheader("Feature Correlations")
                            st.dataframe(stats_analysis['correlations'], use_container_width=True)
                
                with tab3:
                    st.markdown("### 🎯 Quick ML Insights")
                    
                    # Suggest problem type
                    if len(numerical_cols) > 0:
                        st.info("**Suggested ML Tasks**: Regression, Time Series Analysis, Clustering")
                    
                    # Check for class imbalance
                    categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        for col in categorical_cols[:3]:
                            value_counts = dataframe[col].value_counts()
                            if len(value_counts) < 10:
                                st.warning(f"**{col}**: Potential classification target (classes: {len(value_counts)})")
            
            # Generate expert ML/Stats analysis
            st.divider()
            st.header("🎓 Chatbot Analysis")
            
            # Add progress indicator
            progress_text = "⚡ Analyzing your file... may take a moment"
            with st.spinner(progress_text):
                analysis = generate_ml_stats_analysis(
                    file_content, 
                    file_type,
                    stats_analysis
                )
            
            if analysis:
                st.markdown(analysis)
                
                # Download options
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Study Guide (Text)",
                        data=analysis,
                        file_name=f"{uploaded_file.name}_ml_stats_guide.txt",
                        mime="text/plain"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Study Guide (Markdown)",
                        data=analysis,
                        file_name=f"{uploaded_file.name}_ml_stats_guide.md",
                        mime="text/markdown"
                    )
        else:
            st.error("❌ Could not extract content from the file.")

# Advanced ML/Stats Chatbot
st.divider()
st.header("💬 Ask Questions!")

# Initialize chat history FIRST
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Add helpful tips above chat
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("*Ask anything about your uploaded materials or general ML/Stats concepts*")
with col2:
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", key="clear_top"):
            st.session_state.messages = []
            st.rerun()

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show helpful starter questions if no messages yet
if not st.session_state.messages and uploaded_file:
    st.info("💡 **Suggested questions to get started:**\n- What are the main concepts in this file?\n- Can you explain [concept] in simpler terms?\n- What practice problems can you give me?\n- How would I implement this in Python?")
elif not st.session_state.messages:
    st.info("💡 **Try asking:**\n- Explain the main concepts from this file in simple terms\n- Give me 5 true/false questions on [topic] with answers\n- Create 3 practice problems about [concept] and show solutions\n- Explain [specific algorithm] like I'm 5 years old\n- The more specific the better!")

# Chat input
if prompt := st.chat_input("Ask about algorithms, statistics, math, or implementation..."):
    # Add to session state first
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show assistant response with clear loading indicator
    with st.chat_message("assistant"):
        # Create placeholder for the response
        response_placeholder = st.empty()
        
        # Show a friendly loading message
        with response_placeholder.container():
            st.markdown("### 🤔 Processing your question...")
            st.progress(0.5)
            st.caption("⏳ This might take a moment")
        
        try:
            # Use faster model with optimized settings
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
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
            file_context = ""
            if 'uploaded_file' in locals() and uploaded_file and 'file_content' in locals() and file_content:
                file_context = f"\n\nContext from uploaded file ({uploaded_file.name}):\n{file_content[:3000]}"
            
            # Build conversation for Gemini
            full_prompt = f"{context_message}{file_context}\n\nConversation:\n"
            for msg in st.session_state.messages:
                full_prompt += f"\n{msg['role']}: {msg['content']}"
            
            response = model.generate_content(full_prompt)
            assistant_response = response.text
            
            # Replace loading message with actual response
            # Clear loading indicator and show response
            response_placeholder.empty()
            response_placeholder.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            response_placeholder.empty()
            with response_placeholder.container():
                st.warning(f"⚠️ {str(e)}")
                st.info("💡 **Tips...")
            st.info("💡 Tip: Try rephrasing your question or check your API key.")
    
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
    st.header("🎯 Reference & Resource Hub")
    
    st.success("💰 **100% FREE** - Powered by Google Gemini")
    st.info("📊 **1,500 requests/day** available")
    
    st.divider()
    
    st.markdown("""
    ### 📖 Quick Reference
    
    **Core ML Algorithms:**
    - Linear/Logistic Regression
    - Decision Trees & Random Forests
    - SVM, KNN, Naive Bayes
    - Neural Networks & Deep Learning
    - Clustering (K-means, DBSCAN)
    - Dimensionality Reduction (PCA, t-SNE)
    
    **Statistical Methods:**
    - Hypothesis Testing (t-test, ANOVA, χ²)
    - Regression Analysis
    - Bayesian Inference
    - Time Series Analysis
    - Experimental Design
    
    **Key Concepts:**
    - Bias-Variance Tradeoff
    - Cross-Validation
    - Regularization (L1/L2)
    - Feature Engineering
    - Ensemble Methods
    - Gradient Descent
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
