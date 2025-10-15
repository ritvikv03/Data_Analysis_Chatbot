# Data Analytics Chatbot 🤖

An interactive **Streamlit app** that helps you simplify and understand complex data analytics concepts.  
You can upload lecture notes, datasets (CSV, Excel), PDFs, or Word documents — and the chatbot will provide simplified explanations, insights, and practice examples powered by **Google Gemini AI**.

---

## Features

- Chatbot powered by **Google Gemini AI**
- Upload files: PDF, DOCX, XLSX, CSV
- Automatically extract and analyze tabular data
- Generate summaries, explanations, and insights
- Perfect for students or professionals studying **data analytics** and **statistics**

---

## Requirements

This app uses the following dependencies:

- `streamlit`
- `google-generativeai`
- `pdfplumber`
- `pandas`
- `python-docx`
- `openpyxl`
- `numpy`

These are all listed in the `pyproject.toml` file.

---

## Local Setup

### Clone this repository
```bash
1) git clone https://github.com/<your-username>/data-analytics-chatbot.git
2) cd data-analytics-chatbot
3) pip install streamlit google-generativeai pdfplumber pandas python-docx openpyxl numpy
4) streamlit run chatbot.py


