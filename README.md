# HESS LLM Evaluation Platform - Setup Instructions

## Requirements

### Python Dependencies (requirements.txt)

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
python-dotenv>=1.0.0
jinja2>=3.1.0
openpyxl>=3.1.0

# LLM API Libraries
google-generativeai>=0.3.0
openai>=1.0.0
anthropic>=0.7.0

# Optional: For enhanced data processing
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Environment Variables (.env file)

Create a `.env` file in your project root:

```env
# LLM API Keys (at least one required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Experiment configuration
DEFAULT_EXPERIMENT_ID=hess_eval_2025
LOG_LEVEL=INFO
```

## Installation Steps

### 1. Clone/Download Project
```bash
git clone <your-repo-url>
cd hess-llm-evaluation
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

1. **Gemini (Google):**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create API key
   - Add to `.env` file

2. **OpenAI:**
   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create API key
   - Add to `.env` file

3. **Anthropic (Claude):**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create API key
   - Add to `.env` file

### 6. Run Application
```bash
streamlit run app.py
```



For questions or issues, check the troubleshooting section or review the code comments for detailed implementation notes.
