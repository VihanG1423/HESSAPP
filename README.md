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

### 5. Prepare Data Files

Ensure you have:
- `template.md` - Knowledge base template
- `Metadata_Systems.xlsx` - System specifications metadata
- Your HESS CSV data files

### 6. Run Application
```bash
streamlit run main_hess_app.py
```

## Project Structure

```
hess-llm-evaluation/
├── main_hess_app.py           # Main Streamlit application
├── improved_utils.py          # Enhanced utilities and LLM management
├── evaluation_config.py       # Test scenarios and evaluation metrics
├── prompting.py              # Prompt generation (your existing file)
├── cleaning.py               # Data cleaning (your existing file)
├── template.md               # Knowledge base template
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
├── experiment_chat_logs/     # Generated chat logs
├── experiment_results/       # Evaluation results
└── reports/                  # Generated reports
```

## Key Improvements Made

### 1. **Data Processing Fixes**
- ✅ Robust file metadata extraction with multiple patterns
- ✅ Proper temporary file management and cleanup
- ✅ Enhanced error handling and validation
- ✅ Better CSV structure validation

### 2. **LLM Integration Improvements**
- ✅ Unified LLM management across all providers
- ✅ Comprehensive error handling for API failures
- ✅ Automatic fallback when models are unavailable
- ✅ Response time tracking and performance metrics

### 3. **Evaluation Framework**
- ✅ Standardized test scenarios for fair comparison
- ✅ Automated evaluation metrics (accuracy, clarity, completeness)
- ✅ Comparative analysis and ranking system
- ✅ Progress tracking and real-time results

### 4. **User Interface Enhancements**
- ✅ Intuitive Streamlit interface with clear navigation
- ✅ Real-time progress tracking during evaluations
- ✅ Interactive charts and visualizations
- ✅ Export capabilities (JSON, CSV)

### 5. **Experiment Management**
- ✅ Structured experiment tracking with unique IDs
- ✅ Comprehensive logging and audit trails
- ✅ Results aggregation across multiple scenarios
- ✅ Statistical analysis and reporting

## Usage Guide

### Quick Start (Single Query Test)
1. Upload your HESS CSV file
2. Select "Quick Test" mode
3. Choose LLM models to compare
4. Enter or select a query
5. Click "Run Quick Test"
6. View comparative results

### Comprehensive Evaluation (Standardized Scenarios)
1. Upload your HESS CSV file
2. Select "Standardized Scenarios" mode
3. Choose scenario categories in sidebar
4. Select models to compare
5. Run all scenarios or sample
6. Review aggregate performance metrics

### Custom Analysis
1. Upload your HESS CSV file
2. Select "Custom Query" mode
3. Write detailed analysis query
4. Configure code generation options
5. Run evaluation across selected models
6. Export detailed results

## Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: Gemini API not configured
```
- Check `.env` file exists and has correct API key
- Verify API key is valid and has quota remaining

**2. Data Loading Issues**
```
Error: Could not extract metadata from filename
```
- Ensure CSV filename follows pattern: `cleaned_YYYY_MM_System_ID_XX`
- Check that CSV has required columns

**3. Memory Issues with Large Files**
```
Error: DataFrame too large
```
- Consider data sampling for files >1M rows
- Increase system memory or use chunked processing

**4. Module Import Errors**
```
ModuleNotFoundError: No module named 'google.generativeai'
```
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

### Performance Optimization

**For Large Datasets:**
- Enable data resampling for plots (automatically handled)
- Use subset of data for initial testing
- Consider running evaluations in smaller batches

**For Slow API Responses:**
- Check internet connection
- Verify API quotas and rate limits
- Consider using faster models for initial testing

## Advanced Configuration

### Custom Test Scenarios
Edit `evaluation_config.py` to add your own test scenarios:

```python
TestScenario(
    id="custom_001",
    category="your_category",
    user_query="Your custom query here",
    expected_analysis_type="analysis_type",
    success_criteria=["criterion1", "criterion2"],
    difficulty_level="intermediate",
    expected_response_elements=["element1", "element2"]
)
```

### Evaluation Metrics Customization
Modify scoring weights in `evaluation_config.py`:

```python
"response_quality": {
    "accuracy": {"weight": 0.30},      # Increase accuracy importance
    "completeness": {"weight": 0.25},  # Adjust other weights accordingly
    "clarity": {"weight": 0.20},
    "actionability": {"weight": 0.15},
    "technical_correctness": {"weight": 0.10}
}
```

## Next Steps

1. **Validation:** Test with your existing HESS data files
2. **Customization:** Add domain-specific evaluation criteria
3. **Scaling:** Set up batch processing for large-scale evaluation
4. **Integration:** Connect with your existing analysis pipeline
5. **Reporting:** Customize report formats for your requirements

For questions or issues, check the troubleshooting section or review the code comments for detailed implementation notes.