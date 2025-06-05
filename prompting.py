import json
import os

# --- Knowledge Base Utility ---
def get_knowledge_prompt_string():
    knowledge_base_file = "hess_knowledge_base.md"
    try:
        if os.path.exists(knowledge_base_file):
            with open(knowledge_base_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Knowledge base file '{knowledge_base_file}' not found.")
            return "**Critical Error: HESS Knowledge Base file (hess_knowledge_base.md) not found.**"
    except Exception as e:
        print(f"Error loading Knowledge Base from '{knowledge_base_file}': {e}")
        return f"**Critical Error: Could not load HESS Knowledge Base: {e}**"

# === Enhanced Planner Prompt ===
def generate_planner_prompt(columns, min_timestamp=None, max_timestamp=None, filename=None, last_summary=None):
    if not columns:
        columns = ["Dataframe columns not available or data not loaded"]
    column_list_str = ", ".join([f"'{col}'" for col in columns])

    if filename:
        data_filename_str = f"from the HESS data file '{filename}'"
    else:
        data_filename_str = "from the loaded HESS data"

    data_context = """
**Current Data Context:**
- Data Source: Pre-loaded pandas DataFrame `df` """ + data_filename_str + """
- Available Columns in `df`: """ + column_list_str + """
- Timestamp Column: 'Timestamp' (datetime format, typically 1-second resolution)
- Data Time Range: """ + (min_timestamp if min_timestamp else 'Not specified') + """ to """ + (max_timestamp if max_timestamp else 'Not determined') + """
- Data Volume: Assume `df` can be large (potentially >100k rows for monthly data)

**Data Quality Notes:**
- Power values follow HESS convention: positive = charging, negative = discharging
- Some rows may have missing/invalid timestamps (NaT values)
- Numeric columns may contain NaN values requiring handling
"""

    previous_interaction_context = ""
    if last_summary:
        previous_interaction_context = """
**Previous Interaction Summary (for follow-up context):**
---
""" + last_summary + """
---
"""

    knowledge_base_content = get_knowledge_prompt_string()
    knowledge_base_context = """
**HESS Knowledge Base (for domain understanding and planning):**
--- START KB ---
""" + knowledge_base_content + """
--- END KB ---
"""

    prompt = """
You are an expert **HESS (Home Energy Storage System) Analysis Planner**.
Your role is to understand a user's request about HESS data, consult the HESS Knowledge Base for context, and generate a structured JSON plan for analysis with focus on explainability and confidence.

""" + data_context + """

""" + previous_interaction_context + """

""" + knowledge_base_context + """

**Your Core Mission: Explainable AI Analysis**
Every analysis must provide:
1. **Direct Answer** - Clear response to user's question
2. **Contextual Explanation** - Why this matters for HESS systems
3. **Limitations** - What the analysis cannot determine
4. **Confidence Rating** - High/Medium/Low with justification

**Intent Categories (in order of precedence):**

1. **Conversational/Off-topic**: Greeting, thanks, or non-HESS topics
   ```json
   [{"action": "pass", "parameters": {}, "description": "Conversational response"}]
   ```

2. **Direct Knowledge Base Query**: Factual questions answerable from KB alone
   ```json
   [{"action": "direct_answer", "parameters": {"question": "User's KB question text, be specific about what information is needed from the KB"}, "description": "Answer from KB"}]
   ```


3. **Data Analysis Request**: Requires calculations, plotting, or data manipulation
   - Plan Structure: Sequential steps with validation and explanation focus
   - Each step must contribute to explainable analysis

**CRITICAL PLOTTING RULES:**
- For `plot_data` actions: `columns_to_plot` should ONLY contain data columns (voltage_volts, power_watts, etc.)
- NEVER include 'Timestamp' in `columns_to_plot` - it's handled automatically as x-axis
- Always include plot validation and interpretation steps

**Enhanced Action Types for Explainable Analysis:**

1. **validate_data_quality** - Check data completeness and quality
   ```json
   {"action": "validate_data_quality", "parameters": {"required_columns": ["voltage_volts"], "time_range_check": true}, "description": "Ensure data quality for reliable analysis"}
   ```

2. **plot_data** - Create visualizations (FIXED plotting logic)
   ```json
   {"action": "plot_data", "parameters": {"columns_to_plot": ["voltage_volts"], "plot_title": "Battery Voltage Over Time", "include_analysis": true}, "description": "Visualize voltage patterns with contextual analysis"}
   ```

3. **calculate_metrics** - Compute HESS-specific metrics
   ```json
   {"action": "calculate_metrics", "parameters": {"metric_type": "operational_states", "benchmark_comparison": true}, "description": "Calculate operational states and compare to research benchmarks"}
   ```

4. **analyze_patterns** - Detect and explain patterns
   ```json
   {"action": "analyze_patterns", "parameters": {"pattern_type": "daily_cycles", "seasonal_context": true}, "description": "Analyze daily operational patterns with seasonal considerations"}
   ```

5. **assess_health_indicators** - Battery health assessment
   ```json
   {"action": "assess_health_indicators", "parameters": {"indicators": ["voltage_behavior", "temperature_performance"], "chemistry_specific": true}, "description": "Assess battery health using chemistry-specific indicators"}
   ```

6. **interpret_results** - Provide explanations and context
   ```json
   {"action": "interpret_results", "parameters": {"focus": "user_implications", "confidence_assessment": true}, "description": "Interpret findings with user-focused explanations and confidence rating"}
   ```

**Example Enhanced Plans:**

**User Request:** "Plot my battery voltage"
```json
[
  {"step_id": 1, "action": "validate_data_quality", "parameters": {"required_columns": ["voltage_volts", "Timestamp"], "check_completeness": true}, "description": "Validate voltage data availability and quality"},
  {"step_id": 2, "action": "plot_data", "parameters": {"columns_to_plot": ["voltage_volts"], "plot_title": "Battery Voltage Over Time", "chemistry_context": true}, "description": "Create voltage plot with chemistry-specific context"},
  {"step_id": 3, "action": "analyze_patterns", "parameters": {"pattern_type": "voltage_behavior", "benchmark_comparison": true}, "description": "Analyze voltage patterns against expected ranges for system chemistry"},
  {"step_id": 4, "action": "interpret_results", "parameters": {"focus": "voltage_health_implications", "limitations": "short_term_data"}, "description": "Explain voltage behavior implications with confidence assessment"}
]
```

**User Request:** "Check my battery health"
```json
[
  {"step_id": 1, "action": "validate_data_quality", "parameters": {"required_columns": ["voltage_volts", "current_amperes", "temp_battery_celsius"], "time_span_check": true}, "description": "Ensure sufficient data for health assessment"},
  {"step_id": 2, "action": "calculate_metrics", "parameters": {"metric_type": "operational_states", "include_benchmarks": true}, "description": "Calculate operational state distribution vs research benchmarks"},
  {"step_id": 3, "action": "assess_health_indicators", "parameters": {"indicators": ["voltage_range", "temperature_performance", "cycling_behavior"], "chemistry_specific": true}, "description": "Assess multiple health indicators using chemistry-specific thresholds"},
  {"step_id": 4, "action": "plot_data", "parameters": {"columns_to_plot": ["voltage_volts", "temp_battery_celsius"], "plot_title": "Health Indicators Over Time", "include_thresholds": true}, "description": "Visualize key health indicators with normal operating ranges"},
  {"step_id": 5, "action": "interpret_results", "parameters": {"focus": "health_summary", "actionable_recommendations": true, "confidence_rating": true}, "description": "Provide comprehensive health assessment with confidence level and recommendations"}
]
```

**User Request:** "What are my system specifications?"
```json
[
  {"action": "direct_answer", "parameters": {"question": "User query: What are my system specifications? Provide details such as Capacity nominal in Ah, Voltage nominal in V, Energy nominal in kWh, Manufacturer, Chemistry, System Age, and Capacity test dates for the current HESS System ID from the Knowledge Base."}, "description": "Provide system specifications from the HESS Knowledge Base based on the user's question."}
]
```

**Planning Guidelines:**
1. **Always include interpretation steps** for explainability
2. **Use chemistry-specific context** from knowledge base
3. **Include benchmark comparisons** when relevant
4. **Plan for confidence assessment** in final steps
5. **Consider data limitations** in planning
6. **Break complex requests** into logical, explainable steps

**Response Format:** Respond ONLY with valid JSON. No additional text or explanations.

---
**User Request:**
"""
    return prompt.strip()

# === Enhanced Executor Prompt ===
def generate_executor_prompt(step_details, columns, preview, min_timestamp=None, max_timestamp=None, filename=None):
    if not columns:
        columns = ["Dataframe columns not available or data not loaded"]
    column_list_str = ", ".join([f"'{col}'" for col in columns])
    
    if filename:
        data_filename_str = f"from HESS data file '{filename}'"
    else:
        data_filename_str = "from loaded HESS data"
    
    time_range_info = ""
    if min_timestamp and max_timestamp:
        time_range_info = f"- Data covers period: {min_timestamp} to {max_timestamp}"
    
    data_context = """
**Current Data Context:**
- Data Source: Pre-loaded pandas DataFrame `df` """ + data_filename_str + """
- Available Columns: """ + column_list_str + """
- Timestamp Column: 'Timestamp' (datetime format)
- Data Time Range: """ + (min_timestamp if min_timestamp else 'Not specified') + """ to """ + (max_timestamp if max_timestamp else 'Not determined') + """
""" + time_range_info + """

**CRITICAL DATA HANDLING RULES:**
1. Always check column existence: `if 'column_name' in df.columns:`
2. Handle missing timestamps: Filter out NaT values for time-based analysis
3. Use .copy() for DataFrame modifications: `df_work = df.copy()`
4. Handle NaN values: Use .dropna() or .fillna() as appropriate
5. For plotting: NEVER include 'Timestamp' in columns_to_plot - it's the x-axis
"""
    
    step_details_str = json.dumps(step_details, indent=2)
    action = step_details.get('action', 'UnknownAction')
    params_dict = step_details.get('parameters', {})
    step_id = step_details.get('step_id', 'N/A')
    
    knowledge_base_content = get_knowledge_prompt_string()
    knowledge_base_context = """
**HESS Knowledge Base (CRITICAL for analysis logic):**
--- START KB ---
""" + knowledge_base_content + """
--- END KB ---
"""
    
    # Create the enhanced plotting template as a simple string - NO f-strings!
    params_str = str(params_dict)
    enhanced_plot_template = """
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.io as pio

final_output_dict = None

try:
    action_params = """ + params_str + """
    
    # FIXED PLOTTING LOGIC - NO DUPLICATE TIMESTAMPS
    columns_to_plot = action_params.get("columns_to_plot", [])
    
    # CRITICAL: Remove any 'Timestamp' entries from columns_to_plot
    if isinstance(columns_to_plot, list):
        columns_to_plot = [col for col in columns_to_plot if col != 'Timestamp' and col in df.columns]
    
    if not columns_to_plot:
        final_output_dict = {"status": "Failed", "message": "No valid data columns to plot"}
    elif 'Timestamp' not in df.columns:
        final_output_dict = {"status": "Failed", "message": "No Timestamp column found"}
    else:
        # Prepare clean plotting data
        plot_columns = ['Timestamp'] + columns_to_plot
        df_plot = df[plot_columns].copy()
        
        # Remove rows with invalid timestamps or NaN values
        df_plot = df_plot.dropna()
        
        if len(df_plot) == 0:
            final_output_dict = {"status": "Failed", "message": "No valid data after removing NaN values"}
        else:
            # Resample if dataset is too large
            if len(df_plot) > 20000:
                df_plot.set_index('Timestamp', inplace=True)
                df_plot = df_plot.resample('15T').mean().reset_index()
                resampled_info = "Data resampled to 15-minute intervals"
            else:
                resampled_info = "Original resolution used"
            
            plot_title = action_params.get('plot_title', 'HESS Data')
            
            # Create enhanced plot with contextual information
            fig = px.line(df_plot, x='Timestamp', y=columns_to_plot, title=plot_title)
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            # Add chemistry-specific context if requested
            if action_params.get('chemistry_context') and 'voltage_volts' in columns_to_plot:
                # Add reference lines for typical voltage ranges (from KB)
                fig.add_hline(y=46.0, line_dash="dash", line_color="green", annotation_text="Nominal Voltage")
            
            if action_params.get('include_thresholds') and 'temp_battery_celsius' in columns_to_plot:
                fig.add_hline(y=40.0, line_dash="dash", line_color="red", annotation_text="Warning Temp")
                fig.add_hrect(y0=15.0, y1=35.0, fillcolor="green", opacity=0.1, annotation_text="Optimal Range")
            
            final_output_dict = {
                "plot_json": pio.to_json(fig),
                "metadata": {
                    "title": plot_title,
                    "library": "plotly",
                    "columns_plotted": columns_to_plot,
                    "data_points": len(df_plot),
                    "resampled_info": resampled_info,
                    "analysis_ready": True
                }
            }
            
except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Plotting error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
"""
    
    preview_text = preview if preview else "No data preview available"
    
    prompt = """
You are an expert **HESS Analysis Code Generator** focused on explainable, reliable analysis.
Your task is to write Python code for the specific analysis step below.

""" + data_context + """

**Data Preview (first 5 rows):**
```
""" + preview_text + """
```

""" + knowledge_base_context + """

**Your Specific Task (Step """ + str(step_id) + """):**
```json
""" + step_details_str + """
```
Action: """ + action + """
Parameters: """ + str(params_dict) + """

**Code Generation Rules:**

1. **Essential Imports:** Start with required imports
   ```python
   import pandas as pd
   import numpy as np
   import json
   ```

2. **Robust Data Handling:**
   - Always validate column existence: `if 'column_name' in df.columns:`
   - Handle missing data: `.dropna()` for analysis, `.fillna()` where appropriate
   - Work with copies: `df_work = df.copy()` for modifications
   - Check for NaT timestamps and handle appropriately

3. **Knowledge Base Integration:**
   - Use KB information for domain-specific calculations
   - Apply chemistry-specific logic (LFP, NMC, LMO behaviors)
   - Reference research benchmarks for comparisons
   - Include normal operating ranges and thresholds

4. **Explainable Analysis Pattern:**
   ```python
   final_output_dict = {"status": "Failed", "message": "Default error"}
   
   try:
       # Data validation
       if 'required_column' not in df.columns:
           final_output_dict = {"status": "Failed", "message": "Missing required column"}
       else:
           # Analysis code here
           
           # Calculate results with context
           result_value = your_calculation()
           
           # Add explainability
           interpretation = "What this means for the user..."
           limitations = "What this analysis cannot determine..."
           confidence = "High/Medium/Low based on data quality"
           
           final_output_dict = {
               "status": "Success",
               "primary_result": result_value,
               "interpretation": interpretation,
               "limitations": limitations,
               "confidence": confidence,
               "methodology": "Brief description of approach",
               "benchmark_comparison": "How results compare to typical values"
           }
   except Exception as e:
       final_output_dict = {"status": "Failed", "message": f"Analysis error: {str(e)}"}
   ```

5. **Action-Specific Templates:**

**For plot_data actions:** Use this FIXED template:
""" + enhanced_plot_template + """

**For calculate_metrics actions:**
```python
# Use knowledge base benchmarks for comparison
# Example: Operational states should be ~80-85% idle, 8-10% charging/discharging
```

**For validate_data_quality actions:**
```python
# Check data completeness, timestamp validity, value ranges
# Provide quality score and specific issues found
```

**For assess_health_indicators actions:**
```python
# Use chemistry-specific thresholds from KB
# Compare against normal operating ranges
# Provide health score with explanations
```

6. **Final Output:** Always end with `print(json.dumps(final_output_dict, indent=2))`

**Response Format:** Provide ONLY Python code in ```python ``` blocks. No explanations outside the code.
"""
    return prompt.strip()

# === Enhanced Summary Prompt ===
def generate_summary_prompt(user_query: str, plan: list, step_results: list, last_summary=None, filename=None, model_name=None):
    """Generate comprehensive, explainable summary with confidence assessment"""

    is_claude = model_name and "claude" in model_name.lower()
    
    if is_claude:
        # Truncate step results to essential info only
        truncated_results = []
        for success, output_str, extra in step_results:
            if success:
                try:
                    parsed = json.loads(output_str)
                    # Keep only essential fields, remove massive plot data
                    clean_result = {
                        "status": parsed.get("status", "Success"),
                        "primary_result": str(parsed.get("primary_result", ""))[:200],
                        "interpretation": str(parsed.get("interpretation", ""))[:200]
                    }
                    truncated_results.append((True, json.dumps(clean_result), extra))
                except:
                    truncated_results.append((True, output_str[:200] + "...", extra))
            else:
                truncated_results.append((False, str(output_str)[:200] + "...", extra))
        step_results = truncated_results

    if filename:
        data_filename_str = f"from HESS data file '{filename}'"
    else:
        data_filename_str = "from the loaded HESS data"
    
    previous_interaction_context = ""
    if last_summary:
        previous_interaction_context = """
**Context from Previous Analysis:**
---
""" + last_summary + """
---
The current query may be a follow-up to this previous analysis.
"""
    
    # Process step results for explainable summary
    successful_results = []
    failed_steps = []
    plot_generated = False
    
    for i, step_info_tuple in enumerate(step_results):
        success_flag, output_content = step_info_tuple[0], step_info_tuple[1]
        
        current_plan_step = {}
        if isinstance(plan, list) and i < len(plan) and isinstance(plan[i], dict):
            current_plan_step = plan[i]
        
        step_id = current_plan_step.get('step_id', i + 1)
        action = current_plan_step.get('action', 'Unknown')
        description = current_plan_step.get('description', 'N/A')
        
        if success_flag:
            try:
                parsed_output = json.loads(output_content) if isinstance(output_content, str) else output_content
                successful_results.append({
                    "step_id": step_id,
                    "action": action,
                    "description": description,
                    "output": parsed_output
                })
                
                # Check for plot generation
                if isinstance(parsed_output, dict) and ("plot_json" in parsed_output or "plot_base64" in parsed_output):
                    plot_generated = True
                    
            except json.JSONDecodeError:
                successful_results.append({
                    "step_id": step_id,
                    "action": action,
                    "description": description,
                    "output": {"raw_output": str(output_content)}
                })
        else:
            failed_steps.append({
                "step_id": step_id,
                "action": action,
                "description": description,
                "error": output_content
            })
    
    knowledge_base_content = get_knowledge_prompt_string()
    knowledge_base_context = """
**HESS Knowledge Base (for interpreting results and providing context):**
--- START KB ---
""" + knowledge_base_content + """
--- END KB ---
"""
    
    # Build plot status message without f-strings
    plot_status = "✅ Plot generated and should be visible" if plot_generated else "❌ No plot generated"
    
    # Build visualization section without f-strings
    visualization_section = ""
    if plot_generated:
        visualization_section = "- **Visualization:** The generated plot shows [describe what the plot reveals]"
    
    prompt = """
You are an expert **HESS Analysis Communicator** specializing in explainable AI responses.
Your role is to provide comprehensive, understandable, and trustworthy answers about Home Energy Storage Systems.

**User's Original Query:**
\"""" + user_query + """\" (Analysis performed on data """ + data_filename_str + """)

""" + previous_interaction_context + """

**Analysis Plan Executed:**
```json
""" + json.dumps(plan, indent=2) + """
```

**Successful Analysis Results:**
```json
""" + json.dumps(successful_results, indent=2) + """
```

**Failed Steps (if any):**
```json
""" + json.dumps(failed_steps, indent=2) + """
```

**Plot Generation Status:** """ + plot_status + """

""" + knowledge_base_context + """

**Your Task: Create an Explainable AI Response**

You can structure your response as follows if the user query is about data analysis or system performance:

## 🎯 **Direct Answer**
[Directly answer the user's question in 1-2 clear sentences]

## 📊 **Analysis Results**
[Present the key findings from successful analysis steps]
""" + visualization_section + """

## 🔍 **Contextual Explanation**
[Explain what these results mean in the context of HESS systems, using knowledge base information]
- Compare against research benchmarks where relevant
- Explain the significance for battery health/performance
- Reference chemistry-specific behaviors if applicable

## ⚠️ **Limitations & Considerations**
[Be transparent about what the analysis cannot determine]
- Data limitations (time period, missing measurements, etc.)
- Analysis scope restrictions
- Factors not considered in this assessment

## 🎯 **Actionable Insights**
[If applicable, provide practical recommendations or next steps]

## 📈 **Confidence Assessment**
**Confidence Rating:** [High/Medium/Low]
**Justification:** [Explain the confidence level based on:]
- Data quality and completeness
- Analysis method reliability
- Alignment with expected patterns from knowledge base
- Impact of any failed analysis steps

---

**Response Guidelines:**
1. **Be Direct:** Start with a clear answer to the user's question
2. **Be Explainable:** Use knowledge base context to explain why results matter
3. **Be Honest:** Acknowledge limitations and uncertainties
4. **Be Actionable:** Provide practical insights where possible
5. **Be Confident:** Assess and justify your confidence level
6. **Use Domain Knowledge:** Reference HESS research and benchmarks appropriately

**Writing Style:**
- Clear, conversational tone appropriate for battery system owners
- Technical accuracy with accessible explanations
- Structured format for easy scanning
- Confidence without overstatement

**Assistant Response:**
"""
    return prompt.strip()