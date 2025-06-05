import json
import os
import re

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

#for claude and gpt4o due to their token limits, we need to truncate the knowledge base.
def get_truncated_knowledge_base():
    """Get knowledge base truncated at 'Key Formulas' section"""
    try:
        with open("hess_knowledge_base.md", 'r', encoding='utf-8') as f:
            full_kb = f.read()
        
        # Find the "Key Formulas" section
        key_formulas_pattern = r"## \d+\.\s*Key Formulas"
        match = re.search(key_formulas_pattern, full_kb, re.IGNORECASE)
        
        if match:
            # Find the end of the Key Formulas section (next ## heading or end of file)
            start_pos = match.start()
            # Look for the next ## heading after Key Formulas
            remaining_text = full_kb[start_pos:]
            next_section_match = re.search(r"\n## \d+\.", remaining_text)
            
            if next_section_match:
                # Truncate at the next section
                end_pos = start_pos + next_section_match.start()
                truncated_kb = full_kb[:end_pos]
            else:
                # Include the Key Formulas section to the end
                truncated_kb = full_kb
        else:
            # Fallback: if no Key Formulas found, look for "Remember:" which usually comes before research sections
            remember_match = re.search(r"Remember:.*?(?=\n## \d+\.)", full_kb, re.DOTALL | re.IGNORECASE)
            if remember_match:
                truncated_kb = full_kb[:remember_match.end()]
            else:
                # Last resort: take first 60% of the knowledge base
                truncated_kb = full_kb[:int(len(full_kb) * 0.6)]
        
        print(f"Knowledge base truncated from {len(full_kb)} to {len(truncated_kb)} characters")
        return truncated_kb
        
    except FileNotFoundError:
        print("Warning: Knowledge base file not found")
        return "**HESS Analysis Context:** Home Energy Storage System data analysis."
    except Exception as e:
        print(f"Error truncating knowledge base: {e}")
        return "**HESS Analysis Context:** Home Energy Storage System data analysis."
    

# === Planner Prompt ===
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


# === Executor Prompt ===
def generate_executor_prompt(step_details, columns, preview, min_timestamp=None, max_timestamp=None, filename=None):
    """Generate executor prompt with SIMPLE, WORKING code templates"""
    
    if not columns:
        columns = ["Dataframe columns not available"]
    column_list_str = ", ".join([f"'{col}'" for col in columns])
    
    data_context = f"""
**Data Available:**
- DataFrame: `df` with columns: {column_list_str}
- Timestamp column: 'Timestamp' (if available)
- Always check if columns exist before using them
"""
    
    step_details_str = json.dumps(step_details, indent=2)
    action = step_details.get('action', 'unknown')
    params = step_details.get('parameters', {})
    
    # SIMPLE CODE TEMPLATES that actually work
    if action == "validate_data_quality":
        code_template = '''
import pandas as pd
import numpy as np
import json

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    # Simple data quality check
    total_rows = len(df)
    
    # Check for required columns
    required_cols = ["voltage_volts", "current_amperes", "temp_battery_celsius"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        final_output_dict = {
            "status": "Success",
            "primary_result": f"Missing columns: {missing_cols}",
            "interpretation": "Some data columns are missing for full health assessment",
            "confidence": "Medium"
        }
    else:
        # Count non-null values
        quality_scores = {}
        for col in required_cols:
            if col in df.columns:
                non_null_pct = (df[col].notna().sum() / total_rows) * 100
                quality_scores[col] = round(non_null_pct, 1)
        
        avg_quality = sum(quality_scores.values()) / len(quality_scores)
        
        final_output_dict = {
            "status": "Success",
            "primary_result": f"Data quality: {avg_quality:.1f}% complete",
            "interpretation": f"Quality scores by column: {quality_scores}",
            "confidence": "High" if avg_quality > 90 else "Medium"
        }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "calculate_metrics":
        code_template = '''
import pandas as pd
import numpy as np
import json

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    if 'current_amperes' not in df.columns:
        final_output_dict = {
            "status": "Success",
            "primary_result": "Current data not available",
            "interpretation": "Cannot calculate operational states without current measurements",
            "confidence": "Low"
        }
    else:
        # Simple operational state calculation
        current_data = df['current_amperes'].dropna()
        
        if len(current_data) == 0:
            final_output_dict = {
                "status": "Success", 
                "primary_result": "No valid current data",
                "interpretation": "All current measurements are missing",
                "confidence": "Low"
            }
        else:
            # Define thresholds (simple approach)
            charging = (current_data > 0.1).sum()
            discharging = (current_data < -0.1).sum() 
            idle = len(current_data) - charging - discharging
            
            total = len(current_data)
            idle_pct = (idle / total) * 100
            charge_pct = (charging / total) * 100
            discharge_pct = (discharging / total) * 100
            
            final_output_dict = {
                "status": "Success",
                "primary_result": f"Idle: {idle_pct:.1f}%, Charging: {charge_pct:.1f}%, Discharging: {discharge_pct:.1f}%",
                "interpretation": "Normal systems are ~80-85% idle. Your system shows different patterns.",
                "confidence": "Medium"
            }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "assess_health_indicators":
        code_template = '''
import pandas as pd
import numpy as np
import json

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    health_issues = []
    health_score = 100
    
    # Check voltage if available
    if 'voltage_volts' in df.columns:
        voltage_data = df['voltage_volts'].dropna()
        if len(voltage_data) > 0:
            avg_voltage = voltage_data.mean()
            if avg_voltage < 40:  # Low voltage warning
                health_issues.append("Low average voltage detected")
                health_score -= 20
    else:
        health_issues.append("Voltage data not available")
        health_score -= 10
    
    # Check temperature if available  
    if 'temp_battery_celsius' in df.columns:
        temp_data = df['temp_battery_celsius'].dropna()
        if len(temp_data) > 0:
            high_temp_count = (temp_data > 40).sum()
            if high_temp_count > len(temp_data) * 0.1:  # >10% high temp
                health_issues.append("Frequent high temperature events")
                health_score -= 15
    else:
        health_issues.append("Temperature data not available")
        health_score -= 10
    
    if health_score >= 80:
        health_status = "Good"
    elif health_score >= 60:
        health_status = "Fair" 
    else:
        health_status = "Poor"
    
    final_output_dict = {
        "status": "Success",
        "primary_result": f"Health Score: {health_score}/100 ({health_status})",
        "interpretation": f"Issues found: {health_issues if health_issues else ['None']}", 
        "confidence": "Medium"
    }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "plot_data":
        code_template = '''
import pandas as pd
import numpy as np
import json

# Check if plotly is available
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    if not PLOTLY_AVAILABLE:
        final_output_dict = {"status": "Failed", "message": "Plotly not available for plotting"}
    elif 'Timestamp' not in df.columns:
        final_output_dict = {"status": "Failed", "message": "No Timestamp column for plotting"}
    else:
        # Get columns to plot (exclude Timestamp)
        plot_columns = ["voltage_volts", "temp_battery_celsius"]
        available_columns = [col for col in plot_columns if col in df.columns]
        
        if not available_columns:
            final_output_dict = {"status": "Failed", "message": "No plottable columns found"}
        else:
            # Prepare plotting data
            plot_data = df[['Timestamp'] + available_columns].dropna()
            
            if len(plot_data) == 0:
                final_output_dict = {"status": "Failed", "message": "No valid data after removing NaN values"}
            else:
                # Sample data if too large
                if len(plot_data) > 10000:
                    plot_data = plot_data.sample(n=10000).sort_values('Timestamp')
                
                # Create simple line plot
                fig = px.line(plot_data, x='Timestamp', y=available_columns, 
                             title='Battery Health Indicators Over Time')
                
                final_output_dict = {
                    "plot_json": pio.to_json(fig),
                    "metadata": {
                        "title": "Battery Health Indicators",
                        "library": "plotly", 
                        "columns_plotted": available_columns,
                        "data_points": len(plot_data)
                    }
                }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Plotting error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "interpret_results":
        code_template = '''
import pandas as pd
import numpy as np
import json

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    # Simple interpretation based on available data
    insights = []
    
    if 'voltage_volts' in df.columns:
        voltage_data = df['voltage_volts'].dropna()
        if len(voltage_data) > 0:
            avg_voltage = voltage_data.mean()
            insights.append(f"Average voltage: {avg_voltage:.2f}V")
    
    if 'temp_battery_celsius' in df.columns:
        temp_data = df['temp_battery_celsius'].dropna()
        if len(temp_data) > 0:
            avg_temp = temp_data.mean()
            max_temp = temp_data.max()
            insights.append(f"Temperature range: {avg_temp:.1f}°C avg, {max_temp:.1f}°C max")
    
    if not insights:
        insights = ["Limited data available for interpretation"]
    
    final_output_dict = {
        "status": "Success",
        "primary_result": "Health assessment completed with available data",
        "interpretation": "; ".join(insights),
        "confidence": "Medium"
    }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "direct_answer":
        code_template = '''
import json

# Direct answer - no data processing needed
final_output_dict = {
    "status": "Success", 
    "primary_result": "Direct answer provided",
    "interpretation": "This question was answered directly from the knowledge base",
    "confidence": "High"
}

print(json.dumps(final_output_dict, indent=2))
'''

    elif action == "analyze_patterns":
        code_template = '''
import pandas as pd
import numpy as np
import json

final_output_dict = {"status": "Failed", "message": "Default error"}

try:
    # Simple pattern analysis
    patterns = []
    
    if 'voltage_volts' in df.columns:
        voltage_data = df['voltage_volts'].dropna()
        if len(voltage_data) > 0:
            voltage_std = voltage_data.std()
            voltage_range = voltage_data.max() - voltage_data.min()
            patterns.append(f"Voltage stability: std={voltage_std:.3f}V, range={voltage_range:.3f}V")
    
    if 'current_amperes' in df.columns:
        current_data = df['current_amperes'].dropna()
        if len(current_data) > 0:
            charge_time = (current_data > 0.1).sum()
            discharge_time = (current_data < -0.1).sum()
            patterns.append(f"Activity: {charge_time} charging periods, {discharge_time} discharge periods")
    
    if not patterns:
        patterns = ["Limited data available for pattern analysis"]
    
    final_output_dict = {
        "status": "Success",
        "primary_result": "Pattern analysis completed",
        "interpretation": "; ".join(patterns),
        "confidence": "Medium"
    }

except Exception as e:
    final_output_dict = {"status": "Failed", "message": f"Error: {str(e)}"}

print(json.dumps(final_output_dict, indent=2))
'''

    else:
        # Generic fallback - FIXED
        code_template = f'''
import pandas as pd
import numpy as np
import json

final_output_dict = {{
    "status": "Success",
    "primary_result": "Analysis step completed",
    "interpretation": "Generic analysis for action: {action}",
    "confidence": "Low"
}}

print(json.dumps(final_output_dict, indent=2))
'''

    prompt = f"""
You are a **HESS Code Generator**. Write Python code for this analysis step.

{data_context}

**Your Task:**
{step_details_str}

**IMPORTANT:** Use this EXACT code template (just copy it):

```python
{code_template.strip()}
```

**Rules:**
1. Copy the template exactly as shown
2. Do NOT modify the template 
3. The template is already customized for your action type
4. Always end with print(json.dumps(final_output_dict, indent=2))

**Response:** Provide ONLY the Python code in ```python ``` blocks.
"""
    
    return prompt.strip()

# === Summariser Prompt ===
def generate_summary_prompt(user_query: str, plan: list, step_results: list, last_summary=None, filename=None, model_name=None):
    """Generate comprehensive summary with model-specific knowledge base sizing
    
    Knowledge Base Strategy:
    - GPT-4 & Claude: Use truncated KB (up to Key Formulas) due to token limits
    - Gemini & others: Use full KB since they have larger context windows
    """
    
    # Simple token estimation
    def estimate_tokens(text):
        return len(str(text)) // 4
    
    # MODEL-SPECIFIC TOKEN LIMITS
    token_limit = 25000  # Conservative default
    if model_name:
        model_lower = model_name.lower()
        if 'gpt-4-turbo' in model_lower:
            token_limit = 25000
        elif 'gpt-3.5' in model_lower:
            token_limit = 14000
        elif 'claude' in model_lower:
            token_limit = 180000
        elif 'gemini' in model_lower:
            token_limit = 900000
    
    print(f"Using token limit {token_limit} for model {model_name}")
    
    # Truncate step results if needed
    truncated_results = []
    for success, output_str, extra in step_results:
        if success:
            try:
                parsed = json.loads(output_str)
                if isinstance(parsed, dict):
                    # Keep essential fields, remove plot data
                    clean_result = {
                        "status": parsed.get("status", "Success"),
                        "primary_result": str(parsed.get("primary_result", ""))[:300],
                        "interpretation": str(parsed.get("interpretation", ""))[:200]
                    }
                    # Remove massive plot JSON
                    if "plot_json" in parsed:
                        clean_result["plot_generated"] = "Yes (Plotly chart)"
                    if "plot_base64" in parsed:
                        clean_result["plot_generated"] = "Yes (Image chart)"
                    truncated_results.append((True, json.dumps(clean_result), extra))
                else:
                    truncated_results.append((True, str(output_str)[:300], extra))
            except:
                truncated_results.append((True, str(output_str)[:300], extra))
        else:
            truncated_results.append((False, str(output_str)[:200], extra))
    
    step_results = truncated_results
    
    if filename:
        data_filename_str = f"from HESS data file '{filename}'"
    else:
        data_filename_str = "from the loaded HESS data"
    
    previous_context = ""
    if last_summary:
        previous_context = f"""
**Previous Analysis Context:**
{last_summary[:800]}...
"""
    
    # Process step results
    successful_results = []
    failed_steps = []
    plot_generated = False
    
    for i, (success_flag, output_content, _) in enumerate(step_results):
        current_plan_step = plan[i] if i < len(plan) else {}
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
                
                if isinstance(parsed_output, dict) and ("plot_json" in str(parsed_output) or "plot_generated" in str(parsed_output)):
                    plot_generated = True
                    
            except:
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
                "error": output_content
            })
    
    # Conditionally truncate knowledge base for specific models
    # TRUNCATED KB: GPT-4, Claude (limited context windows)
    # FULL KB: Gemini, other models (larger context windows)
    needs_truncation = False
    if model_name:
        model_lower = model_name.lower()
        # Apply truncation for GPT-4 and Claude models
        if ('gpt-4' in model_lower or 'claude' in model_lower):
            needs_truncation = True
    
    if needs_truncation:
        print(f"Truncating knowledge base for model: {model_name}")
        kb_content = get_truncated_knowledge_base()
        kb_label = "Essential Context"
    else:
        print(f"Using full knowledge base for model: {model_name}")
        kb_content = get_knowledge_prompt_string()  # Full knowledge base
        kb_label = "Complete Context"
    
    knowledge_base_context = f"""
**HESS Knowledge Base ({kb_label}):**
--- START KB ---
{kb_content}
--- END KB ---
"""
    
    plot_status = "✅ Plot generated and should be visible" if plot_generated else "❌ No plot generated"
    visualization_section = "- **Visualization:** The generated plot shows [describe what the plot reveals]" if plot_generated else ""
    
    prompt = f"""You are an expert **HESS Analysis Communicator** specializing in explainable AI responses.
Your role is to provide comprehensive, understandable, and trustworthy answers about Home Energy Storage Systems.

**User's Original Query:**
"{user_query}" (Analysis performed on data {data_filename_str})

{previous_context}

**Analysis Plan Executed:**
```json
{json.dumps(plan, indent=2)}
```

**Successful Analysis Results:**
```json
{json.dumps(successful_results, indent=2)}
```

**Failed Steps (if any):**
```json
{json.dumps(failed_steps, indent=2)}
```

**Plot Generation Status:** {plot_status}

{knowledge_base_context}

**Your Task: Create an Explainable AI Response**

Structure your response as follows:

## 🎯 **Direct Answer**
[Directly answer the user's question in 1-2 clear sentences]

## 📊 **Analysis Results**
[Present the key findings from successful analysis steps]
{visualization_section}

## 🔍 **Contextual Explanation**
[Explain what these results mean in the context of HESS systems, using knowledge base information]
- Compare against normal operating ranges where relevant
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

**Writing Style:**
- Clear, conversational tone appropriate for battery system owners
- Technical accuracy with accessible explanations
- Structured format for easy scanning

**Assistant Response:**
"""
    
    # Check if prompt is too large and apply emergency truncation if needed
    estimated_tokens = estimate_tokens(prompt)
    if estimated_tokens > token_limit:
        print(f"Prompt still too large ({estimated_tokens} tokens), applying emergency truncation...")
        
        # Emergency: Use minimal knowledge base for all models when oversized
        mini_kb = """
**Essential HESS Context:**
- Voltage ranges vary by chemistry (LFP: 2.5-3.8V/cell, NMC: 3.0-4.2V/cell)
- Optimal temperature: 15-35°C, Warning: >40°C
- Power convention: Positive=charging, Negative=discharging
- Normal operation: ~80-85% idle time, 8-10% charging/discharging
"""
        
        prompt = f"""You are an expert HESS Analysis Communicator.

**User Query:** "{user_query}" (Analysis on data {data_filename_str})

{previous_context}

**Analysis Results:**
Successful Steps: {json.dumps(successful_results, indent=2)}
Failed Steps: {json.dumps(failed_steps, indent=2)}
Plot Status: {plot_status}

{mini_kb}

**Create a helpful response with:**

## 🎯 Direct Answer
[Answer the user's question clearly]

## 📊 Key Findings  
[Present the main results from analysis]

## 🔍 What This Means
[Explain significance for battery performance]

## ⚠️ Limitations
[Note what analysis cannot determine]

## 📈 Confidence: [High/Medium/Low]
[Justify confidence level based on data quality and analysis completeness]

Keep response focused, practical, and accessible to battery system owners.
"""
    
    return prompt