# utils.py (Planner-Executor Code Generation Workflow Version 3.0)
# Added multi-LLM support for Gemini, ChatGPT, and Claude with dynamic selection
# Added retry mechanisms for LLM API calls using 'backoff'

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import io      # For capturing stdout
import sys     # For redirecting stdout and exception info
import traceback # For detailed error messages
import json # Needed for handling plot JSON output within executed code, and for logging
import datetime # For logging timestamps
from cleaning import clean_csv # Assuming cleaning.py is in the same directory or accessible
import tempfile
from jinja2 import Template
import re
import calendar
from datetime import datetime, timezone # Explicitly importing datetime from datetime
import csv
import backoff # Added for retry mechanisms

# --- LLM API Imports and Specific Exceptions ---
# Gemini
try:
    from google.generativeai import GenerativeModel, configure as gemini_configure
    from google.api_core import exceptions as google_exceptions # For specific Gemini exceptions
    GEMINI_AVAILABLE = True
    print("Google Gemini API library imported successfully.")
except ImportError:
    print("Warning: Google Gemini API library not found.")
    GEMINI_AVAILABLE = False
    google_exceptions = None # Placeholder if library not found

# OpenAI ChatGPT
try:
    import openai
    OPENAI_AVAILABLE = True
    print("OpenAI API library imported successfully.")
except ImportError:
    print("Warning: OpenAI API library not found.")
    OPENAI_AVAILABLE = False
    # openai exceptions like openai.RateLimitError will not be defined if 'openai' itself fails to import.
    # The backoff decorator will handle this by not being able to catch them specifically if OPENAI_AVAILABLE is False.

# Anthropic Claude
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print("Anthropic Claude API library imported successfully.")
except ImportError:
    print("Warning: Anthropic Claude API library not found.")
    ANTHROPIC_AVAILABLE = False
    # anthropic exceptions will not be defined if 'anthropic' fails.

# In utils.py

def extract_json_from_llm_response(response_text: str) -> dict | list | None:
    """
    Robustly extracts JSON objects or arrays from LLM responses,
    handling common markdown code block enclosures.
    Returns the parsed JSON or None if parsing fails.
    """
    if not response_text or not isinstance(response_text, str):
        print("Warning: Invalid input to extract_json_from_llm_response.")
        return None

    # Attempt 1: Look for JSON within markdown code blocks
    # Matches ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError after stripping markdown: {e}. Content: '{json_str[:200]}...'")
            # Fall through to attempt parsing the whole string if markdown stripping failed to yield valid JSON

    # Attempt 2: Try to parse the whole string as JSON (if no markdown or if stripping didn't work)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        # Attempt 3: Search for the first occurrence of '{' or '[' if the whole string isn't valid JSON
        # This is a more aggressive attempt to find embedded JSON.
        json_like_str = None
        start_brace = response_text.find('{')
        start_bracket = response_text.find('[')

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            json_like_str = response_text[start_brace:]
        elif start_bracket != -1:
            json_like_str = response_text[start_bracket:]

        if json_like_str:
            try:
                return json.loads(json_like_str)
            except json.JSONDecodeError:
                print(f"Final JSONDecodeError for content: '{json_like_str[:200]}...' Original error: {e}")
        else:
             print(f"JSONDecodeError on full text and no JSON markers found: {e}. Content: '{response_text[:200]}...'")

    print(f"Could not parse JSON from response: '{response_text[:200]}...'")
    return None


# --- Plotting Library Availability ---
# Plotly
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    print("Plotly libraries loaded successfully (available for generated code).")
except ImportError:
    print("Warning: Plotly libraries not found. Plot generation by generated code using Plotly will fail.")
    px = None 
    pio = None
    PLOTLY_AVAILABLE = False

# Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt 
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib library loaded successfully (available for generated code).")
except ImportError:
    print("Warning: Matplotlib library not found. Plot generation by generated code using Matplotlib will fail.")
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    print("Seaborn library loaded successfully (available for generated code).")
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Seaborn is available, but Matplotlib (a dependency) is not. Seaborn plotting may fail.")
except ImportError:
    print("Warning: Seaborn library not found. Plot generation by generated code using Seaborn will fail.")
    sns = None
    SEABORN_AVAILABLE = False

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Global variables to store configured clients
_gemini_configured = False
_openai_client = None
_anthropic_client = None

def configure_llm_apis():
    """Configure all available LLM APIs at startup"""
    global _gemini_configured, _openai_client, _anthropic_client
    
    if GEMINI_API_KEY and GEMINI_AVAILABLE:
        try:
            gemini_configure(api_key=GEMINI_API_KEY)
            _gemini_configured = True
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}. Check API key.")
            _gemini_configured = False
    else:
        if not GEMINI_AVAILABLE:
            print("Warning: Gemini library not available for API configuration.")
        else:
            print("Warning: GEMINI_API_KEY environment variable not found.")

    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            openai.api_key = OPENAI_API_KEY 
            _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print("OpenAI API configured successfully.")
        except Exception as e:
            print(f"Error configuring OpenAI API: {e}. Check API key.")
            _openai_client = None
    else:
        if not OPENAI_AVAILABLE:
            print("Warning: OpenAI library not available for API configuration.")
        else:
            print("Warning: OPENAI_API_KEY environment variable not found.")
            
    if ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
        try:
            _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print("Anthropic Claude API configured successfully.")
        except Exception as e:
            print(f"Error configuring Anthropic Claude API: {e}. Check API key.")
            _anthropic_client = None
    else:
        if not ANTHROPIC_AVAILABLE:
            print("Warning: Anthropic library not available for API configuration.")
        else:
            print("Warning: ANTHROPIC_API_KEY environment variable not found.")

configure_llm_apis()

def get_available_llm_models():
    """Return a list of available LLM models based on configured APIs"""
    available_models = []
    if _gemini_configured:
        available_models.extend(["gemini-1.5-flash", "gemini-1.5-pro"])
    if _openai_client:
        available_models.extend(["gpt-3.5-turbo", "gpt-4-turbo"])
    if _anthropic_client:
        available_models.extend(["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"])
    return available_models

def query_llm(prompt, model_name="gemini-1.5-flash"):
    """Send prompt to the specified LLM model and get response"""
    if model_name.startswith("gemini"):
        return query_gemini(prompt, model_name)
    elif model_name.startswith("gpt"):
        return query_openai(prompt, model_name)
    elif model_name.startswith("claude"):
        return query_anthropic(prompt, model_name)
    else:
        return f"Error: Unknown model '{model_name}'. Available models: {get_available_llm_models()}"

# Define retryable exceptions for Gemini if available
gemini_retry_exceptions = ()
if GEMINI_AVAILABLE and google_exceptions:
    gemini_retry_exceptions = (
        google_exceptions.ResourceExhausted,  # Rate limits
        google_exceptions.DeadlineExceeded,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
        google_exceptions.Aborted,
        # google.generativeai.types.generation_types.StopCandidateException might be another,
        # but typically indicates content filtering or other non-retryable issues.
        # For now, focusing on API availability / rate limits.
    )

@backoff.on_exception(backoff.expo,
                      gemini_retry_exceptions if gemini_retry_exceptions else Exception, # Fallback to generic Exception if specific ones not loaded
                      max_tries=5,
                      jitter=backoff.full_jitter)
def query_gemini(prompt, model_name="gemini-1.5-flash"):
    """Send the prompt to Gemini API and get the response."""
    if not _gemini_configured:
        return "Error: Gemini API not configured or unavailable."
    if not GEMINI_AVAILABLE: # Should be caught by _gemini_configured but as a safeguard
        return "Error: Gemini library not available."

    try:
        model = GenerativeModel(model_name)
        print(f"Sending prompt to Gemini API ({model_name}), len {len(prompt)} chars. Preview: {prompt[:200]}...")
        response = model.generate_content(prompt)
        response_text = None

        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            elif candidate.finish_reason != "STOP":
                response_text = f"Error: Response generation stopped due to: {candidate.finish_reason}."
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    response_text += f" Safety Ratings: {candidate.safety_ratings}"
            else:
                response_text = ""
        elif hasattr(response, 'text'):
            response_text = response.text
        elif isinstance(response, str):
            response_text = response

        if response_text is None:
            error_detail = f"Response object structure: {response}" if response else "Response object was None."
            print(f"Empty or unexpected response format received. {error_detail}")
            return f"Error: Received empty or unexpected response format from Gemini. {error_detail}"

        print(f"Response received from Gemini API ({model_name}), len {len(response_text)} chars. Preview: {response_text[:200]}...")
        return response_text

    except Exception as e:
        # If it's one of the retryable exceptions, backoff will handle it.
        # For other exceptions, log and return an error message.
        if GEMINI_AVAILABLE and google_exceptions and isinstance(e, gemini_retry_exceptions):
            print(f"Gemini API retryable error ({type(e).__name__}): {e}. Retrying as per backoff policy...")
            raise # Re-raise for backoff to handle
        
        print(f"Error querying Gemini ({model_name}): {e}")
        traceback.print_exc()
        return f"Error: Unable to get response from Gemini ({model_name}). Details: {type(e).__name__}. Check server logs for more."

# Define retryable exceptions for OpenAI if available
openai_retry_exceptions = ()
if OPENAI_AVAILABLE and hasattr(openai, 'RateLimitError'): # Check if openai module and specific error are loaded
    openai_retry_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIStatusError # For 5xx errors
    )

@backoff.on_exception(backoff.expo,
                      openai_retry_exceptions if openai_retry_exceptions else Exception,
                      max_tries=5,
                      jitter=backoff.full_jitter)
def query_openai(prompt, model_name="gpt-4"): # Default model as per your original code
    """Send the prompt to OpenAI API and get the response."""
    if not _openai_client: # Checks if client is configured
        return "Error: OpenAI API not configured or unavailable."
    if not OPENAI_AVAILABLE: # Redundant if _openai_client check is robust, but safe
         return "Error: OpenAI library not available."

    try:
        print(f"Sending prompt to OpenAI API ({model_name}), len {len(prompt)} chars. Preview: {prompt[:200]}...")
        response = _openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.1
        )
        response_text = response.choices[0].message.content
        if response_text is None:
            return "Error: Received empty response from OpenAI API."
        print(f"Response received from OpenAI API ({model_name}), len {len(response_text)} chars. Preview: {response_text[:200]}...")
        return response_text
    except Exception as e:
        if OPENAI_AVAILABLE and hasattr(openai, 'RateLimitError') and isinstance(e, openai_retry_exceptions):
            print(f"OpenAI API retryable error ({type(e).__name__}): {e}. Retrying as per backoff policy...")
            raise
        print(f"Error querying OpenAI ({model_name}): {e}")
        traceback.print_exc()
        return f"Error: Unable to get response from OpenAI ({model_name}). Details: {type(e).__name__}. Check server logs for more."

# Define retryable exceptions for Anthropic if available
anthropic_retry_exceptions = ()
if ANTHROPIC_AVAILABLE and hasattr(anthropic, 'RateLimitError'):
    anthropic_retry_exceptions = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.APIStatusError # For 5xx errors
    )

@backoff.on_exception(backoff.expo,
                      anthropic_retry_exceptions if anthropic_retry_exceptions else Exception,
                      max_tries=2,
                      jitter=backoff.full_jitter)
def query_anthropic(prompt, model_name="claude-3-opus-20240229"): # Default from your code, changed from claude-3-sonnet
    """Send the prompt to Anthropic Claude API and get the response."""
    if not _anthropic_client:
        return "Error: Anthropic Claude API not configured or unavailable."
    if not ANTHROPIC_AVAILABLE:
        return "Error: Anthropic library not available."
        
    try:
        print(f"Sending prompt to Anthropic API ({model_name}), len {len(prompt)} chars. Preview: {prompt[:200]}...")
        response = _anthropic_client.messages.create(
            model=model_name,
            max_tokens=4000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text if response.content else None
        if response_text is None:
            return "Error: Received empty response from Anthropic Claude API."
        print(f"Response received from Anthropic API ({model_name}), len {len(response_text)} chars. Preview: {response_text[:200]}...")
        return response_text
    except Exception as e:
        if ANTHROPIC_AVAILABLE and hasattr(anthropic, 'RateLimitError') and isinstance(e, anthropic_retry_exceptions):
            print(f"Anthropic API retryable error ({type(e).__name__}): {e}. Retrying as per backoff policy...")
            raise
        print(f"Error querying Anthropic Claude ({model_name}): {e}")
        traceback.print_exc()
        return f"Error: Unable to get response from Anthropic Claude ({model_name}). Details: {type(e).__name__}. Check server logs for more."

# --- Chat Logging ---
LOG_DIR = "experiment_chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

current_log_file_path_global = "" # Renamed to avoid conflict with function parameter

def get_log_filename(session_id_prefix="session"):
    safe_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in session_id_prefix)
    # Using datetime from the datetime module, explicitly
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return os.path.join(LOG_DIR, f"chat_log_{safe_prefix}_{timestamp}.jsonl")

def initialize_log_session(experiment_id, user_id, llm_model_name, custom_metadata=None):
    global current_log_file_path_global
    session_id_prefix = f"{experiment_id}_{user_id}_{llm_model_name}"
    log_file_path = get_log_filename(session_id_prefix)
    
    session_info = {
        "log_version": "1.2",
        "experiment_id": experiment_id,
        "user_id": user_id,
        "llm_model_name": llm_model_name,
        "session_start_time_utc": datetime.now(timezone.utc).isoformat(), # Explicit timezone
        "custom_metadata": custom_metadata or {},
        "library_availability": {
            "plotly": PLOTLY_AVAILABLE, "matplotlib": MATPLOTLIB_AVAILABLE, "seaborn": SEABORN_AVAILABLE,
        },
        "llm_api_availability": {
            "gemini": _gemini_configured, "openai": _openai_client is not None,
            "anthropic": _anthropic_client is not None, "available_models": get_available_llm_models()
        }
    }
    log_interaction(log_file_path, "session_start", session_info) # Pass log_file_path directly
    print(f"Initialized new log session: {log_file_path}")
    current_log_file_path_global = log_file_path
    return log_file_path

def log_interaction(log_file_path_param, event_type, event_data, llm_model_used=None): # Renamed parameter
    if not log_file_path_param: # Use the passed parameter
        # Fallback to global if not provided, though it should always be passed from app.py
        if current_log_file_path_global:
            print(f"Warning: log_file_path not explicitly passed to log_interaction for event {event_type}. Using global.")
            log_file_path_param = current_log_file_path_global
        else:
            print("Error: Log file path is not set. Cannot log interaction.")
            return

    payload = event_data.copy() if isinstance(event_data, dict) else {"message": str(event_data)}
    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), # Explicit timezone
        "event_type": event_type, "payload": payload
    }
    if llm_model_used: log_entry["llm_model_used"] = llm_model_used
    
    try:
        with open(log_file_path_param, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Error writing to log file '{log_file_path_param}': {e}")

# --- End Chat Logging ---

def generate_hess_markdown(context, template_path='template.md', output_path='hess_knowledge_base.md'):
    print("Generating markdown...")
    print(f"Updating markdown at: {output_path}")
    try:
        with open(template_path, 'r', encoding='utf-8') as file: # Added encoding
            template_content = file.read()
        template = Template(template_content)
        rendered_content = template.render(context)
        with open(output_path, 'w', encoding='utf-8') as file: # Added encoding
            file.write(rendered_content)
        print("Markdown file updated...")
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
    except Exception as e:
        print(f"Error generating HESS markdown: {e}")


def get_data_for_markdown(filename, metadata_path='Metadata_Systems.xlsx', capacity_path='Capacity_Tests.xlsx'):
    match = re.search(r"cleaned_(\d{4})_(\d{2})_System_ID_(\d+)", filename)
    if not match:
        print(f"Filename '{filename}' does not match expected pattern for metadata extraction.")
        return None # Return None if pattern doesn't match
        
    year, month_str, system_id_str = match.group(1), match.group(2), match.group(3)
    system_id = int(system_id_str)
    month_name = calendar.month_name[int(month_str)]
    print(f"Year: {year}, Month: {month_name}, System ID: {system_id}")

    try:
        cap_df = pd.read_excel(capacity_path)
        cap_df['Start Time'] = pd.to_datetime(cap_df['Start Time'])
        filtered_df = cap_df[cap_df['ID'] == system_id]
        capacity_test_date = filtered_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    except FileNotFoundError:
        print(f"Error: Capacity data file not found at {capacity_path}")
        return None # Or handle by setting capacity_test_date to empty list
    except Exception as e:
        print(f"Error loading Capacity Excel: {e}")
        capacity_test_date = [] # Default to empty list on error

    try:
        df_meta = pd.read_excel(metadata_path) # Renamed df to df_meta to avoid clash
        row_series = df_meta[df_meta['ID'] == system_id] # Use df_meta
        if row_series.empty:
            print(f"No data found for System ID {system_id} in metadata.")
            return None
        
        row = row_series.iloc[0]
        ts_storage_val = row.get("Date_storage_system_installation")
        ts_measure_val = row.get("Date_measurement_system_installation")

        # Ensure timestamps are valid numbers before int conversion
        dt_storage = datetime.fromtimestamp(int(ts_storage_val), timezone.utc) if pd.notna(ts_storage_val) else None
        dt_measurement = datetime.fromtimestamp(int(ts_measure_val), timezone.utc) if pd.notna(ts_measure_val) else None
        
        system_age = "N/A"
        if dt_storage:
            print("Storage system installation:", dt_storage.strftime('%Y-%m-%d'))
            dt_target = datetime(year=int(year), month=int(month_str), day=1, tzinfo=timezone.utc)
            time_difference = dt_target - dt_storage
            system_age = time_difference.days // 30
        else:
            print("Storage system installation date not available or invalid.")

        if dt_measurement: print("Measurement system installation:", dt_measurement.strftime('%Y-%m-%d'))
        else: print("Measurement system installation date not available or invalid.")

        context = {
            "system_id": system_id, "month": month_name, "year": year,
            "Capacity_nominal_in_Ah": row.get("Capacity_nominal_in_Ah"),
            "Voltage_nominal_in_V": row.get("Voltage_nominal_in_V"),
            "Energy_nominal_in_kWh": row.get("Energy_nominal_in_kWh"),
            "Energy_usable_datasheet_in_kWh": row.get("Energy_usable_datasheet_in_kWh"),
            "Cell_number_in_series": row.get("Cell_number_in_series"),
            "Cell_number_in_parallel": row.get("Cell_number_in_parallel"),
            "Cell_number": row.get("Cell_number"),
            "Inverter_nominal_power": row.get("Inverter_nominal_power"),
            "Manufacturer": row.get("Manufacturer"), "Chemistry": row.get("Chemistry"),
            "Chemistry_detail": row.get("Chemistry_detail"),
            "Date_storage_system_installation": dt_storage.strftime('%Y-%m-%d %H:%M:%S%z') if dt_storage else "N/A",
            "Date_measurement_system_installation": dt_measurement.strftime('%Y-%m-%d %H:%M:%S%z') if dt_measurement else "N/A",
            "System_age": system_age, "Capacity_test_dates": capacity_test_date,
        }
        return context
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None
    except Exception as e:
        print(f"Error processing metadata for markdown: {e}")
        traceback.print_exc()
        return None


def load_data(file_uploader_object):
    """
    Load the uploaded CSV file from a Streamlit file uploader object.
    Shows column names and first few rows.
    """
    original_path = None # Initialize
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_input_file:
            tmp_input_file.write(file_uploader_object.getbuffer())
            original_path = tmp_input_file.name
        print(f"\n\n\n------- Uploaded file saved to temp: {original_path}\n\n", flush=True)

        cleaned_filename = f"cleaned_{os.path.basename(file_uploader_object.name)}" # Use os.path.basename
        cleaned_path = os.path.join(tempfile.gettempdir(), cleaned_filename)
        print(f"Cleaned file will be at: {cleaned_path}")

        clean_csv(original_path, cleaned_path) # clean_csv needs to handle FileNotFoundError for original_path
        
        # Check if metadata and capacity test files exist before calling get_data_for_markdown
        # These paths should ideally be configurable or passed in
        metadata_excel_path = 'Metadata_Systems.xlsx'
        capacity_excel_path = 'Capacity_Tests.xlsx'

        if not os.path.exists(metadata_excel_path):
            print(f"Warning: Metadata Excel file '{metadata_excel_path}' not found. Markdown context will be limited.")
        if not os.path.exists(capacity_excel_path):
             print(f"Warning: Capacity Test Excel file '{capacity_excel_path}' not found. Markdown context will be limited.")

        data_for_md = get_data_for_markdown(cleaned_filename, metadata_excel_path, capacity_excel_path)
        if data_for_md:
            generate_hess_markdown(data_for_md) # generate_hess_markdown needs to handle missing template.md
        else:
            print("Warning: Could not generate data for HESS markdown. Knowledge base might be incomplete.")

        df = pd.read_csv(cleaned_path)
        if df.empty:
            print("Warning: Loaded CSV file is empty after cleaning.")
            return None, None, None
        columns = df.columns.tolist()
        preview = df.head(5).to_string(index=False)
        print(f"Data loaded successfully. Columns: {columns}")
        return df, columns, preview
        
    except FileNotFoundError as e: # More specific error for cleaning step if files are missing
        print(f"Error loading data: A required file was not found. Details: {e}")
        traceback.print_exc()
        return None, None, None
    except pd.errors.EmptyDataError:
        print("Error loading data: The file is empty.")
        return None, None, None
    except pd.errors.ParserError as e:
        print(f"Error loading data: Failed to parse the file. Check CSV format. Details: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred in load_data: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        if original_path and os.path.exists(original_path): # Clean up temp input file
            try:
                os.remove(original_path)
                print(f"Successfully removed temporary file: {original_path}")
            except OSError as e:
                print(f"Error removing temporary file {original_path}: {e}")
        # Cleaned path is also in temp, Streamlit might clean it or might need explicit management if many files are processed

def execute_generated_code(code_string, df):
    if not isinstance(df, pd.DataFrame):
        print("Error in execute_generated_code: Invalid DataFrame provided.")
        return False, json.dumps({"status": "Execution Failed", "error_type": "SetupError", "error_message": "Invalid DataFrame provided to executor."})
    if not isinstance(code_string, str) or not code_string.strip():
        print("Error in execute_generated_code: Empty or invalid code string provided.")
        return False, json.dumps({"status": "Execution Failed", "error_type": "SetupError", "error_message": "Empty or invalid code string for execution."})

    local_vars = {'df': df.copy(), 'pd': pd, 'np': np, 'json': json}
    global_vars = {'pd': pd, 'np': np, 'json': json} # Minimal globals

    if PLOTLY_AVAILABLE: local_vars.update({'px_module': px, 'pio_module': pio}); global_vars.update({'px_module': px, 'pio_module': pio})
    if MATPLOTLIB_AVAILABLE: local_vars['plt_module'] = plt; global_vars['plt_module'] = plt
    if SEABORN_AVAILABLE: local_vars['sns_module'] = sns; global_vars['sns_module'] = sns
    
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    output_str = None

    try:
        exec(code_string, global_vars, local_vars)
        output_str = redirected_output.getvalue().strip()
        if not output_str:
            print("Code execution produced no output.")
            return False, json.dumps({"status": "Execution Failed", "error_type": "OutputError", "error_message": "Generated code produced no output."})
        print(f"Code execution successful. Captured output (first 200 chars): {output_str[:200]}...")
        return True, output_str
    except Exception as e:
        error_type, error_value, tb_obj = sys.exc_info()
        tb_list = traceback.format_exception(error_type, error_value, tb_obj)
        relevant_tb_lines = [line.strip() for line in reversed(tb_list) if 'exec(code_string,' not in line][:5] # Simplified traceback focus
        error_details_snippet = "\n".join(reversed(relevant_tb_lines))
        error_message_dict = {"status": "Execution Failed", "error_type": error_type.__name__, "error_message": str(error_value), "traceback_snippet": error_details_snippet.splitlines()}
        error_json_output = json.dumps(error_message_dict, indent=2)
        print(f"Code execution failed!\n--- Failing Code ---:\n{code_string}\n------------\n--- Error ---:\n{error_json_output}\n------------")
        traceback.print_exception(error_type, error_value, tb_obj, file=sys.stderr) # Full server log
        return False, error_json_output
    finally:
        sys.stdout = old_stdout

def add_session_details_to_csv(participant, llm_model, total_experiment_time, user_feedback_data, current_log_file_param):
    planner_steps = {}
    successfully_executed_steps = {}

    if not os.path.exists(current_log_file_param):
        print(f"Error in add_session_details_to_csv: Log file not found at {current_log_file_param}")
        return

    with open(current_log_file_param, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if log_entry.get("event_type") != "csv_logs": continue

                payload = log_entry.get("payload", {})
                task_id_str = payload.get("task_id", "").lower() # Renamed to avoid clash
                task_number_str = ''.join(filter(str.isdigit, task_id_str)) # Renamed
                if not task_number_str: continue
                task_key = f"Task {task_number_str}"

                if payload.get("step") == "planning":
                    step_count = payload.get("step_count", 0)
                    if not isinstance(step_count, int): continue
                    planner_steps[task_key] = planner_steps.get(task_key, 0) + step_count
                elif payload.get("step") == "summarizing": # Changed from "executing_plan" to "summarizing" as per your logic's structure
                    # This logic now correctly assumes summarizing happens after all execution steps for a task are logged.
                    # And CSV log for "summarizing" contains "result_of_all_steps"
                    results = payload.get("result_of_all_steps", [])
                    if not isinstance(results, list) : continue # Ensure results is a list

                    # This task_key should already have planner_steps if planning happened.
                    # No need to re-check task_has_planning here if logic is sequential for a task in logs.
                    
                    total_executed_for_task = 0
                    successful_for_task = 0

                    # The 'results' in your CSV log under 'summarizing' appears to be a list of tuples/lists:
                    # (success_flag, output_content_str, None_or_other_data)
                    for item_result in results:
                        if isinstance(item_result, (list, tuple)) and len(item_result) >= 2:
                            status_flag = item_result[0]
                            response_str = item_result[1]
                            
                            # Only count as an "executed step" if it wasn't a "pass" or "direct_answer" type.
                            # This requires checking the nature of the step, which isn't directly in this log entry.
                            # For simplicity here, we assume all items in "result_of_all_steps" correspond to attempted executable steps.
                            # A more robust way would be to also log the action type with each step result.
                            
                            total_executed_for_task += 1 # Count each result item as an attempted step execution.
                            
                            if status_flag: # If the step itself reported success
                                try:
                                    # Check if the content of the successful step indicates an *internal* failure
                                    parsed_response = json.loads(response_str)
                                    if isinstance(parsed_response, dict) and \
                                       parsed_response.get("status", "").lower() in ["failed", "execution failed"]:
                                        pass # It was a failure despite status_flag = True
                                    else:
                                        successful_for_task += 1 # Truly successful
                                except json.JSONDecodeError:
                                    successful_for_task += 1 # Successful if status_flag is True and output not JSON or not failure JSON
                                except TypeError: # If response_str is not a string (e.g. None)
                                    if status_flag: # If primary status was true, but response is problematic
                                        successful_for_task +=1 # Count as success based on primary flag
                            # else: it was a failure (status_flag was False)
                        # else: item_result format is unexpected

                    successfully_executed_steps[task_key] = successfully_executed_steps.get(task_key, 0) + successful_for_task
                    # Note: planner_steps might be higher if some planned steps are non-code (e.g. direct_answer)
                    # This successfully_executed_steps counts code executions that succeeded.

            except json.JSONDecodeError: continue
            except Exception as e_log_parse:
                print(f"Unexpected error parsing log line for CSV summary: {e_log_parse} on line: {line[:100]}")
                continue
    
    fieldnames = [
        "Participant_no", "LLM_model", "Experiment_time_minutes",
        "Execution_time_task1_seconds", "Planner_steps_task1", "Successfully_executed_steps_task1",
        "User_feedback_task1", "User_feedback_description_task1",
        "Execution_time_task2_seconds", "Planner_steps_task2", "Successfully_executed_steps_task2",
        "User_feedback_task2", "User_feedback_description_task2",
        "Execution_time_task3_seconds", "Planner_steps_task3", "Successfully_executed_steps_task3",
        "User_feedback_task3", "User_feedback_description_task3"
    ]
    row_data = { # Renamed to avoid clash
        "Participant_no": participant, "LLM_model": llm_model,
        "Experiment_time_minutes": round(total_experiment_time, 2) if isinstance(total_experiment_time, (int,float)) else total_experiment_time,
    }
    for i in range(1, 4):
        task_key_fill = f"Task {i}" # Renamed
        feedback = user_feedback_data.get(task_key_fill, {})
        row_data[f"Execution_time_task{i}_seconds"] = feedback.get("time_taken", "")
        row_data[f"Planner_steps_task{i}"] = planner_steps.get(task_key_fill, 0) # Default to 0
        row_data[f"Successfully_executed_steps_task{i}"] = successfully_executed_steps.get(task_key_fill, 0) # Default to 0
        row_data[f"User_feedback_task{i}"] = feedback.get("feedback", "")
        row_data[f"User_feedback_description_task{i}"] = feedback.get("feedback_reason", "")

    output_csv_path = "experiment_log_summary.csv"
    file_exists = os.path.isfile(output_csv_path)
    try:
        with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(output_csv_path) == 0: # Check if file is empty too
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Session details successfully added to {output_csv_path}")
    except IOError as e:
        print(f"Error writing to CSV {output_csv_path}: {e}")
    except Exception as e_csv:
        print(f"Unexpected error during CSV writing: {e_csv}")