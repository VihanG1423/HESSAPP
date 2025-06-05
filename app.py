import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import datetime
import plotly.io as pio
import atexit
import json
import traceback

# --- Import Functions and Availability Flags from utils.py ---
try:
    from utils import (
        load_data, query_llm, execute_generated_code, get_available_llm_models, add_session_details_to_csv,
        PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE,
        initialize_log_session, log_interaction
    )
    print("Imported from utils: load_data, query_llm, execute_generated_code, plotting flags, logging.")
except ImportError as e:
    st.error(f"Failed to import functions/flags from `utils.py`. Ensure `utils.py` is correct and present. Error: {e}")
    st.stop()

# --- Import Prompts from prompting.py ---
try:
    from prompting import generate_planner_prompt, generate_executor_prompt, generate_summary_prompt
    print("Imported prompting functions.")
except ImportError as e:
     st.error(f"Failed to import Planner/Executor/Summarizer prompts from `prompting.py`. Error: {e}")
     st.stop()
except Exception as e:
     st.error(f"An error occurred during importing from prompting.py: {e}")
     st.stop()

# Conditionally import plotly.express for the dashboard tab
if PLOTLY_AVAILABLE:
    import plotly.express as px
    print("Plotly available for dashboard rendering.")
else:
    px = None
    print("Plotly not available. Dashboard plots will be skipped.")

# --- Constants & Configuration ---
TEMP_WARNING_THRESHOLD_C = 40.0
TEMP_OPTIMAL_LOW_C = 15.0
TEMP_OPTIMAL_HIGH_C = 35.0
MAX_POINTS_TO_PLOT_DASHBOARD = 50000
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
LOG_DIR = "experiment_chat_logs"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="HESS AI Experiment", page_icon="🔋", layout="wide")
st.title("🔋 HESS AI Analysis Experiment")
st.caption("Evaluate AI agent performance on battery analysis tasks")

# --- Enhanced Session State Initialization ---
defaults = {
    # Experiment Setup
    "experiment_phase": "setup",  # setup, ready, task_active, completed
    "participant_index": 1,
    "experiment_id": "hess_eval_2025",
    "user_id": "",
    "llm_model_select": DEFAULT_LLM_MODEL,
    "setup_complete": False,
    
    # Data & File Processing
    "data_frame": None, "columns": None, "preview": None, "file_processed": False,
    "file_uploader_key": 0, "data_source_name": "N/A",
    "min_timestamp_str": None, "max_timestamp_str": None, "data_loading_error": None,

    # Conversation & Analysis
    "conversation_history": [], "current_step": 'initial', "last_user_question": None,
    "analysis_plan": None, "current_plan_step_index": 0,
    "step_results": [],
    "last_plot_metadata": None, 
    "current_ai_plot_data": None,
    "last_analysis_summary": None,
    "last_message_id": 0,

    # Logging
    "current_log_file": None,

    # Enhanced Task Management
    "tasks": { 
        "task1": {
            "name": "Load System Specifications",
            "desc": "Ask the AI to load and display your battery system specifications", 
            "hint": "Try: 'What are my system specs?' or 'Show me my battery specifications'", 
            "question": "Did the AI successfully load your system specifications?", 
            "completed": False, 
            "satisfaction_rating": None, 
            "satisfaction_explanation": "",
            "user_prompt_used": "",
            "analysis_time_seconds": 0
        },
        "task2": {
            "name": "Check Battery Health",
            "desc": "Ask the AI to assess your battery's health and condition", 
            "hint": "Try: 'Check my battery health' or 'How is my battery performing?'", 
            "question": "Did the AI provide useful insights about your battery health?", 
            "completed": False, 
            "satisfaction_rating": None, 
            "satisfaction_explanation": "",
            "user_prompt_used": "",
            "analysis_time_seconds": 0
        },
        "task3": {
            "name": "Analyze Energy Usage",
            "desc": "Ask the AI to analyze your energy consumption patterns", 
            "hint": "Try: 'Analyze my energy usage' or 'Show me my energy patterns'", 
            "question": "Did the AI provide clear insights about your energy usage?", 
            "completed": False, 
            "satisfaction_rating": None, 
            "satisfaction_explanation": "",
            "user_prompt_used": "",
            "analysis_time_seconds": 0
        },
    },
    "task_keys_ordered": ["task1", "task2", "task3"],
    "current_task_index": 0, 
    "task_active": False, 
    "task_start_time": 0.0,
    "experiment_start_time": None,
    "experiment_end_time": None,
}

# Initialize session state
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Helper Functions ---
def add_message(role, content, msg_id=None, step_index=None):
    if not content and not isinstance(content, bool): 
        print(f"Skipping adding message due to empty content: Role={role}")
        return None
    if msg_id is None:
        st.session_state.last_message_id += 1
        msg_id = st.session_state.last_message_id

    display_content = str(content) if not isinstance(content, str) else content
    new_message = { "role": role, "content": display_content, "id": msg_id }
    if step_index is not None: new_message["step_index"] = step_index
    st.session_state.conversation_history.append(new_message)
    print(f"Added message: ID={msg_id}, Role={role}, Step={step_index}, Content='{display_content[:100]}...'")

    if st.session_state.current_log_file and role not in ["user", "assistant", "code_preview"]:
         log_interaction(st.session_state.current_log_file, "internal_message_added",
                         {"role": role, "content_preview": display_content[:200]})
    return msg_id

def create_datetime_column(df):
    required_cols = ['Year', 'Month', 'Date', 'ClockTime']
    if df is None or df.empty: return df, None, None, "Input DataFrame is empty."
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: return df, None, None, f"Missing required columns: {', '.join(missing_cols)}."

    min_ts_str, max_ts_str = None, None
    df_processed = df.copy()
    error_message = None
    try:
        df_processed['ClockTimeStr'] = df_processed['ClockTime'].astype(str)
        def format_time(time_str):
            parts = str(time_str).split(':')
            try:
                h = int(parts[0]) if len(parts) > 0 else 0
                m = int(parts[1]) if len(parts) > 1 else 0
                s_part = parts[2].split('.')[0] if len(parts) > 2 else '0'
                s = int(s_part) if s_part else 0
                if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59): raise ValueError("Invalid time component")
                return f"{h:02d}:{m:02d}:{s:02d}"
            except: return '00:00:00'
        df_processed['ClockTimeStr'] = df_processed['ClockTimeStr'].apply(format_time)
        datetime_str_series = (df_processed['Year'].astype(str) + '-' +
                               df_processed['Month'].astype(str).str.zfill(2) + '-' +
                               df_processed['Date'].astype(str).str.zfill(2) + ' ' +
                               df_processed['ClockTimeStr'])
        df_processed['Timestamp'] = pd.to_datetime(datetime_str_series, errors='coerce')
        df_processed = df_processed.drop(columns=['ClockTimeStr'], errors='ignore')

        num_failed_ts = df_processed['Timestamp'].isnull().sum()
        if num_failed_ts == len(df_processed):
            error_message = "Critical: Failed to create any valid timestamps. Check source data format."
            return df, None, None, error_message
        elif num_failed_ts > 0:
            warning_msg = f"{num_failed_ts} timestamps could not be parsed (NaT). These rows may affect time-based analysis."
            error_message = warning_msg 

        valid_timestamps = df_processed['Timestamp'].dropna()
        if not valid_timestamps.empty:
            min_ts_str = valid_timestamps.min().strftime('%Y-%m-%d %H:%M:%S')
            max_ts_str = valid_timestamps.max().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Timestamp range: {min_ts_str} to {max_ts_str}")
        elif not error_message: 
             error_message = "No valid timestamps found after processing."

    except Exception as e:
        error_message = f"Error creating timestamp column: {e}"
        traceback.print_exc()
        return df, None, None, error_message
    return df_processed, min_ts_str, max_ts_str, error_message

def reset_experiment():
    """Reset experiment to initial state"""
    for key_to_reset in defaults: 
        if key_to_reset in st.session_state: 
            st.session_state[key_to_reset] = defaults[key_to_reset]
    st.session_state.file_uploader_key += 1
    st.rerun()

def complete_current_task():
    """Complete the current task and move to next"""
    current_task_key = st.session_state.task_keys_ordered[st.session_state.current_task_index]
    task_end_time = time.monotonic()
    
    # Calculate task duration
    task_duration = task_end_time - st.session_state.task_start_time
    st.session_state.tasks[current_task_key]['analysis_time_seconds'] = round(task_duration, 2)
    st.session_state.tasks[current_task_key]['completed'] = True
    
    # Log task completion
    if st.session_state.current_log_file:
        task_data = {
            "event_subtype": "task_complete",
            "task_id": current_task_key,
            "task_name": st.session_state.tasks[current_task_key]['name'],
            "duration_seconds": task_duration,
            "satisfaction_rating": st.session_state.tasks[current_task_key]['satisfaction_rating'],
            "satisfaction_explanation": st.session_state.tasks[current_task_key]['satisfaction_explanation'],
            "user_prompt_used": st.session_state.tasks[current_task_key]['user_prompt_used'],
            "participant_index": st.session_state.participant_index
        }
        log_interaction(st.session_state.current_log_file, "experiment_task_event", task_data)
    
    # Move to next task or complete experiment
    st.session_state.current_task_index += 1
    st.session_state.task_active = False
    
    if st.session_state.current_task_index >= len(st.session_state.task_keys_ordered):
        st.session_state.experiment_phase = "completed"
        st.session_state.experiment_end_time = datetime.datetime.now()
        if st.session_state.current_log_file:
            log_interaction(st.session_state.current_log_file, "experiment_event", {
                "event_subtype": "experiment_complete",
                "participant_index": st.session_state.participant_index,
                "total_duration_minutes": (st.session_state.experiment_end_time - st.session_state.experiment_start_time).total_seconds() / 60
            })
    
    # Clear conversation for next task
    st.session_state.conversation_history = []
    st.session_state.current_ai_plot_data = None
    st.session_state.current_step = 'idle'

# --- Sidebar: Experiment Flow ---
with st.sidebar:
    st.header("🧪 Experiment Setup")
    
    # Phase 1: Participant Setup
    if st.session_state.experiment_phase == "setup":
        st.subheader("Step 1: Participant Information")
        
        # Auto-generate next participant index
        participant_index = st.number_input(
            "Participant Number", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.participant_index,
            help="Unique participant number for this experiment session"
        )
        st.session_state.participant_index = participant_index
        
        # User ID based on participant index
        default_user_id = f"participant_{participant_index:02d}"
        user_id = st.text_input(
            "Participant ID", 
            value=default_user_id,
            help="Unique identifier for this participant"
        )
        st.session_state.user_id = user_id
        
        # LLM Model Selection
        available_models = get_available_llm_models()
        if available_models:
            default_index = 0
            if DEFAULT_LLM_MODEL in available_models:
                default_index = available_models.index(DEFAULT_LLM_MODEL)
            
            st.session_state.llm_model_select = st.selectbox(
                "AI Model",
                options=available_models,
                index=default_index,
                help="Select the AI model for this experiment session"
            )
        else:
            st.error("❌ No AI models available. Check API configuration.")
            st.stop()
        
        st.divider()
        
        # File Upload Section
        st.subheader("Step 2: Upload HESS Data")
        uploaded_file = st.file_uploader(
            "Upload your HESS data file (CSV)", 
            type=["csv"],
            key=f"file_uploader_{st.session_state.file_uploader_key}",
            help="Upload a CSV file containing HESS battery system data"
        )

        if uploaded_file is not None and not st.session_state.file_processed:
            with st.spinner("Processing data..."):
                # Initialize logging
                st.session_state.current_log_file = initialize_log_session(
                    st.session_state.experiment_id, 
                    st.session_state.user_id,
                    st.session_state.llm_model_select,
                    custom_metadata={
                        "action": "file_upload", 
                        "filename": uploaded_file.name,
                        "participant_index": st.session_state.participant_index
                    }
                )
                
                log_interaction(st.session_state.current_log_file, "system_event", {
                    "event": "file_upload_start", 
                    "filename": uploaded_file.name,
                    "participant_index": st.session_state.participant_index
                })

                # Reset data-related state
                data_keys = ["data_frame", "columns", "preview", "min_timestamp_str", "max_timestamp_str", 
                           "conversation_history", "last_plot_metadata", "current_ai_plot_data", 
                           "analysis_plan", "step_results", "last_analysis_summary", "data_loading_error"]
                for key in data_keys:
                    st.session_state[key] = defaults[key]
                st.session_state.current_plan_step_index = 0

                # Load and process data
                df_loaded, columns, preview = load_data(uploaded_file)
                if df_loaded is not None and columns is not None:
                    df_processed, min_ts, max_ts, ts_error = create_datetime_column(df_loaded)
                    st.session_state.data_loading_error = ts_error 

                    st.session_state.data_frame = df_processed
                    st.session_state.columns = df_processed.columns.tolist() if df_processed is not None else columns
                    st.session_state.preview = preview
                    st.session_state.min_timestamp_str = min_ts
                    st.session_state.max_timestamp_str = max_ts
                    st.session_state.file_processed = True
                    st.session_state.current_step = 'idle'
                    st.session_state.data_source_name = uploaded_file.name

                    st.success(f"✅ Data loaded: {uploaded_file.name}")
                    
                    # Log successful upload
                    log_data = {
                        "event": "file_upload_success", 
                        "filename": uploaded_file.name,
                        "rows": len(df_processed) if df_processed is not None else 0,
                        "cols": len(st.session_state.columns) if st.session_state.columns else 0,
                        "timestamp_status": "success" if min_ts and max_ts and not ts_error else "failed_or_partial",
                        "timestamp_error": ts_error,
                        "participant_index": st.session_state.participant_index
                    }
                    
                    if ts_error and "Critical" not in ts_error: 
                        st.warning(f"⚠️ {ts_error}") 
                    elif ts_error and "Critical" in ts_error: 
                        st.error(f"❌ {ts_error}") 

                    log_interaction(st.session_state.current_log_file, "system_event", log_data)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to load data. Check file format.")
                    log_interaction(st.session_state.current_log_file, "system_error", {
                        "type": "file_load_error", 
                        "filename": uploaded_file.name,
                        "participant_index": st.session_state.participant_index
                    })

        # Setup completion check
        setup_ready = (
            st.session_state.user_id and 
            st.session_state.llm_model_select and 
            st.session_state.file_processed
        )
        
        if setup_ready:
            st.divider()
            if st.button("🚀 Start Experiment", type="primary", use_container_width=True):
                st.session_state.experiment_phase = "ready"
                st.session_state.experiment_start_time = datetime.datetime.now()
                if st.session_state.current_log_file:
                    log_interaction(st.session_state.current_log_file, "experiment_event", {
                        "event_subtype": "experiment_start",
                        "participant_index": st.session_state.participant_index,
                        "total_tasks": len(st.session_state.task_keys_ordered)
                    })
                st.rerun()
        else:
            missing_items = []
            if not st.session_state.user_id: missing_items.append("Participant ID")
            if not st.session_state.llm_model_select: missing_items.append("AI Model")
            if not st.session_state.file_processed: missing_items.append("Data Upload")
            
            st.info(f"Complete setup to continue:\n• {chr(10).join(['• ' + item for item in missing_items])}")

    # Phase 2: Task Execution
    elif st.session_state.experiment_phase == "ready":
        st.success("✅ Setup Complete")
        st.caption(f"Participant: {st.session_state.user_id}")
        st.caption(f"AI Model: {st.session_state.llm_model_select}")
        st.caption(f"Data: {st.session_state.data_source_name}")
        
        st.divider()
        st.subheader("📋 Experiment Tasks")
        
        # Progress indicator
        completed_tasks = sum(1 for task in st.session_state.tasks.values() if task['completed'])
        total_tasks = len(st.session_state.tasks)
        progress = completed_tasks / total_tasks
        st.progress(progress, text=f"Progress: {completed_tasks}/{total_tasks} tasks completed")
        
        # Current task display
        if st.session_state.current_task_index < len(st.session_state.task_keys_ordered):
            current_task_key = st.session_state.task_keys_ordered[st.session_state.current_task_index]
            current_task = st.session_state.tasks[current_task_key]
            
            if not st.session_state.task_active:
                # Show next task to start
                st.markdown(f"**Task {st.session_state.current_task_index + 1}: {current_task['name']}**")
                st.write(current_task['desc'])
                st.info(f"💡 {current_task['hint']}")
                
                if st.button(f"▶️ Start Task {st.session_state.current_task_index + 1}", type="primary", use_container_width=True):
                    st.session_state.task_active = True
                    st.session_state.task_start_time = time.monotonic()
                    if st.session_state.current_log_file:
                        log_interaction(st.session_state.current_log_file, "experiment_task_event", {
                            "event_subtype": "task_begin", 
                            "task_id": current_task_key,
                            "task_name": current_task['name'],
                            "participant_index": st.session_state.participant_index
                        })
                    st.rerun()
            else:
                # Show active task info
                st.markdown(f"**Active: Task {st.session_state.current_task_index + 1}**")
                st.write(f"🔄 {current_task['name']}")
                
                # Task satisfaction survey
                st.subheader("📝 Task Feedback")
                
                # Store the user's prompt
                if st.session_state.last_user_question:
                    current_task['user_prompt_used'] = st.session_state.last_user_question
                
                # Satisfaction rating
                satisfaction_options = ["Select...", "Yes - Satisfied", "No - Not Satisfied", "Partially - Mixed Results"]
                satisfaction_rating = st.selectbox(
                    current_task['question'],
                    options=satisfaction_options,
                    key=f"satisfaction_{current_task_key}"
                )
                
                if satisfaction_rating != "Select...":
                    current_task['satisfaction_rating'] = satisfaction_rating
                    
                    # Optional explanation (especially for "No" or "Partially")
                    explanation_label = "Please explain why:" if "No" in satisfaction_rating or "Partially" in satisfaction_rating else "Any additional comments (optional):"
                    explanation = st.text_area(
                        explanation_label,
                        key=f"explanation_{current_task_key}",
                        height=100
                    )
                    current_task['satisfaction_explanation'] = explanation
                    
                    # Complete task button
                    if st.button("✅ Complete Task & Continue", type="primary", use_container_width=True):
                        complete_current_task()
                        st.rerun()
                else:
                    st.info("Please rate your satisfaction to continue")
        
        else:
            # All tasks completed
            st.session_state.experiment_phase = "completed"
    
    # Phase 3: Experiment Completed
    elif st.session_state.experiment_phase == "completed":
        st.success("🎉 Experiment Complete!")
        st.balloons()
        
        # Show completion summary
        total_time = (st.session_state.experiment_end_time - st.session_state.experiment_start_time).total_seconds() / 60
        st.metric("Total Time", f"{total_time:.1f} minutes")

        user_feedback_data = {}
        
        st.subheader("📊 Summary")
        for i, task_key in enumerate(st.session_state.task_keys_ordered):
            task = st.session_state.tasks[task_key]
            task_num = i + 1

            rating = str(task['satisfaction_rating']).lower()
            feedback_reason = str(task['satisfaction_explanation']).lower()
    
            if "yes" in rating:
                feedback = "yes"
            elif "no" in rating:
                feedback = "no"
            else:
                feedback = "mixed"

            user_feedback_data[f"Task {task_num}"] = {
                "feedback": feedback,
                "feedback_reason": feedback_reason,
                "time_taken": round(task['analysis_time_seconds'], 1)
            }

            satisfaction_emoji = "✅" if "Yes" in str(task['satisfaction_rating']) else "❌" if "No" in str(task['satisfaction_rating']) else "⚠️"
            st.write(f"{satisfaction_emoji} **Task {task_num}**: {task['name']} ({task['analysis_time_seconds']:.1f}s)")
        
        st.divider()
        st.write("Thank you for participating in the HESS AI evaluation experiment!")

        log_interaction(st.session_state.current_log_file, "Experiment_Successful",
                         {"Total_Time": total_time, "User_Feedback": user_feedback_data})
        
        if not st.session_state.get("csv_written", False):
            add_session_details_to_csv(
                st.session_state.participant_index,
                st.session_state.llm_model_select,
                total_time,
                user_feedback_data,
                st.session_state.current_log_file
            )
            st.session_state.csv_written = True # ✅ Mark as written
        
        
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.csv_written = False  # ✅ Reset for next session
            reset_experiment()
            

    # Reset option (always available)
    st.divider()
    if st.button("🔄 Reset Experiment", help="Start over with new participant"):
        st.session_state.csv_written = False  # ✅ Reset for next session
        reset_experiment()
        

# --- Main Content Area ---
if st.session_state.experiment_phase == "setup":
    st.info("👈 Please complete the setup in the sidebar to begin the experiment")
    
    # Show setup instructions
    st.markdown("""
    ## Welcome to the HESS AI Experiment
    
    This experiment evaluates AI performance on battery energy storage system analysis tasks.
    
    ### Instructions:
    1. **Enter your participant information** in the sidebar
    2. **Upload a HESS data file** (CSV format)
    3. **Start the experiment** to begin the three tasks
    4. **Complete each task** by interacting with the AI agent
    5. **Provide feedback** on the AI's performance
    
    ### The Three Tasks:
    - **Task 1**: Load system specifications
    - **Task 2**: Check battery health  
    - **Task 3**: Analyze energy usage
    
    Each task involves asking the AI agent questions and evaluating its responses.
    """)

elif st.session_state.experiment_phase in ["ready", "completed"]:
    
    # Show dashboard and chat interface
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Agent Chat", "🏆 LLM Competition"])

    with tab1:
        st.header("📊 System Dashboard")
        dashboard_df = st.session_state.data_frame
        if dashboard_df is None or dashboard_df.empty:
            st.warning("No data loaded or data is empty for dashboard.")
        elif 'Timestamp' not in dashboard_df.columns or not pd.api.types.is_datetime64_any_dtype(dashboard_df['Timestamp']) or dashboard_df['Timestamp'].isnull().all():
             st.warning("Valid 'Timestamp' column not found. Dashboard plots require it.")
             if st.session_state.data_loading_error: st.error(f"Timestamp creation issue: {st.session_state.data_loading_error}")
        else:
            min_ts_dash, max_ts_dash = st.session_state.min_timestamp_str, st.session_state.max_timestamp_str 
            st.info(f"Displaying data from **{st.session_state.data_source_name}** (Range: {min_ts_dash or 'N/A'} to {max_ts_dash or 'N/A'})")
            if st.session_state.data_loading_error: st.warning(f"Note on data quality: {st.session_state.data_loading_error}")
            
            with st.spinner("Preparing dashboard plots..."):
                try:
                    plot_df_full_dash = dashboard_df.dropna(subset=['Timestamp'])
                    if plot_df_full_dash.empty: 
                        st.warning("No data with valid timestamps for dashboard plotting.")
                    else:
                        num_points_dash = len(plot_df_full_dash)
                        plot_df_display_dash = plot_df_full_dash.sample(n=min(num_points_dash, MAX_POINTS_TO_PLOT_DASHBOARD), random_state=42).sort_values(by='Timestamp') if num_points_dash > MAX_POINTS_TO_PLOT_DASHBOARD else plot_df_full_dash
                        if num_points_dash > MAX_POINTS_TO_PLOT_DASHBOARD: 
                            st.info(f"Dashboard plots sampled to {len(plot_df_display_dash)} points.")
                        
                        if px: 
                            plot_configs_dash = {
                                "Voltage (V)": {"cols": ["voltage_volts"]},
                                "Current (A)": {"cols": ["current_amperes"]},
                                "Temperatures (°C)": {"cols": ["temp_battery_celsius", "temp_room_celsius"], "hline": TEMP_WARNING_THRESHOLD_C, "hrect": [TEMP_OPTIMAL_LOW_C, TEMP_OPTIMAL_HIGH_C]},
                                "Power (W)": {"cols": ["power_watts"]}
                            }
                            for title_dash, config_dash in plot_configs_dash.items():
                                cols_to_plot_dash = [col for col in config_dash["cols"] if col in plot_df_display_dash.columns and pd.api.types.is_numeric_dtype(plot_df_display_dash[col])]
                                if cols_to_plot_dash:
                                    try:
                                        fig_dash = px.line(plot_df_display_dash, x='Timestamp', y=cols_to_plot_dash, title=title_dash)
                                        if config_dash.get("hline") is not None: 
                                            fig_dash.add_hline(y=config_dash["hline"], line_dash="dash", line_color="red", annotation_text="Warning")
                                        if config_dash.get("hrect") is not None: 
                                            fig_dash.add_hrect(y0=config_dash["hrect"][0], y1=config_dash["hrect"][1], fillcolor="green", opacity=0.1, annotation_text="Optimal")
                                        st.plotly_chart(fig_dash, use_container_width=True)
                                    except Exception as e_dash: 
                                        st.error(f"Error plotting '{title_dash}': {e_dash}")
                                else: 
                                    st.warning(f"Skipping dashboard plot '{title_dash}': Required numeric column(s) not found.")
                        else: 
                            st.warning("Plotly (px) not available for dashboard plots.")
                        
                        st.divider()
                        st.dataframe(dashboard_df.head())
                except Exception as e_dash_main: 
                    st.error(f"Error generating dashboard: {e_dash_main}\n{traceback.format_exc()}")

    with tab2:
        st.header("💬 AI Agent Chat")
        
        # Task status indicator
        if st.session_state.task_active:
            current_task_key = st.session_state.task_keys_ordered[st.session_state.current_task_index]
            current_task = st.session_state.tasks[current_task_key]
            st.info(f"🎯 Active Task: {current_task['name']}")
        
        status_placeholder = st.empty()
        plot_placeholder = st.empty() 
        chat_container = st.container(height=800)

        # Display plot from previous analysis (if any and in idle state)
        if st.session_state.current_step == 'idle' and st.session_state.current_ai_plot_data:
            plot_info_idle = st.session_state.current_ai_plot_data
            plot_type_idle = plot_info_idle.get("type")
            plot_data_content_idle = plot_info_idle.get("data")
            
            with plot_placeholder.container():
                if plot_type_idle == "plotly" and plot_data_content_idle and PLOTLY_AVAILABLE and pio:
                    try:
                        fig_idle_show = pio.from_json(plot_data_content_idle)
                        st.plotly_chart(fig_idle_show, use_container_width=True)
                    except Exception as e_idle_plotly:
                        st.error(f"Error re-displaying Plotly chart: {e_idle_plotly}")
                elif plot_type_idle in ["matplotlib", "seaborn"] and plot_data_content_idle and (MATPLOTLIB_AVAILABLE or SEABORN_AVAILABLE):
                    try:
                        st.image(f"data:image/png;base64,{plot_data_content_idle}")
                    except Exception as e_idle_img:
                        st.error(f"Error re-displaying {plot_type_idle} image: {e_idle_img}")

        # Chat history display
        with chat_container:
                        
            for msg_idx, message in enumerate(st.session_state.conversation_history):
                role, content = message.get("role"), message.get("content", "")
                msg_id, step_idx_assoc = message.get("id"), message.get("step_index", -1)
                if not content and not isinstance(content, bool): continue
                
                if role == "user":
                    with st.chat_message("user"): 
                        st.markdown(content)
                elif role == "assistant":
                    with st.chat_message("assistant", avatar="🤖"): 
                        st.markdown(content, unsafe_allow_html=True)
                elif role == "code_preview":
                    with st.chat_message("assistant", avatar="💻"):
                        st.markdown(f"**Plan Step {step_idx_assoc + 1}:** I propose running this Python code:")
                        st.code(content, language="python")
                        
                elif role == "planner_step":
                    with st.chat_message("assistant", avatar="📝"):
                        try:
                            step_data = json.loads(content)
                            st.markdown(f"**Planning Step {step_data.get('step_id', '?')}:** {step_data.get('description', 'N/A')}")
                        except json.JSONDecodeError: 
                            st.markdown(f"**Planning:** {content}")

        

        # State machine for AI processing
        current_step_state = st.session_state.current_step
        status_text = ""
        if current_step_state == 'planning': status_text = "📝 Planning..."
        elif current_step_state == 'executing_plan': status_text = f"⚙️ Executing Plan Step {st.session_state.current_plan_step_index + 1}/{len(st.session_state.analysis_plan or [])}..."
        elif current_step_state == 'summarizing': status_text = "✍️ Summarizing..."

        if status_text and current_step_state in ['planning', 'executing_plan', 'summarizing']:
            with st.spinner(status_text): 
                # Planning phase
                if current_step_state == 'planning':
                    st.session_state.current_ai_plot_data = None
                    plot_placeholder.empty()
                    
                    st.session_state.update({"analysis_plan": None, "current_plan_step_index": 0, "step_results": []})
                    if st.session_state.columns and st.session_state.last_user_question:
                        planner_prompt_text_plan = generate_planner_prompt(
                            st.session_state.columns, st.session_state.min_timestamp_str,
                            st.session_state.max_timestamp_str, st.session_state.data_source_name,
                            st.session_state.last_analysis_summary 
                        )

                        
                        
                        data_loading_context_plan = ""
                        if st.session_state.data_loading_error:
                            data_loading_context_plan = f"IMPORTANT DATA LOADING CONTEXT (Timestamp issues: {st.session_state.data_loading_error}). This may limit time-based analysis or require data cleaning steps.\n\n"
                        full_planner_prompt_plan = data_loading_context_plan + planner_prompt_text_plan + st.session_state.last_user_question

                        llm_plan = st.session_state.llm_model_select
                        planner_response_raw_plan = query_llm(full_planner_prompt_plan, llm_plan)

             

                        step_count = planner_response_raw_plan.count('"step_id"')
                        
                        
                        if st.session_state.current_log_file: 
                            log_interaction(st.session_state.current_log_file, "llm_call_planner", {
                                "prompt_len": len(full_planner_prompt_plan), 
                                "response_preview": planner_response_raw_plan[:200],
                                "raw_response_truncated": planner_response_raw_plan[:1000],
                                "participant_index": st.session_state.participant_index,
                                "planner_steps": step_count
                            }, llm_plan)

                            log_interaction(st.session_state.current_log_file, "csv_logs", {
                                "task_id": current_task_key, 
                                "step": "planning",
                                "step_count": step_count,
                                "llm_used": llm_plan,
                                "participant_index": st.session_state.participant_index
                                })


                        
                        try:
                            cleaned_response_plan = planner_response_raw_plan.strip().lstrip('```json').rstrip('```').strip()
                            parsed_plan_plan = json.loads(cleaned_response_plan)
                            if isinstance(parsed_plan_plan, dict) and 'action' in parsed_plan_plan:
                                 # Claude returned a single step object, wrap it in a list
                                parsed_plan_plan = [parsed_plan_plan]
                            elif isinstance(parsed_plan_plan, dict) and 'analysis_plan' in parsed_plan_plan:
                                parsed_plan_plan = parsed_plan_plan['analysis_plan']
                            if not isinstance(parsed_plan_plan, list) or not parsed_plan_plan or not all(isinstance(s, dict) and 'action' in s for s in parsed_plan_plan):
                                raise ValueError("Plan not a valid list of steps or empty.")
                            st.session_state.analysis_plan = parsed_plan_plan
                            first_action_plan = parsed_plan_plan[0].get("action")
                            if first_action_plan in ["pass", "direct_answer", "interpret_summary"]: 
                                output_marker_plan = "PASS" if first_action_plan == "pass" else \
                                                f"DIRECT_ANSWER_REQUEST: {parsed_plan_plan[0].get('description', st.session_state.last_user_question)}" if first_action_plan == "direct_answer" else \
                                                "INTERPRET_SUMMARY_REQUEST"
                                st.session_state.step_results.append((True, output_marker_plan, None))
                                st.session_state.current_step = 'summarizing'
                            else: 
                                st.session_state.current_step = 'executing_plan'
                                add_message("system", f"Plan: {len(parsed_plan_plan)} steps.")
                                for i_plan, step_item_plan in enumerate(parsed_plan_plan): 
                                    add_message("planner_step", json.dumps(step_item_plan), step_index=i_plan, msg_id=f"plan_{i_plan}")
                        except Exception as e_plan:
                            add_message("assistant", f"Sorry, plan creation failed: {e_plan}. Raw response preview: {planner_response_raw_plan[:200]}...")
                            if st.session_state.current_log_file: 
                                log_interaction(st.session_state.current_log_file, "error_planner", {"error": str(e_plan), "raw_response": planner_response_raw_plan}, llm_plan)
                            st.session_state.current_step = 'idle'
                        st.rerun()
                    else: 
                        st.error("Missing data columns or user question for planning.")
                        st.session_state.current_step = 'idle'
                        st.rerun()

                # Execution phase
                elif current_step_state == 'executing_plan':
                    plan_exec = st.session_state.analysis_plan
                    idx_exec = st.session_state.current_plan_step_index
                    if not plan_exec or idx_exec >= len(plan_exec): 
                        st.session_state.current_step = 'summarizing'
                        st.rerun()
                    else:
                        step_details_exec = plan_exec[idx_exec]
                        action_exec, step_id_exec = step_details_exec.get("action"), step_details_exec.get("step_id", idx_exec + 1)
                        
                        if action_exec in ["pass", "direct_answer", "interpret_summary"]: 
                            output_marker_exec_skip = "PASS" if action_exec == "pass" else \
                                             f"DIRECT_ANSWER_REQUEST: {step_details_exec.get('description', '')}" if action_exec == "direct_answer" else \
                                             "INTERPRET_SUMMARY_REQUEST"
                            st.session_state.step_results.append((True, output_marker_exec_skip, None))
                            st.session_state.current_plan_step_index += 1
                            st.rerun()
                        
                        
                        else: # Generate code
                            executor_prompt_text_gen = generate_executor_prompt(
                                step_details_exec, st.session_state.columns, st.session_state.preview,
                                st.session_state.min_timestamp_str, st.session_state.max_timestamp_str
                            )
                            llm_exec_gen = st.session_state.llm_model_select
                            executor_response_raw_gen = query_llm(executor_prompt_text_gen, llm_exec_gen)
                            
                            if st.session_state.current_log_file: 
                                log_interaction(st.session_state.current_log_file, "llm_call_executor_codegen", {
                                    "step_index": idx_exec,
                                    "step_id":step_id_exec, 
                                    "action":action_exec, 
                                    "response_preview":executor_response_raw_gen[:200],
                                    "participant_index": st.session_state.participant_index
                                }, llm_exec_gen)

                            code_match_gen = re.search(r"```python\n(.*?)```", executor_response_raw_gen, re.DOTALL | re.IGNORECASE)
                            extracted_code_gen = code_match_gen.group(1).strip() if code_match_gen else ""
                            
                            unsafe_keywords_found_gen = [kw for kw in ['os.', 'sys.', 'subprocess.', 'eval(', 'exec(', 'open(', '__file__'] if kw in extracted_code_gen]


                            if not unsafe_keywords_found_gen and extracted_code_gen:
                                add_message("code_preview", extracted_code_gen, step_index=idx_exec)

                                if st.session_state.current_log_file:
                                    log_interaction(st.session_state.current_log_file, "code_generation_success", {
                                        "step_index": idx_exec,
                                        "step_id": step_id_exec,
                                        "action": action_exec,
                                        "full_code_hash": hash(extracted_code_gen),
                                        "participant_index": st.session_state.participant_index
                                    })

                                                                        

                                # Immediately execute the code
                                code_to_run_exec = extracted_code_gen
                                df_for_exec_run = st.session_state.data_frame
                                success_run, result_output_str_run = False, json.dumps({
                                    "status": "Execution Failed",
                                    "error_type": "PreExecutionError",
                                    "message": "Code or DataFrame missing."
                                })

                                if code_to_run_exec and df_for_exec_run is not None:
                                    success_run, result_output_str_run = execute_generated_code(code_to_run_exec, df_for_exec_run)

                                if st.session_state.current_log_file:
                                    log_interaction(st.session_state.current_log_file, "code_execution_result", {
                                        "step_index": idx_exec,
                                        "step_id": step_id_exec,
                                        "action": action_exec,
                                        "success": success_run,
                                        "full_output_or_error_truncated": result_output_str_run,
                                        "participant_index": st.session_state.participant_index
                                    })

                                    log_interaction(st.session_state.current_log_file, "csv_logs", {
                                        "task_id": current_task_key, 
                                        "step": "executing_plan",
                                        "step_id": step_id_exec,
                                        "step_action": action_exec, 
                                        "step_result": success_run, 
                                        "step_output": result_output_str_run,
                                        "llm_used": st.session_state.llm_model_select,
                                        "participant_index": st.session_state.participant_index
                                        })


                                # If it's a plot step, store the plot info
                                if success_run and action_exec == "plot_data":
                                    try:
                                        plot_output_dict_exec = json.loads(result_output_str_run)
                                        plot_metadata_exec = plot_output_dict_exec.get("metadata")
                                        if isinstance(plot_metadata_exec, dict):
                                            st.session_state.last_plot_metadata = plot_metadata_exec
                                            library_exec = plot_metadata_exec.get("library")
                                            if "plot_json" in plot_output_dict_exec and library_exec == "plotly":
                                                st.session_state.current_ai_plot_data = {
                                                    "type": "plotly",
                                                    "data": plot_output_dict_exec["plot_json"],
                                                    "metadata": plot_metadata_exec
                                                }
                                            elif "plot_base64" in plot_output_dict_exec and library_exec in ["matplotlib", "seaborn"]:
                                                st.session_state.current_ai_plot_data = {
                                                    "type": library_exec,
                                                    "data": plot_output_dict_exec["plot_base64"],
                                                    "metadata": plot_metadata_exec
                                                }
                                    except Exception as e_plot_store:
                                        success_run = False
                                        result_output_str_run = json.dumps({
                                            "status": "Failed",
                                            "message": f"Error processing plot data for step {step_id_exec}: {e_plot_store}"
                                        })

                                st.session_state.step_results.append((success_run, result_output_str_run, None))
                                st.session_state.current_plan_step_index += 1
                            else:
                                reason_gen = f"Unsafe keywords found: {unsafe_keywords_found_gen}" if unsafe_keywords_found_gen else "No code extracted"
                                err_msg_gen = f"Code generation failed for step {step_id_exec}: {reason_gen}."
                                st.session_state.step_results.append((False, json.dumps({
                                    "status": "Failed",
                                    "error_type": "CodeGenError",
                                    "message": err_msg_gen
                                }), None))

                                if st.session_state.current_log_file:
                                    log_interaction(st.session_state.current_log_file, "code_generation_failed", {
                                        "step_index": idx_exec,
                                        "step_id": step_id_exec,
                                        "reason": reason_gen,
                                        "raw_response": executor_response_raw_gen,
                                        "participant_index": st.session_state.participant_index
                                    })

                                st.session_state.current_plan_step_index += 1

                            st.rerun()


                           

                # Summarization phase
                elif current_step_state == 'summarizing':
                    user_q_sum = st.session_state.last_user_question
                    plan_sum_final = st.session_state.analysis_plan
                    results_sum_final = st.session_state.step_results
                    last_sum_ctx_final = st.session_state.last_analysis_summary
                    

                    log_interaction(st.session_state.current_log_file, "csv_logs", {
                    "task_id": current_task_key, 
                    "step": "summarizing",
                    "result_of_all_steps": results_sum_final,
                    "llm_used": st.session_state.llm_model_select,
                    "participant_index": st.session_state.participant_index
                    })


                    llm_sum_final = st.session_state.llm_model_select

                    if user_q_sum is None or plan_sum_final is None: 
                        add_message("assistant", "Sorry, critical context lost for summarization.")
                        if st.session_state.current_log_file: 
                            log_interaction(st.session_state.current_log_file, "error_summarizer", {"error_type": "missing_context"})
                        st.session_state.last_analysis_summary = None 
                    else:
                        plan_len_sum_final = len(plan_sum_final) if isinstance(plan_sum_final, list) else 0
                        while len(results_sum_final) < plan_len_sum_final: 
                             results_sum_final.append((False, json.dumps({"status":"Skipped", "message":"Step not executed."}), None))
                        # In app.py summarization section:
                        summary_prompt_text_final = generate_summary_prompt(user_q_sum, plan_sum_final, results_sum_final, last_sum_ctx_final, model_name=llm_sum_final)
                        
                        final_response_text = query_llm(summary_prompt_text_final, llm_sum_final)
                        
                        if st.session_state.current_log_file: 
                            log_interaction(st.session_state.current_log_file, "llm_call_summarizer", {
                                "summary_preview": final_response_text[:200],
                                "participant_index": st.session_state.participant_index
                            }, llm_sum_final)

                        if final_response_text.startswith("Error:"):
                            add_message("assistant", f"Sorry, error during summary: {final_response_text}")
                            st.session_state.last_analysis_summary = None 
                        else:
                            add_message("assistant", final_response_text)
                            st.session_state.last_analysis_summary = final_response_text
                    
                    st.session_state.last_user_question = None
                    st.session_state.analysis_plan = None
                    st.session_state.current_plan_step_index = 0
                    st.session_state.step_results = []
                    st.session_state.current_step = 'idle' 
                    st.rerun()
    

        # Chat input
        chat_disabled = (st.session_state.current_step != 'idle') or (not st.session_state.file_processed) or (st.session_state.experiment_phase != "ready")
        placeholder_text = "Processing..." if st.session_state.current_step != 'idle' else \
                          "Complete setup first..." if not st.session_state.file_processed else \
                          "Ask the AI agent about your battery data..." if st.session_state.task_active else \
                          "Start a task to begin chatting..."
        
        if user_input := st.chat_input(placeholder_text, key="agent_input", disabled=chat_disabled):
            if st.session_state.current_step == 'idle' and st.session_state.file_processed and st.session_state.task_active:
                st.session_state.last_user_question = user_input
                add_message("user", user_input)
                if st.session_state.current_log_file: 
                    log_interaction(st.session_state.current_log_file, "user_message_sent", {
                        "content": user_input,
                        "participant_index": st.session_state.participant_index,
                        "active_task": st.session_state.task_keys_ordered[st.session_state.current_task_index] if st.session_state.task_active else None
                    })
                st.session_state.current_step = 'planning' 
                st.rerun()
    with tab3:
        st.header("🏆 Which AI Model Should You Use?")
        st.caption("Simple analysis of experiment results")
        
        # Load experiment data
        try:
            if os.path.exists("experiment_log_summary.csv"):
                df_results = pd.read_csv("experiment_log_summary.csv")
                
                if df_results.empty:
                    st.info("📊 No experiment data available yet. Complete some tasks to see results!")
                else:
                    
                    st.success(f"📊 Found data from {len(df_results)} experiments using {df_results['LLM_model'].nunique()} different AI models")
                    
                    # Simple metrics calculation
                    results = []
                    
                    for model in df_results['LLM_model'].unique():
                        model_data = df_results[df_results['LLM_model'] == model]
                        
                        # Count satisfied users across all tasks
                        total_satisfied = 0
                        total_responses = 0
                        for task in [1, 2, 3]:
                            feedback_col = f'User_feedback_task{task}'
                            if feedback_col in model_data.columns:
                                responses = model_data[feedback_col].dropna()
                                satisfied = responses.str.lower().str.contains('yes', na=False).sum()
                                total_satisfied += satisfied
                                total_responses += len(responses)
                        
                        satisfaction_rate = (total_satisfied / total_responses * 100) if total_responses > 0 else 0
                        
                        # Average task time
                        all_times = []
                        for task in [1, 2, 3]:
                            time_col = f'Execution_time_task{task}_seconds'
                            if time_col in model_data.columns:
                                times = model_data[time_col].dropna()
                                all_times.extend(times.tolist())
                        avg_time = np.mean(all_times) if all_times else 0
                        
                        # Success rate (working code)
                        total_planned = 0
                        total_successful = 0
                        for task in [1, 2, 3]:
                            planned_col = f'Planner_steps_task{task}'
                            success_col = f'Successfully_executed_steps_task{task}'
                            if planned_col in model_data.columns and success_col in model_data.columns:
                                planned = model_data[planned_col].fillna(0).sum()
                                successful = model_data[success_col].fillna(0).sum()
                                total_planned += planned
                                total_successful += successful
                        
                        success_rate = (total_successful / total_planned * 100) if total_planned > 0 else 0
                        
                        results.append({
                            'AI Model': model,
                            'Experiments': len(model_data),
                            'User Satisfaction': f"{satisfaction_rate:.1f}%",
                            'Avg Speed': f"{avg_time:.1f}s",
                            'Code Success': f"{success_rate:.1f}%",
                            'Satisfaction_Num': satisfaction_rate,
                            'Speed_Num': avg_time,
                            'Success_Num': success_rate
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    if len(results_df) > 0:
                        # Key findings
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            best_satisfaction = results_df.loc[results_df['Satisfaction_Num'].idxmax()]
                            st.metric(
                                "😊 Most Liked by Users", 
                                best_satisfaction['AI Model'],
                                f"{best_satisfaction['User Satisfaction']} satisfied"
                            )
                        
                        with col2:
                            fastest = results_df.loc[results_df['Speed_Num'].idxmin()]
                            st.metric(
                                "⚡ Fastest Responses", 
                                fastest['AI Model'],
                                f"{fastest['Avg Speed']} average"
                            )
                        
                        with col3:
                            most_reliable = results_df.loc[results_df['Success_Num'].idxmax()]
                            st.metric(
                                "🎯 Most Reliable Code", 
                                most_reliable['AI Model'],
                                f"{most_reliable['Code Success']} working"
                            )
                        
                        st.divider()
                        
                        # Simple comparison table
                        st.subheader("📊 Full Comparison")
                        st.caption("Code Success = % of planned code steps that executed without errors")
                        display_df = results_df[['AI Model', 'Experiments', 'User Satisfaction', 'Avg Speed', 'Code Success']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Task breakdown - User Satisfaction
                        st.subheader("📋 User Satisfaction by Task Type")
                        st.caption("% of users who said 'Yes' for each task type")
                        
                        task_names = ["Load Battery Specs", "Check Battery Health", "Analyze Energy Usage"]
                        
                        task_results = []
                        for model in df_results['LLM_model'].unique():
                            model_data = df_results[df_results['LLM_model'] == model]
                            row = {'AI Model': model}
                            
                            for i, task_name in enumerate(task_names, 1):
                                feedback_col = f'User_feedback_task{i}'
                                if feedback_col in model_data.columns:
                                    responses = model_data[feedback_col].dropna()
                                    if len(responses) > 0:
                                        satisfied = responses.str.lower().str.contains('yes', na=False).sum()
                                        satisfaction = round((satisfied / len(responses)) * 100, 1)
                                        row[task_name] = f"{satisfaction}%"
                                    else:
                                        row[task_name] = "No data"
                                else:
                                    row[task_name] = "No data"
                            
                            task_results.append(row)
                        
                        task_df = pd.DataFrame(task_results)
                        st.dataframe(task_df, use_container_width=True, hide_index=True)
                        st.caption("Shows % of users satisfied with AI performance for each specific task")
                        
                        # Task breakdown - Code Success
                        st.subheader("🎯 Code Success Rate by Task Type")
                        st.caption("% of planned analysis steps that executed successfully for each task")
                        
                        code_success_results = []
                        for model in df_results['LLM_model'].unique():
                            model_data = df_results[df_results['LLM_model'] == model]
                            row = {'AI Model': model}
                            
                            for i, task_name in enumerate(task_names, 1):
                                planned_col = f'Planner_steps_task{i}'
                                success_col = f'Successfully_executed_steps_task{i}'
                                
                                if planned_col in model_data.columns and success_col in model_data.columns:
                                    total_planned = model_data[planned_col].fillna(0).sum()
                                    total_successful = model_data[success_col].fillna(0).sum()
                                    
                                    if total_planned > 0:
                                        success_rate = round((total_successful / total_planned) * 100, 1)
                                        row[task_name] = f"{success_rate}%"
                                    else:
                                        row[task_name] = "No data"
                                else:
                                    row[task_name] = "No data"
                            
                            code_success_results.append(row)
                        
                        code_success_df = pd.DataFrame(code_success_results)
                        st.dataframe(code_success_df, use_container_width=True, hide_index=True)
                        st.caption("Shows % of analysis steps that worked without technical errors for each task")
                        
                        # Simple recommendation
                        st.subheader("💡 Bottom Line")
                        
                        # Overall winner (best satisfaction + good reliability)
                        results_df['Overall_Score'] = results_df['Satisfaction_Num'] + (results_df['Success_Num'] * 0.5)
                        winner = results_df.loc[results_df['Overall_Score'].idxmax()]
                        
                        st.success(f"🏆 **Recommended AI Model: {winner['AI Model']}**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Why this model:**")
                            st.write(f"• {winner['User Satisfaction']} user satisfaction")
                            st.write(f"• {winner['Code Success']} code reliability")
                            st.write(f"• {winner['Avg Speed']} response time")
                        
                        with col2:
                            st.write("**Summary:**")
                            if winner['Satisfaction_Num'] >= 70:
                                st.write("✅ Users find responses helpful")
                            else:
                                st.write("⚠️ Could improve user satisfaction")
                            
                            if winner['Success_Num'] >= 70:
                                st.write("✅ Code works reliably")
                            else:
                                st.write("⚠️ Code reliability needs work")
                            
                            if winner['Speed_Num'] <= 60:
                                st.write("✅ Responds quickly")
                            else:
                                st.write("⚠️ Takes time to respond")
                        
                        # Show data source and realism note
                        st.divider()
                        col1 = st.columns(2)
                        with col1:
                            st.caption(f"Analysis based on {len(df_results)} experiments across {df_results['LLM_model'].nunique()} AI models")

                        
                        # Expected performance based on real-world usage
                        with st.expander("📝 Expected Real-World Performance"):
                            st.write("**Based on typical usage patterns:**")
                            st.write("• **Claude**: Best for complex analysis, excellent context handling, very reliable, moderate speed")
                            st.write("• **GPT-4**: Good balance of capabilities, slower but thorough")  
                            st.write("• **Gemini**: Fast responses, good for simple tasks, may struggle with complex analysis")
                            st.caption("Actual results depend on your specific use case and data complexity")
                    
                    else:
                        st.warning("No valid data found for analysis")
                        
            else:
                st.info("📊 No experiment data file found.")
                st.write("Expected file: `experiment_log_summary.csv`")
                st.write("Complete some experiments to see results here!")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            if st.checkbox("Show error details"):
                st.code(str(e))

elif st.session_state.current_step == 'initial':
    st.info("👋 Welcome! Please complete the experiment setup in the sidebar to get started.")
    st.caption("Ensure required files are present and configure participant information in the sidebar for logging.")