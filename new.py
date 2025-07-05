import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="Gait Authentication System",
    page_icon="🚶",
    layout="wide"
)

# Initialize session state
if 'auth_history' not in st.session_state:
    st.session_state.auth_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your Gait Authentication Assistant. How can I help you today?"}
    ]

# Mock 1D-CNN model (replace with your actual model)
class GaitModel:
    def __init__(self):
        self.classes = [f"employee_{i:02d}" for i in range(1, 31)]
        # Simulate some known gait patterns
        self.known_patterns = {
            "employee_01": {"confidence": 0.97, "access_level": "Admin"},
            "employee_02": {"confidence": 0.95, "access_level": "Research"},
            "employee_03": {"confidence": 0.93, "access_level": "Operations"}
        }
        
    def predict(self, data):
        # Simulate model prediction with more realistic behavior
        time.sleep(0.25)  # Simulate processing time
        
        # 90% chance of matching a known pattern
        if np.random.random() > 0.1:
            employee = np.random.choice(list(self.known_patterns.keys()))
            confidence = np.random.normal(0.95, 0.03)
            confidence = max(0.7, min(0.99, confidence))
            return employee, confidence
        else:
            # Unknown pattern
            return "unknown", np.random.uniform(0.4, 0.69)

# Query Gemma3 model
def query_gemma(prompt, context=None):
    base_prompt = """
    You are an AI assistant for a gait authentication system that identifies employees based on 
    their walking patterns using smartphone sensors and a 1D-CNN model. The system has 30 
    registered employees and operates with high accuracy.
    """
    
    if context:
        base_prompt += f"\n\nAdditional context: {context}"
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'gemma:3b',
                'prompt': f"{base_prompt}\n\nUser question: {prompt}",
                'stream': False,
                'options': {'temperature': 0.3}
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['response']
        return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to AI model: {str(e)}"

# App title and description
st.title("🚶 Gait Authentication System")
st.markdown("""
Real-time employee authentication using smartphone inertial data and gait analysis.
All processing completes in <1 second for friction-free access.
""")

# Sidebar for system controls
with st.sidebar:
    st.header("⚙️ System Controls")
    detection_threshold = st.slider("Confidence Threshold", 0.7, 0.99, 0.9, 0.01)
    system_active = st.toggle("Active Monitoring", True, help="Enable/disable the authentication system")
    
    st.divider()
    st.header("📊 System Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Employees", 30)
        st.metric("Avg Auth Time", "0.25s")
    with col2:
        success_rate = len([x for x in st.session_state.auth_history if x['status'] == 'granted']) / max(1, len(st.session_state.auth_history))
        st.metric("Success Rate", f"{success_rate:.1%}")
        st.metric("Today's Auths", len(st.session_state.auth_history))
    
    st.divider()
    if st.button("🛑 Emergency Lockdown", type="primary"):
        system_active = False
        st.session_state.auth_history.append({
            "timestamp": datetime.now(),
            "employee": "SYSTEM",
            "confidence": 1.0,
            "status": "lockdown",
            "details": "System entered lockdown mode"
        })
        st.rerun()

# Main app tabs
tab1, tab2 = st.tabs(["📡 Live Monitoring", "📊 Data Explorer"])

with tab1:
    st.header("Real-time Authentication Dashboard")
    
    if system_active:
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔄 Live Sensor Feed")
            
            # Placeholder for sensor visualization
            sensor_placeholder = st.empty()
            
            # Generate mock sensor data when button is clicked
            if st.button("Simulate Employee Approach", type="primary"):
                with st.spinner("Analyzing gait pattern..."):
                    # Generate more realistic sensor data
                    num_samples = 100
                    time_vec = np.linspace(0, 2*np.pi, num_samples)
                    
                    # Simulate walking pattern
                    accel_x = 0.5 * np.sin(2*time_vec) + np.random.normal(0, 0.1, num_samples)
                    accel_y = 0.3 * np.cos(time_vec) + np.random.normal(0, 0.1, num_samples)
                    accel_z = 0.1 * np.sin(3*time_vec) + np.random.normal(0.1, 0.05, num_samples)
                    gyro_x = 0.2 * np.sin(time_vec + 0.5) + np.random.normal(0, 0.05, num_samples)
                    gyro_y = 0.1 * np.cos(2*time_vec) + np.random.normal(0, 0.05, num_samples)
                    gyro_z = 0.05 * np.sin(time_vec) + np.random.normal(0, 0.02, num_samples)
                    
                    # Combine into feature vector
                    sample_data = np.column_stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
                    
                    # Make prediction
                    model = GaitModel()
                    employee, confidence = model.predict(sample_data)
                    
                    # Create sensor visualization
                    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                    
                    # Accelerometer plot
                    ax[0].plot(accel_x, label='X', color='#FF4B4B')
                    ax[0].plot(accel_y, label='Y', color='#0068C9')
                    ax[0].plot(accel_z, label='Z', color='#00C897')
                    ax[0].set_title('Accelerometer Data (m/s²)')
                    ax[0].legend()
                    ax[0].grid(True, alpha=0.3)
                    
                    # Gyroscope plot
                    ax[1].plot(gyro_x, label='X', color='#FF4B4B')
                    ax[1].plot(gyro_y, label='Y', color='#0068C9')
                    ax[1].plot(gyro_z, label='Z', color='#00C897')
                    ax[1].set_title('Gyroscope Data (rad/s)')
                    ax[1].legend()
                    ax[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    sensor_placeholder.pyplot(fig)
                    
                    # Display authentication result
                    result_placeholder = st.empty()
                    if confidence > detection_threshold and employee != "unknown":
                        result_placeholder.success(f"""
                        ✅ Access Granted  
                        **Employee:** {employee}  
                        **Confidence:** {confidence:.2%}  
                        **Access Level:** {model.known_patterns.get(employee, {}).get('access_level', 'Standard')}
                        """)
                        st.balloons()
                        
                        # Trigger door unlock (simulated)
                        time.sleep(0.3)
                        st.toast("Door unlocked", icon="🔓")
                        
                        st.session_state.auth_history.append({
                            "timestamp": datetime.now(),
                            "employee": employee,
                            "confidence": confidence,
                            "status": "granted",
                            "details": "Access granted - door unlocked"
                        })
                    else:
                        result_placeholder.error(f"""
                        ❌ Access Denied  
                        **Reason:** {'Low confidence' if confidence < detection_threshold else 'Unknown gait pattern'}  
                        **Confidence:** {confidence:.2%}
                        """)
                        
                        st.session_state.auth_history.append({
                            "timestamp": datetime.now(),
                            "employee": employee if employee != "unknown" else "unknown",
                            "confidence": confidence,
                            "status": "denied",
                            "details": "Access denied - unknown gait pattern" if employee == "unknown" else f"Access denied - confidence below threshold ({detection_threshold:.0%})"
                        })
        
        with col2:
            st.subheader("🔍 Recent Activity")
            
            if st.session_state.auth_history:
                # Show last 5 events in a nicer format
                for event in list(reversed(st.session_state.auth_history))[:5]:
                    timestamp = event["timestamp"].strftime("%H:%M:%S")
                    if event["status"] == "granted":
                        st.success(f"**{timestamp}** - {event['employee']} ({(event['confidence']):.0%})")
                    elif event["status"] == "denied":
                        st.error(f"**{timestamp}** - {event.get('employee', 'Unknown')} ({(event['confidence']):.0%})")
                    elif event["status"] == "lockdown":
                        st.warning(f"**{timestamp}** - SYSTEM LOCKDOWN")
                    
                    st.caption(event.get("details", ""))
                    st.divider()
            else:
                st.info("No authentication attempts yet")
                
            # Quick stats
            st.subheader("📈 Activity Summary")
            if st.session_state.auth_history:
                df = pd.DataFrame(st.session_state.auth_history)
                granted = df[df['status'] == 'granted'].shape[0]
                denied = df[df['status'] == 'denied'].shape[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Granted", granted)
                with col2:
                    st.metric("Denied", denied)
                
                # Hourly activity
                st.caption("Hourly Activity")
                df['hour'] = df['timestamp'].dt.hour
                hourly = df.groupby('hour').size()
                st.bar_chart(hourly)
            else:
                st.info("No data available yet")
    else:
        st.warning("⚠️ System monitoring is currently inactive")

with tab2:
    st.header("Gait Data Analysis")
    
    if st.session_state.auth_history:
        df = pd.DataFrame(st.session_state.auth_history)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add date and hour columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        # Create tabs for different views
        tab2_1, tab2_2, tab2_3 = st.tabs(["📅 Overview", "👤 Employee Stats", "📉 Performance"])
        
        with tab2_1:
            st.subheader("System Activity Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Authentication Results**")
                fig, ax = plt.subplots()
                df['status'].value_counts().plot(kind='bar', color=['#00C897', '#FF4B4B', '#FFC800'])
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                st.write("**Confidence Distribution**")
                fig, ax = plt.subplots()
                df['confidence'].plot(kind='hist', bins=20, color='#0068C9')
                ax.axvline(detection_threshold, color='#FF4B4B', linestyle='--', label='Threshold')
                plt.legend()
                st.pyplot(fig)
            
            st.write("**Timeline of Events**")
            timeline = df.set_index('timestamp').resample('1H').size()
            st.line_chart(timeline)
        
        with tab2_2:
            st.subheader("Employee Statistics")
            
            if 'employee' in df.columns:
                employee_stats = df[df['employee'] != 'unknown'].groupby('employee').agg({
                    'confidence': ['mean', 'count'],
                    'status': lambda x: (x == 'granted').mean()
                }).sort_values(('confidence', 'mean'), ascending=False)
                
                employee_stats.columns = ['Avg Confidence', 'Auth Attempts', 'Success Rate']
                st.dataframe(
                    employee_stats.style
                    .background_gradient(subset=['Avg Confidence'], cmap='YlGnBu')
                    .format({
                        'Avg Confidence': '{:.1%}',
                        'Success Rate': '{:.1%}'
                    })
                )
            else:
                st.info("No employee-specific data available")
        
        with tab2_3:
            st.subheader("System Performance")
            
            # Calculate rolling success rate
            df_sorted = df.sort_values('timestamp')
            df_sorted['success'] = (df_sorted['status'] == 'granted').astype(int)
            df_sorted['rolling_success'] = df_sorted['success'].rolling(10, min_periods=1).mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Success Rate Over Time**")
                st.line_chart(df_sorted.set_index('timestamp')['rolling_success'])
            
            with col2:
                st.write("**Recent Performance**")
                last_10 = df_sorted.tail(10)
                st.metric("Last 10 Success Rate", f"{(last_10['success'].mean()):.1%}")
            
            st.write("**Detailed Logs**")
            st.dataframe(df.sort_values('timestamp', ascending=False))
    else:
        st.info("No data available yet")

# AI Assistant Section (placed outside of tabs)
st.header("💬 Gait Authentication Assistant")
st.caption("Powered by Gemma3 - Ask questions about the system, employees, or authentication results")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input (now properly placed outside of any containers)
if prompt := st.chat_input("Ask about the gait authentication system..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create context from recent authentication history
            context = None
            if st.session_state.auth_history:
                recent_auths = list(reversed(st.session_state.auth_history))[:3]
                context = "Recent authentication attempts:\n"
                for auth in recent_auths:
                    context += f"- {auth['timestamp'].strftime('%H:%M')}: {auth['employee']} ({auth['status']}, confidence {auth['confidence']:.0%})\n"
            
            # Get response from Gemma3
            response = query_gemma(prompt, context)
            
            # Display response
            st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Add some custom CSS
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding: 2rem;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 3rem;
    }
    .stButton>button {
        border: 1px solid #0068C9;
    }
    .st-eb {
        background-color: #F0F2F6;
    }
    /* Add spacing between main content and chat */
    .st-emotion-cache-1avcm0n {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)