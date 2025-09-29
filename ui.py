import streamlit as st
import os
import sys
from pathlib import Path
import uuid
import base64
import io
import shutil
from datetime import datetime
import logging
# Import your existing components
from fileprocessor import DynamicFileProcessor, ProcessingLogger
from modified_main import setup_agent
from modified_main import read_file
# Set page configuration
st.set_page_config(
    page_title="CodeSmith - File Processing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
import dotenv
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv() 

class StreamlitFileProcessor:
    def __init__(self):
        self.processor = DynamicFileProcessor(logger)
        self.logger = ProcessingLogger(logger)
        self.agent = setup_agent()
        self.conversation = []
        
    def process_user_request(self, user_input: str):
        """Process user request using the AI agent"""
        # Add to conversation
        self.conversation.append({"role": "user", "content": user_input})
        
        # Log the request
        self.logger.add_log("USER_REQUEST", user_input)
        
        try:
            # Execute agent with the user input
            result = self.agent.invoke({
                "input": f"User request: {user_input}"
            })
            
            response = result.get("output", "No response generated")
            
            # Add to conversation
            self.conversation.append({"role": "assistant", "content": response})
            
            # Log the response
            self.logger.add_log("AGENT_RESPONSE", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.conversation.append({"role": "assistant", "content": error_msg})
            self.logger.add_log("ERROR", error_msg)
            return error_msg

def init_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = StreamlitFileProcessor()
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'logs' not in st.session_state:
        st.session_state.logs = []

def display_conversation():
    """Display the conversation in the left column"""
    st.subheader("💬 Conversation")
    
    for message in st.session_state.conversation[-10:]:  # Show last 10 messages
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

def display_processing_logs():
    """Display processing logs in the right column"""
    st.subheader("📊 Processing Logs")
    
    # Get recent logs
    recent_logs = st.session_state.processor.logger.logs[-20:]  # Show last 20 logs
    
    if not recent_logs:
        st.info("No processing logs yet. Start a conversation to see logs here.")
        return
    
    for log in recent_logs:
        timestamp = log['timestamp'].strftime('%H:%M:%S')
        
        # Color code based on log type
        if log['type'] in ['ERROR', 'FAILURE']:
            st.error(f"**[{timestamp}] {log['type']}**: {log['content']}")
        elif log['type'] in ['SUCCESS', 'AGENT_RESPONSE']:
            st.success(f"**[{timestamp}] {log['type']}**: {log['content']}")
        elif log['type'] == 'USER_REQUEST':
            st.info(f"**[{timestamp}] {log['type']}**: {log['content']}")
        else:
            st.write(f"**[{timestamp}] {log['type']}**: {log['content']}")

def handle_file_upload():
    """Handle file uploads"""
    st.sidebar.subheader("📁 File Management")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload files for processing",
        accept_multiple_files=True,
        help="Upload CSV, Excel, JSON, text files, etc."
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save file to input folder
            file_path = os.path.join("input_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.processor.logger.add_log(
                "FILE_UPLOAD", 
                f"Uploaded {uploaded_file.name}",
                {"filename": uploaded_file.name, "size": uploaded_file.size}
            )
            st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

def list_available_files():
    """List available files in sidebar"""
    st.sidebar.subheader("📂 Available Files")
    
    processor = DynamicFileProcessor(logger)
    files = processor.get_available_files()
    
    if not files:
        st.sidebar.info("No files in input folder. Upload files to get started.")
        return
    
    for file in files:
        file_path = os.path.join(processor.input_folder, file)
        file_size = os.path.getsize(file_path)
        file_type = Path(file).suffix
        
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"**{file}**")
        col2.write(f"`{file_type}`")
        
        # Show file actions
        with st.sidebar.expander(f"Actions for {file}"):
            if st.button(f"View {file}", key=f"view_{file}"):
                content = read_file(file)
                st.sidebar.text_area(f"Content of {file}", content, height=200)

def display_output_files():
    """Display output files in sidebar"""
    st.sidebar.subheader("📤 Output Files")
    
    processor = DynamicFileProcessor(logger)
    output_folder = processor.output_folder
    
    if os.path.exists(output_folder):
        output_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        
        if not output_files:
            st.sidebar.info("No output files yet. Process some files to see outputs here.")
            return
        
        for file in output_files:
            file_path = os.path.join(output_folder, file)
            file_size = os.path.getsize(file_path)
            
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"**{file}**")
            col2.write(f"`{file_size} bytes`")
            
            # Download button for output files
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            st.sidebar.download_button(
                label=f"Download {file}",
                data=file_data,
                file_name=file,
                mime="application/octet-stream",
                key=f"download_{file}"
            )

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🤖 CodeSmith - AI File Processing Agent")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # File management
        handle_file_upload()
        list_available_files()
        display_output_files()
        
        # System info
        st.markdown("---")
        st.subheader("ℹ️ System Info")
        st.write(f"Input Folder: `{os.path.abspath('input_files')}`")
        st.write(f"Output Folder: `{os.path.abspath('output_files')}`")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation = []
            st.session_state.processor.conversation = []
            st.rerun()
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # User input
        user_input = st.chat_input("Enter your file processing request...")
        
        if user_input:
            # Add user message to conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})
            
            # Process the request
            with st.spinner("🤔 Processing your request..."):
                response = st.session_state.processor.process_user_request(user_input)
            
            # Add assistant response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": response})
            
            # Rerun to update the display
            st.rerun()
        
        # Display conversation
        display_conversation()
    
    with col2:
        # Processing logs
        display_processing_logs()
        
        # Quick actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 List Files"):
                result = st.session_state.processor.process_user_request("List all available files")
                st.session_state.conversation.append({"role": "user", "content": "List all available files"})
                st.session_state.conversation.append({"role": "assistant", "content": result})
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear Logs"):
                st.session_state.processor.logger.logs = []
                st.rerun()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("input_files", exist_ok=True)
    os.makedirs("output_files", exist_ok=True)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found! Please set it in your environment variables.")
        st.stop()
    
    main()