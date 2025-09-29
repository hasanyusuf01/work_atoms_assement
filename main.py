import os
import re
import argparse
import sys
from pathlib import Path
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseOutputParser
from e2b_code_interpreter import Sandbox
import pandas as pd
import json
from typing import Dict, Any, List
import uuid
import logging
from datetime import datetime
import base64
import binascii
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (e.g., GOOGLE_API_KEY)


class DynamicFileProcessor:
    def __init__(self, input_folder: str = "input_files"):
        self.input_folder = input_folder
        self.output_folder = "output_folder"
        self.setup_folders()
        
    def setup_folders(self):
        """Create necessary folders"""
        os.makedirs(self.input_folder, exist_ok=True)
        
    def get_available_files(self) -> List[str]:
        """Get list of all files in input folder"""
        try:
            files = []
            for file in os.listdir(self.input_folder):
                if os.path.isfile(os.path.join(self.input_folder, file)):
                    files.append(file)
            return files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

class ProcessingLogger:
    def __init__(self):
        self.logs = []
    
    def add_log(self, log_type: str, content: str, metadata: Dict = None):
        """Add a processing log"""
        log_entry = {
            'id': str(uuid.uuid4())[:8],
            'type': log_type,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.logs.append(log_entry)
        logger.info(f"{log_type}: {content}")
    
    def display_logs(self, log_types: List[str] = None):
        """Display logs in CLI format"""
        if not self.logs:
            print("No logs available.")
            return
            
        filtered_logs = self.logs
        if log_types:
            filtered_logs = [log for log in self.logs if log['type'] in log_types]
        
        print("\n" + "="*80)
        print("PROCESSING LOGS")
        print("="*80)
        
        for log in filtered_logs:
            print(f"\n[{log['timestamp'].strftime('%H:%M:%S')}] {log['type']}:")
            print(f"  {log['content']}")
            if log['metadata']:
                print(f"  Metadata: {json.dumps(log['metadata'], indent=2)}")

import base64
import io
from PIL import Image
import json

@tool
def execute_and_download_code(code: str) -> str:
    """
    Execute Python code in sandbox and handle all types of E2B results including images, files, and outputs.
    """
    try:
        session_id = str(uuid.uuid4())[:8]
        print(f"Executing code in sandbox (Session: {session_id})...")
        
        processor = DynamicFileProcessor()
        files_before = set(processor.get_available_files())
        
        with Sandbox.create() as sandbox:
            # Upload all files as binary
            files = processor.get_available_files()
            for file in files:
                local_path = os.path.join(processor.input_folder, file)
                sandbox_path = f"/home/user/{file}"
                with open(local_path, "rb") as f:
                    sandbox.files.write(sandbox_path, f.read())
            
            print("Files uploaded to sandbox, executing code...")
            
            # Execute the code
            execution = sandbox.run_code(code)
            
            # Debug: Check what we got
            print(f"Execution results type: {type(execution.results)}")
            print(f"Number of results: {len(execution.results)}")
            
            # Process all types of results
            processed_files = []
            
            # 1. Handle execution.results (PIL images, base64, etc.)
            if execution.results:
                print(f"Processing {len(execution.results)} result(s)...")
                for i, result in enumerate(execution.results):
                    print(f"Result {i} type: {type(result)}")
                    print(f"Result {i} attributes: {dir(result)}")
                    
                    # Handle different result types
                    result_files = process_single_result(result, processor.input_folder, i)
                    processed_files.extend(result_files)
            
            # 2. Handle output files from the code (via ### OUTPUT_FILES marker)
            output_text = execution.text or ""
            if not output_text and execution.logs and execution.logs.stdout:
                output_text = "\n".join(execution.logs.stdout)
            
            print(f"Execution output: {output_text}")
            
            output_files = []
            if output_text:
                for line in output_text.split('\n'):
                    if '### OUTPUT_FILES:' in line:
                        files_part = line.split('### OUTPUT_FILES:')[1].strip()
                        output_files = [f.strip() for f in files_part.split(',')]
                        print(f"Output files marker found: {output_files}")
                        break
            
            # 3. Download files created in sandbox filesystem
            downloaded_files = []
            for filename in output_files:
                try:
                    sandbox_path = f"/home/user/{filename}"
                    local_path = os.path.join(processor.input_folder, filename)
                    
                    content = sandbox.files.read(sandbox_path)
                    
                    # Handle binary files properly
                    file_ext = Path(filename).suffix.lower()
                    is_binary = file_ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.pdf'}
                    
                    if is_binary:
                        if isinstance(content, str):
                            # Try base64 first, then fallback to raw bytes
                            try:
                                decoded = base64.b64decode(content)
                                content = decoded
                            except:
                                content = content.encode('latin-1')
                    
                    # Write file
                    with open(local_path, 'wb') as f:
                        if isinstance(content, str):
                            f.write(content.encode('latin-1'))
                        else:
                            f.write(content)
                    
                    downloaded_files.append(filename)
                    print(f"Downloaded: {filename}")
                    
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
            
            # Combine all processed files
            all_files = processed_files + downloaded_files
            
            # Build result
            result_parts = []
            result_parts.append(f"Execution completed (Session: {session_id})")
            
            if output_text:
                result_parts.append(f"Output: {output_text}")
            
            if execution.error:
                result_parts.append(f"Error: {execution.error}")
            
            if all_files:
                result_parts.append(f"Generated files: {', '.join(all_files)}")
            else:
                result_parts.append("No files generated")
            
            # Check for new local files
            files_after = set(processor.get_available_files())
            new_files = files_after - files_before
            if new_files:
                result_parts.append(f"New local files: {', '.join(new_files)}")
            
            return "\n".join(result_parts)
            
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        print(error_msg)
        return error_msg

def process_single_result(result, output_dir: str, index: int) -> List[str]:
    """
    Process a single E2B result and save any files it contains.
    Returns list of filenames created.
    """
    created_files = []
    
    try:
        # Handle PIL Image objects
        if hasattr(result, '__class__') and 'PIL.' in str(result.__class__):
            print(f"Result {index}: PIL Image detected")
            try:
                # Convert PIL Image to bytes and save
                img_buffer = io.BytesIO()
                result.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                
                filename = f"result_{index}.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                
                created_files.append(filename)
                print(f"Saved PIL Image as {filename}")
                
            except Exception as e:
                print(f"Failed to save PIL Image: {e}")
        
        # Handle base64 PNG
        elif hasattr(result, 'png') and result.png:
            print(f"Result {index}: Base64 PNG detected")
            try:
                filename = f"result_{index}.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(result.png))
                
                created_files.append(filename)
                print(f"Saved base64 PNG as {filename}")
            except Exception as e:
                print(f"Failed to save base64 PNG: {e}")
        
        # Handle base64 JPG
        elif hasattr(result, 'jpg') and result.jpg:
            print(f"Result {index}: Base64 JPG detected")
            try:
                filename = f"result_{index}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(result.jpg))
                
                created_files.append(filename)
                print(f"Saved base64 JPG as {filename}")
            except Exception as e:
                print(f"Failed to save base64 JPG: {e}")
        
        # Handle SVG
        elif hasattr(result, 'svg') and result.svg:
            print(f"Result {index}: SVG detected")
            try:
                filename = f"result_{index}.svg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(result.svg)
                
                created_files.append(filename)
                print(f"Saved SVG as {filename}")
            except Exception as e:
                print(f"Failed to save SVG: {e}")
        
        # Handle text results
        elif hasattr(result, 'text') and result.text:
            print(f"Result {index}: Text result detected")
            try:
                filename = f"result_{index}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(result.text)
                
                created_files.append(filename)
                print(f"Saved text as {filename}")
            except Exception as e:
                print(f"Failed to save text: {e}")
        
        # Handle generic object - try to serialize
        else:
            print(f"Result {index}: Generic object - {type(result)}")
            try:
                # Try to convert to string and save
                filename = f"result_{index}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(str(result))
                
                created_files.append(filename)
                print(f"Saved generic object as {filename}")
            except Exception as e:
                print(f"Failed to save generic object: {e}")
    
    except Exception as e:
        print(f"Error processing result {index}: {e}")
    
    return created_files



@tool
def read_file(filename: str) -> str:
    """
    Read and return the content of a file from the input folder.
    
    Args:
        filename: Name of the file to read
    """
    try:
        processor = DynamicFileProcessor()
        file_path = os.path.join(processor.input_folder, filename)
        
        if not os.path.exists(file_path):
            return f"File {filename} not found in input folder."
        
        # Read based on file type
        file_ext = Path(filename).suffix.lower()
        
        if file_ext in ['.txt', '.csv', '.json', '.xml', '.html', '.log']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return f"Content of {filename}:\n{content}"
        
        # elif file_ext in ['.xlsx', '.xls']:
        #     df = pd.read_excel(file_path)
        #     return f"Excel file {filename} content (first 100 rows):\n{df.head(100).to_string()}"
        
        else:
            return f"File {filename} is of type {file_ext}. Use execute_python_code for binary or complex file processing."
            
    except Exception as e:
        return f"Error reading file {filename}: {str(e)}"

@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the input folder.
    This works directly with the local file system.
    
    Args:
        filename: Name of the file to write (e.g., 'a.csv')
        content: Content to write to the file
    """
    try:
        processor = DynamicFileProcessor()
        file_path = os.path.join(processor.input_folder, filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote to {filename} (Size: {len(content)} bytes)"
        
    except Exception as e:
        return f"Error writing to file {filename}: {str(e)}"

@tool
def generate_python_code(task_description: str, file_info: str = "") -> str:
    """
    Generate Python code for file processing tasks using AI.
    This tool actually generates real, dynamic code based on the task.
    """
    try:
        # Use the same LLM as the agent to generate actual code
        llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)   
        
        code_generation_prompt = f"""
        You are an expert Python programmer. Generate Python code to solve this task:
        
        TASK: {task_description}
        
        FILE INFORMATION: {file_info}
        
        REQUIREMENTS:
        1. Generate COMPLETE, READY-TO-RUN Python code
        2. use the required libraries that are needed 
        3. Handle errors gracefully with try-except
        4. Print the final result clearly
        5. locally Files are located in './input_files/' directory

        6. for the generated code Files in sandbox are located at: '/home/user/'
        7. for the generated code Use ABSOLUTE paths: '/home/user/filename.csv'
        8. for the generated code Save output files to: '/home/user/output_filename.csv'
        6. Return ONLY the Python code, no explanations
        7. Make sure the code is self-contained and executable
        8.  If creating output files, save them in the CURRENT DIRECTORY (not in subfolders)
        9. After processing, print a special marker with the output files
        10. Use this exact format for output file tracking: 
           ### OUTPUT_FILES: filename1.csv, filename2.txt, etc.

        IMPORTANT: If the file content is provided as text, process it directly instead of reading from file.
        
        Generate the Python code now:

        """
        



        
        response = llm.invoke(code_generation_prompt)
        generated_code = response.content.strip()
        
        # Remove markdown code blocks if present
        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        if generated_code.startswith("```"):
            generated_code = generated_code[3:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
            
        return generated_code.strip()
        
    except Exception as e:
        return f"Error generating code: {str(e)}"

@tool
def list_files() -> str:
    """
    List all files available in the input folder for processing.
    Returns a list of filenames with their types.
    IMPORTANT: This tool takes no input parameters.
    """
    try:
        processor = DynamicFileProcessor()
        files = processor.get_available_files()
        if not files:
            return "No files found in input folder. Please upload files first."
        
        file_info = []
        for file in files:
            file_path = os.path.join(processor.input_folder, file)
            file_type = Path(file).suffix.lower() or "unknown"
            size = os.path.getsize(file_path)
            file_info.append(f"{file} (Type: {file_type}, Size: {size} bytes)")
        
        return "Available files:\n" + "\n".join(file_info)
    except Exception as e:
        return f"Error listing files: {str(e)}"
@tool
def download_files_from_sandbox(filenames: str) -> str:
    """
    Download specific files from the last sandbox execution.
    
    Args:
        filenames: Comma-separated list of filenames to download (e.g., "a.csv,output.txt")
    """
    try:
        processor = DynamicFileProcessor()
        files_to_download = [f.strip() for f in filenames.split(',')]
        downloaded_files = []
        
        # Note: This would need to track the last sandbox session
        # For now, we'll implement a simpler version that tries to download from current directory
        
        print(f"<======> Attempting to download files: {', '.join(files_to_download)}")
        
        # Since we can't access the previous sandbox, we'll create a new one and check for files
        # This is a limitation - in a production system you'd track sandbox sessions
        with Sandbox.create() as sandbox:
            for filename in files_to_download:
                try:
                    # Try to read the file from sandbox
                    content = sandbox.files.read(f"/home/user/{filename}")
                    local_path = os.path.join(processor.input_folder, filename)
                    
                    # Write to local system
                    with open(local_path, 'wb') as f:
                        f.write(content)
                    downloaded_files.append(filename)
                    print(f"\n <======<(^-^)>=====>Downloaded: {filename}")
                    
                except Exception as e:
                    print(f"\n <======<(^-^)>=====> Could not download {filename}: {e}")
                    continue
        
        if downloaded_files:
            return f"Successfully downloaded: {', '.join(downloaded_files)}"
        else:
            return "No files were downloaded. They may not exist in the sandbox."
            
    except Exception as e:
        return f"Error downloading files: {str(e)}"
@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a secure sandbox for file processing.
    Returns execution output and any files created during execution.
    """
    try:
        session_id = str(uuid.uuid4())[:8]
        print(f"Executing code in sandbox (Session: {session_id})...")
        
        processor = DynamicFileProcessor()
        
        with Sandbox.create() as sandbox:
            # Upload all input files to sandbox
            files = processor.get_available_files()
            for file in files:
                local_path = os.path.join(processor.input_folder, file)
                sandbox_path = f"/home/user/{file}"
                with open(local_path, "rb") as f:
                    sandbox.files.write(sandbox_path, f.read())
            
            print("\n <======<(^-^)>=====> Files uploaded to sandbox, executing code...")
            
            # Execute the code
            execution = sandbox.run_code(code)
            
            result = f"Execution completed (Session: {session_id})\n"
            result += f"Output:\n{execution.text if execution.text else 'No output captured'}\n"
            
            print(f"\n <======<(^-^)>=====> Code execution completed")
            return result
            
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        print(f"\n <======<(^-^)>=====> {error_msg}")
        return error_msg

def setup_agent():
    """Set up the LangChain agent with tools - FIXED: Better configuration"""
    
    # Define tools
    tools = [list_files, read_file, write_file,
            generate_python_code, 
            # execute_python_code,
            execute_and_download_code,
            download_files_from_sandbox]
    
    # System prompt for dynamic file processing
    system_prompt = """You are a dynamic file processing assistant. You can:
    1. List available files in the input folder using list_files tool
    2. Read and analyze file contents using read_file tool  
    4. Write modified files back using write_file tool
    5. When you need to generate code, ALWAYS use the generate_python_code tool
    6. The generate_python_code tool will create ACTUAL working Python code, not placeholders
    7. Always provide clear task_description and file_info to generate_python_code
    9. Return the final results to the user
    10. For execution WITH automatic file downloads: use execute_and_download_code (RECOMMENDED)
    11. For manual file downloads: use download_files_from_sandbox
        
        Always follow this process:
    1. First, use list_files tool to understand what files are available
    2. If needed, use read_file tool to understand file structure
    3. Generate appropriate Python code using generate_python_code tool for the requested task
    4. Execute the code using execute_and_download_code tool and return results
    
The execute_and_download_code tool automatically detects output files from code execution and downloads them.
    Be specific about which files you're processing and what operations you're performing.
    IMPORTANT: Always use the tools to interact with files, don't assume file contents.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)   
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # FIXED: Use better configuration for agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    return agent_executor

class CLIFileProcessor:

    def __init__(self):
        self.processor = DynamicFileProcessor()
        self.logger = ProcessingLogger()
        self.agent = setup_agent()
        self.conversation = []

    def test_agent_tools(self):
        """Test if agent tools are working properly"""
        print("\n\n <======<(^-^)>=====> Testing agent tools...")
        try:
            # Test the agent with a simple file listing request
            test_result = self.agent.invoke({
                "input": "Please list all available files using the list_files tool"
            })
            
            if test_result.get("output"):
                print("\n <======<(^-^)>=====> Agent tools are working correctly!")
                return True
            else:
                print("\n <======<(^-^)>=====> Agent tools test failed")
                return False
                
        except Exception as e:
            print(f"\n <======<(^-^)>=====> Agent tools test failed: {e}")
            return False
        
    def display_banner(self):
        """Display CLI banner"""
        print("\n" + "="*80)
        print("\n <======<(^-^)>=====> DYNAMIC FILE PROCESSOR - CLI VERSION")
        print("="*80)
        print("Commands:")
        print("  /help, /h      - Show this help message")
        print("  /files, /f     - List available files")
        print("  /upload, /u    - Upload files to input folder")
        print("  /logs, /l      - Show processing logs")
        print("  /code, /c      - Show latest code operations")
        print("  /clear         - Clear screen")
        print("  /exit, /quit   - Exit the application")
        print("  <your query>   - Process files with AI agent")
        print("="*80)
        
    def upload_files(self):
        """Handle file uploads via CLI"""
        print("\n\n <======<(^-^)>=====> File Upload")
        print("Drag and drop files into the 'input_files' folder or specify paths to copy.")
        
        while True:
            file_path = input("Enter file path (or 'done' to finish): ").strip()
            if file_path.lower() in ['done', '']:
                break
                
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.processor.input_folder, filename)
                
                try:
                    # Copy file to input folder
                    import shutil
                    shutil.copy2(file_path, dest_path)
                    self.logger.add_log(
                        "FILE_UPLOAD", 
                        f"Uploaded {filename}",
                        {"filename": filename, "size": os.path.getsize(dest_path)}
                    )
                    print(f"\n <======<(^-^)>=====> Uploaded: {filename}")
                except Exception as e:
                    print(f"\n <======<(^-^)>=====> Error uploading {filename}: {e}")
            else:
                print(f"\n <======<(^-^)>=====> File not found: {file_path}")
    
    def list_available_files(self):
        """List all available files"""
        files = self.processor.get_available_files()
        if not files:
            print("📭 No files in input folder.")
            return
            
        print("\n\n <======<(^-^)>=====> Available Files:")
        print("-" * 50)
        for file in files:
            file_path = os.path.join(self.processor.input_folder, file)
            file_size = os.path.getsize(file_path)
            file_type = Path(file).suffix
            print(f"  \n <======<(^-^)>=====> {file} ({file_type}, {file_size} bytes)")
    
    def show_latest_code_operations(self):
        """Display latest code generation and execution logs"""
        code_logs = [log for log in self.logger.logs 
                    if log['type'] in ['CODE_GENERATION', 'CODE_EXECUTION']]
        
        if not code_logs:
            print("\n <======<(^-^)>=====>  No code operations yet.")
            return
            
        print("\n\n <======<(^-^)>=====>  Latest Code Operations:")
        print("="*60)
        
        for log in code_logs[-5:]:  # Show last 5 operations
            print(f"\n[{log['timestamp'].strftime('%H:%M:%S')}] {log['type']}:")
            if log['type'] == 'CODE_GENERATION':
                print("Generated Code:")
                print("-" * 40)
                print(log['content'])
            else:
                print("Execution Output:")
                print("-" * 40)
                print(log['content'])
            print("-" * 60)
        
    def process_user_request(self, user_input: str):
        """Process user request using the AI agent"""
        # Add to conversation
        self.conversation.append({"role": "user", "content": user_input})
        
        # Log the request
        self.logger.add_log("USER_REQUEST", user_input)
        
        print("\n\n <======<(^-^)>=====> Processing your request...")
        
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
            
            # Display response
            print("\n\n <======<(^-^)>=====> Assistant Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.conversation.append({"role": "assistant", "content": error_msg})
            self.logger.add_log("ERROR", error_msg)
            print(f"\n <======<(^-^)>=====> {error_msg}")
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def run(self):
        """Main CLI loop"""
        self.clear_screen()
        self.display_banner()
        
        # Test agent tools on startup
        print("\n\n <======<(^-^)>=====> Initializing agent tools...")
        if not self.test_agent_tools():
            print("\n <======<(^-^)>=====>  Agent tools initialization had issues, but continuing...")
        
        while True:
            try:
                print("\n" + "─" * 40)
                user_input = input("\n <======<(^-^)>=====>  Enter command or query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    print("\n <======<(^-^)>=====> Goodbye!<======<(^-^)>=====>")
                    break
                    
                elif user_input.lower() in ['/help', '/h']:
                    self.display_banner()
                    
                elif user_input.lower() in ['/files', '/f']:
                    self.list_available_files()  # This uses direct file access, not tools
                    
                elif user_input.lower() in ['/upload', '/u']:
                    self.upload_files()
                    
                elif user_input.lower() in ['/logs', '/l']:
                    self.logger.display_logs()
                    
                elif user_input.lower() in ['/code', '/c']:
                    self.show_latest_code_operations()
                    
                elif user_input.lower() == '/clear':
                    self.clear_screen()
                    self.display_banner()
                    
                elif user_input.lower() in ['/test', '/t']:
                    self.test_agent_tools()
                    
                else:
                    # Process as AI query - this will use tools through the agent
                    self.process_user_request(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n\n <======<(^-^)>=====> Interrupted by user. Use '/exit' to quit properly.")
            except Exception as e:
                print(f"\n <======<(^-^)>=====>Unexpected error: {e}")

def main():
    """Main entry point for CLI version"""
    parser = argparse.ArgumentParser(description='Dynamic File Processor - CLI Version')
    parser.add_argument('--input-folder', '-i', default='input_files', 
                       help='Input folder for files (default: input_files)')
    parser.add_argument('--openai-key', '-k', help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Set OpenAI API key if provided
    if args.openai_key:
        os.environ['GOOGLE_API_KEY'] = args.openai_key
    elif not os.getenv('GOOGLE_API_KEY'):
        print("\n <======<(^-^)>=====> GOOGLE API key not found!")
        print("Please set GOOGLE_API_KEY environment variable or use --openai-key argument")
        sys.exit(1)
    
    # Initialize and run CLI
    try:
        cli = CLIFileProcessor()
        cli.run()
    except Exception as e:
        print(f"\n <======<(^-^)>=====>Failed to start CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()