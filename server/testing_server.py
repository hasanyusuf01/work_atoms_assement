#!/usr/bin/env python3
"""
Dynamic File Processor MCP Server
Enables AI agents to process files based on natural language instructions
with live code generation and execution.
"""

from mcp.server.fastmcp import FastMCP
import os
import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


# Instantiate MCP server
mcp = FastMCP("Dynamic File Processor")

# Configuration
INPUT_FOLDER = Path("input")
OUTPUT_FOLDER = Path("output")
TEMP_CODE_FOLDER = Path("temp_code")

# Ensure directories exist
INPUT_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
TEMP_CODE_FOLDER.mkdir(exist_ok=True)

# ============ FILE MANAGEMENT TOOLS ============

@mcp.tool()
def list_input_files(extension_filter: Optional[str] = None) -> Dict[str, Any]:
    """List all files in the input folder, optionally filtered by extension.
    
    Args:
        extension_filter: Filter by file extension (e.g., 'pdf', 'csv')
    
    Returns:
        Dictionary with list of files and their metadata
    """
    try:
        files = []
        for file_path in INPUT_FOLDER.iterdir():
            if file_path.is_file():
                if extension_filter and not file_path.name.lower().endswith(f".{extension_filter.lower()}"):
                    continue
                
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'extension': file_path.suffix.lower()
                })
        
        return {
            'success': True,
            'files': files,
            'total_count': len(files),
            'input_folder': str(INPUT_FOLDER.absolute())
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@mcp.tool()
def check_file_exists(filename: str) -> Dict[str, Any]:
    """Check if a specific file exists in the input folder.
    
    Args:
        filename: Name of the file to check (e.g., 'abc.pdf')
    
    Returns:
        Dictionary with existence status and file details
    """
    try:
        file_path = INPUT_FOLDER / filename
        exists = file_path.exists() and file_path.is_file()
        
        result = {
            'success': True,
            'exists': exists,
            'filename': filename,
            'full_path': str(file_path.absolute())
        }
        
        if exists:
            stat = file_path.stat()
            result.update({
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'readable': os.access(file_path, os.R_OK)
            })
        
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}

@mcp.tool()
def get_file_info(filename: str) -> Dict[str, Any]:
    """Get detailed information about a specific file.
    
    Args:
        filename: Name of the file to analyze
    
    Returns:
        Dictionary with file metadata and content preview
    """
    try:
        file_path = INPUT_FOLDER / filename
        
        if not file_path.exists():
            return {'success': False, 'error': f"File {filename} not found"}
        
        stat = file_path.stat()
        info = {
            'success': True,
            'filename': filename,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': file_path.suffix.lower(),
            'readable': os.access(file_path, os.R_OK)
        }
        
        # Add content preview for text-based files
        if file_path.suffix.lower() in ['.txt', '.csv', '.py', '.json', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    preview = f.read(1000)  # First 1000 characters
                    info['preview'] = preview
                    info['preview_lines'] = len(preview.split('\n'))
            except:
                info['preview'] = "Unable to read file content"
        
        return info
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============ PERMISSION & SAFETY TOOLS ============

@mcp.tool()
def request_user_permission(filename: str, operation: str, reason: str = "") -> Dict[str, Any]:
    """Request user permission before processing a file.
    
    Args:
        filename: File to be processed
        operation: Type of operation (read, write, execute, etc.)
        reason: Explanation why this operation is needed
    
    Returns:
        Dictionary with permission status (for demo, always approved)
    """
    # In a real implementation, this would show a dialog to the user
    # For demo purposes, we'll log the request and return approved
    
    permission_request = {
        'filename': filename,
        'operation': operation,
        'reason': reason,
        'timestamp': os.times().elapsed,
        'request_id': f"req_{os.times().elapsed}"
    }
    
    print(f"🔐 PERMISSION REQUEST: {json.dumps(permission_request, indent=2)}")
    
    # Simulate user approval (in real implementation, await user input)
    return {
        'success': True,
        'approved': True,
        'message': f"Permission granted for {operation} on {filename}",
        'request_id': permission_request['request_id']
    }

# ============ DYNAMIC CODE GENERATION TOOLS ============

@mcp.tool()
def generate_python_code(instruction: str, target_files: List[str], output_format: str = "text") -> Dict[str, Any]:
    """Generate Python code dynamically based on natural language instruction.
    
    Args:
        instruction: Natural language instruction (e.g., "summarize PDF files")
        target_files: List of files to process
        output_format: Desired output format (text, json, pdf, etc.)
    
    Returns:
        Dictionary with generated code and metadata
    """
    try:
        # Validate target files exist
        missing_files = []
        for filename in target_files:
            if not (INPUT_FOLDER / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            return {
                'success': False, 
                'error': f"Files not found: {missing_files}",
                'suggestion': "Use check_file_exists() first"
            }
        
        # Generate code based on instruction and file types
        code_template = _generate_code_template(instruction, target_files, output_format)
        
        # Save generated code to temporary file
        code_id = f"code_{len(list(TEMP_CODE_FOLDER.glob('*.py')))}"
        code_filename = f"{code_id}.py"
        code_path = TEMP_CODE_FOLDER / code_filename
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code_template['code'])
        
        return {
            'success': True,
            'code_id': code_id,
            'filename': code_filename,
            'code_preview': code_template['code'][:500] + "..." if len(code_template['code']) > 500 else code_template['code'],
            'dependencies': code_template['dependencies'],
            'estimated_complexity': code_template['complexity'],
            'output_files_expected': code_template['output_files']
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_code_template(instruction: str, target_files: List[str], output_format: str) -> Dict[str, Any]:
    """Generate appropriate code template based on instruction and file types."""
    
    # Analyze file types
    file_extensions = [Path(f).suffix.lower() for f in target_files]
    has_pdf = any(ext == '.pdf' for ext in file_extensions)
    has_csv = any(ext == '.csv' for ext in file_extensions)
    has_image = any(ext in ['.jpg', '.jpeg', '.png', '.bmp'] for ext in file_extensions)
    has_text = any(ext == '.txt' for ext in file_extensions)
    
    # Determine operation type from instruction
    instruction_lower = instruction.lower()
    
    if 'summariz' in instruction_lower and has_pdf:
        return _generate_pdf_summarizer_code(target_files, output_format)
    elif 'analyz' in instruction_lower and has_csv:
        return _generate_csv_analyzer_code(target_files, output_format)
    elif 'process' in instruction_lower and has_image:
        return _generate_image_processor_code(target_files, output_format)
    elif 'extract' in instruction_lower:
        return _generate_text_extractor_code(target_files, output_format)
    else:
        # Generic file processor
        return _generate_generic_processor_code(instruction, target_files, output_format)

def _generate_pdf_summarizer_code(filenames: List[str], output_format: str) -> Dict[str, Any]:
    """Generate PDF summarization code."""
    
    code = f'''#!/usr/bin/env python3
"""
PDF Summarization Script
Generated for: {filenames}
"""

import PyPDF2
import re
from pathlib import Path
from collections import Counter

def summarize_pdf(filepath: Path) -> str:
    """Extract and summarize PDF content."""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\\n"
    except Exception as e:
        return f"Error reading PDF: {{e}}"
    
    # Simple summarization: extract key sentences
    sentences = re.split(r'[.!?]+', text)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
    
    # Get first 5 meaningful sentences as summary
    summary = '. '.join(meaningful_sentences[:5]) + '.'
    return summary

def main():
    input_folder = Path("input")
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)
    
    results = {{}}
    for filename in {filenames}:
        filepath = input_folder / filename
        if filepath.exists():
            print(f"Processing {{filename}}...")
            summary = summarize_pdf(filepath)
            results[filename] = summary
            
            # Save individual summary
            output_file = output_folder / f"{{filename}}_summary.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Saved summary to {{output_file}}")
    
    # Save combined results
    combined_output = output_folder / "pdf_summaries.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"All summaries saved to {{combined_output}}")
    return results

if __name__ == "__main__":
    main()
'''

    return {
        'code': code,
        'dependencies': ['PyPDF2'],
        'complexity': 'medium',
        'output_files': [f"{f}_summary.txt" for f in filenames] + ['pdf_summaries.json']
    }

def _generate_csv_analyzer_code(filenames: List[str], output_format: str) -> Dict[str, Any]:
    """Generate CSV analysis code."""
    
    code = f'''#!/usr/bin/env python3
"""
CSV Analysis Script
Generated for: {filenames}
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

def analyze_csv(filepath: Path) -> dict:
    """Analyze CSV file and generate statistics."""
    try:
        df = pd.read_csv(filepath)
        
        analysis = {{
            'filename': filepath.name,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': {{col: str(dtype) for col, dtype in df.dtypes.items()}},
            'null_counts': df.isnull().sum().to_dict(),
            'basic_stats': df.describe().to_dict()
        }}
        
        return analysis
    except Exception as e:
        return {{'error': str(e)}}

def main():
    input_folder = Path("input")
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)
    
    all_analyses = {{}}
    for filename in {filenames}:
        filepath = input_folder / filename
        if filepath.exists():
            print(f"Analyzing {{filename}}...")
            analysis = analyze_csv(filepath)
            all_analyses[filename] = analysis
            
            # Save individual analysis
            output_file = output_folder / f"{{filename}}_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            print(f"Saved analysis to {{output_file}}")
    
    print("CSV analysis completed!")
    return all_analyses

if __name__ == "__main__":
    main()
'''

    return {
        'code': code,
        'dependencies': ['pandas', 'matplotlib'],
        'complexity': 'medium',
        'output_files': [f"{f}_analysis.json" for f in filenames]
    }

# ============ CODE EXECUTION TOOLS ============

@mcp.tool()
def execute_generated_code(code_id: str, stream_output: bool = True) -> Dict[str, Any]:
    """Execute previously generated Python code with live output streaming.
    
    Args:
        code_id: ID of the generated code to execute
        stream_output: Whether to stream output in real-time
    
    Returns:
        Dictionary with execution results and outputs
    """
    try:
        code_path = TEMP_CODE_FOLDER / f"{code_id}.py"
        
        if not code_path.exists():
            return {'success': False, 'error': f"Code with ID {code_id} not found"}
        
        print(f"🚀 EXECUTING CODE: {code_id}")
        print("=" * 60)
        
        # Execute the code and capture output
        result = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            cwd=Path.cwd()  # Run from current directory to access input/output folders
        )
        
        execution_result = {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'code_id': code_id
        }
        
        # Stream output if requested
        if stream_output:
            print("📊 EXECUTION OUTPUT:")
            print(result.stdout)
            if result.stderr:
                print("❌ ERRORS:")
                print(result.stderr)
            print("=" * 60)
        
        return execution_result
    except Exception as e:
        return {'success': False, 'error': str(e)}

@mcp.tool()
def execute_python_code(code: str, stream_output: bool = True) -> Dict[str, Any]:
    """Execute Python code directly with live output streaming.
    
    Args:
        code: Python code to execute
        stream_output: Whether to stream output in real-time
    
    Returns:
        Dictionary with execution results
    """
    try:
        # Create temporary file for execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_code_path = f.name
        
        print(f"🚀 EXECUTING DIRECT CODE")
        print("=" * 60)
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_code_path],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Clean up temporary file
        os.unlink(temp_code_path)
        
        execution_result = {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Stream output if requested
        if stream_output:
            print("📊 EXECUTION OUTPUT:")
            print(result.stdout)
            if result.stderr:
                print("❌ ERRORS:")
                print(result.stderr)
            print("=" * 60)
        
        return execution_result
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============ OUTPUT MANAGEMENT TOOLS ============

@mcp.tool()
def list_output_files() -> Dict[str, Any]:
    """List all files in the output folder.
    
    Returns:
        Dictionary with list of output files
    """
    try:
        files = []
        for file_path in OUTPUT_FOLDER.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
        
        return {
            'success': True,
            'files': files,
            'total_count': len(files),
            'output_folder': str(OUTPUT_FOLDER.absolute())
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@mcp.tool()
def preview_output_file(filename: str, max_lines: int = 20) -> Dict[str, Any]:
    """Preview the content of an output file.
    
    Args:
        filename: Name of the output file to preview
        max_lines: Maximum number of lines to display
    
    Returns:
        Dictionary with file preview
    """
    try:
        file_path = OUTPUT_FOLDER / filename
        
        if not file_path.exists():
            return {'success': False, 'error': f"Output file {filename} not found"}
        
        preview = {}
        
        # Text file preview
        if file_path.suffix.lower() in ['.txt', '.csv', '.json', '.log']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                preview['content'] = ''.join(lines[:max_lines])
                preview['total_lines'] = len(lines)
                preview['type'] = 'text'
        
        # JSON file preview
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                preview['content'] = json.dumps(data, indent=2)[:2000]  # Limit size
                preview['type'] = 'json'
        
        else:
            preview['type'] = 'binary'
            preview['message'] = f"Binary file preview not available for {filename}"
        
        return {
            'success': True,
            'filename': filename,
            'preview': preview,
            'full_path': str(file_path.absolute())
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@mcp.tool()
def cleanup_output_folder(confirm: bool = False) -> Dict[str, Any]:
    """Clean up the output folder (remove all files).
    
    Args:
        confirm: Safety confirmation required
    
    Returns:
        Dictionary with cleanup results
    """
    if not confirm:
        return {
            'success': False,
            'error': 'Safety confirmation required. Set confirm=True to proceed.',
            'warning': 'This will delete all files in the output folder!'
        }
    
    try:
        deleted_files = []
        for file_path in OUTPUT_FOLDER.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_files.append(file_path.name)
        
        return {
            'success': True,
            'deleted_files': deleted_files,
            'message': f"Cleaned up {len(deleted_files)} files from output folder"
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============ RESOURCE DEFINITIONS ============

@mcp.resource("config://paths")
def get_config_paths() -> str:
    """Get current configuration paths."""
    return json.dumps({
        'input_folder': str(INPUT_FOLDER.absolute()),
        'output_folder': str(OUTPUT_FOLDER.absolute()),
        'temp_code_folder': str(TEMP_CODE_FOLDER.absolute()),
        'current_working_directory': str(Path.cwd())
    }, indent=2)

@mcp.resource("status://files")
def get_file_status() -> str:
    """Get current file system status."""
    input_files = len(list(INPUT_FOLDER.iterdir()))
    output_files = len(list(OUTPUT_FOLDER.iterdir()))
    temp_files = len(list(TEMP_CODE_FOLDER.iterdir()))
    
    return json.dumps({
        'input_files_count': input_files,
        'output_files_count': output_files,
        'temp_code_files_count': temp_files,
        'status': 'ready' if input_files > 0 else 'waiting_for_input'
    }, indent=2)

# ============ HELPER FUNCTIONS ============

def _generate_image_processor_code(filenames: List[str], output_format: str) -> Dict[str, Any]:
    """Generate image processing code (placeholder)."""
    code = f"# Image processing code for {filenames}"
    return {
        'code': code,
        'dependencies': ['PIL'],
        'complexity': 'medium',
        'output_files': [f"{f}_processed.jpg" for f in filenames]
    }

def _generate_text_extractor_code(filenames: List[str], output_format: str) -> Dict[str, Any]:
    """Generate text extraction code (placeholder)."""
    code = f"# Text extraction code for {filenames}"
    return {
        'code': code,
        'dependencies': [],
        'complexity': 'low',
        'output_files': [f"{f}_extracted.txt" for f in filenames]
    }

def _generate_generic_processor_code(instruction: str, filenames: List[str], output_format: str) -> Dict[str, Any]:
    """Generate generic file processing code."""
    code = f'''#!/usr/bin/env python3
"""
Generic File Processor
Instruction: {instruction}
Files: {filenames}
"""

from pathlib import Path
import shutil

def main():
    input_folder = Path("input")
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)
    
    print("Processing files based on instruction: {instruction}")
    
    for filename in {filenames}:
        filepath = input_folder / filename
        if filepath.exists():
            print(f"Processing {{filename}}...")
            # Basic copy operation - replace with actual processing logic
            output_path = output_folder / f"processed_{{filename}}"
            shutil.copy2(filepath, output_path)
            print(f"Saved to {{output_path}}")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
'''
    return {
        'code': code,
        'dependencies': [],
        'complexity': 'low',
        'output_files': [f"processed_{f}" for f in filenames]
    }

# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("🚀 Starting Dynamic File Processor MCP Server...")
    print(f"📁 Input folder: {INPUT_FOLDER.absolute()}")
    print(f"📁 Output folder: {OUTPUT_FOLDER.absolute()}")
    # print("🔧 Available tools: list_input_files, check_file_exists, generate_python_code, execute_generated_code, etc.")
    print("=" * 60)
    
    mcp.run(transport="stdio")