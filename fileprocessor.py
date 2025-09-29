import os
from typing import Dict, Any, List
import json
import uuid
from datetime import datetime



class DynamicFileProcessor:
    def __init__(self,logger ,input_folder: str = "input_files", output_folder: str = "output_files"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.logger = logger

        self.setup_folders()
        
    def setup_folders(self):
        """Create necessary folders"""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_available_files(self) -> List[str]:
        """Get list of all files in input folder"""
        try:
            files = []
            for file in os.listdir(self.input_folder):
                if os.path.isfile(os.path.join(self.input_folder, file)):
                    files.append(file)
            return files
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            return []


class ProcessingLogger:
    def __init__(self,logger):
        self.logs = []
        self.logger = logger
    
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
        self.logger.info(f"{log_type}: {content}")
    
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

