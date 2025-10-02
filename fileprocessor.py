import os
from typing import Dict, Any, List
import json
import uuid
from datetime import datetime



class DynamicFileProcessor:
    def __init__(self,logger):
        self.input_folder = None
        self.output_folder = None
        self.logger = logger
        # self.setup_folders(input_folder,output_folder)

    def get_folder_path(self,folder_type):
        folder_type = folder_type.lower()
        if "input" in folder_type :
            return self.input_folder
        if "output" in folder_type :
            return self.output_folder
    
    def setup_folders(self,input_folder: str = "input_files", output_folder: str = "output_files"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        """Create necessary folders"""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_available_files(self,folder_type) -> List[str]:
        """Get list of all files in input folder"""
        path = self.get_folder_path(folder_type)
        try:
            files = []
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)):
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

