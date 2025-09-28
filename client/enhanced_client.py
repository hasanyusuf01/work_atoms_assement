#!/usr/bin/env python
"""
Enhanced MCP client with Claude Desktop-like interface
- Shows thinking process and tool usage
- Maintains conversation history
- Real-time streaming of agent actions
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from dotenv import load_dotenv
load_dotenv()

class MessageType(Enum):
    HUMAN = "human"
    AI = "ai"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"

@dataclass
class ConversationMessage:
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class EnhancedMCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.conversation_history: List[ConversationMessage] = []
        self.mcp_client = None
        self.agent = None
        self.tools = []
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Server parameters
        self.server_params = StdioServerParameters(
            command="python" if server_script.endswith(".py") else "node",
            args=[server_script],
        )
        
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup console styling constants"""
        self.colors = {
            'human': '\033[94m',  # Blue
            'ai': '\033[92m',     # Green
            'thinking': '\033[93m', # Yellow
            'tool_call': '\033[95m', # Magenta
            'tool_result': '\033[96m', # Cyan
            'system': '\033[91m', # Red
            'timestamp': '\033[90m', # Gray
            'reset': '\033[0m'
        }
        
        self.icons = {
            'human': '👤',
            'ai': '🤖',
            'thinking': '💭',
            'tool_call': '🛠️',
            'tool_result': '📊',
            'system': '⚙️',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }
    
    def _print_message(self, message: ConversationMessage):
        """Print a formatted message based on its type"""
        color = self.colors[message.type.value]
        icon = self.icons[message.type.value]
        timestamp = message.timestamp.strftime("%H:%M:%S")
        
        print(f"\n{self.colors['timestamp']}[{timestamp}]{self.colors['reset']} {color}{icon} ", end="")
        
        if message.type == MessageType.THINKING:
            print("THINKING PROCESS" + self.colors['reset'])
            print(f"{color}└─ {message.content}{self.colors['reset']}")
        
        elif message.type == MessageType.TOOL_CALL:
            tool_name = message.metadata.get('tool_name', 'Unknown Tool')
            print(f"TOOL CALL: {tool_name}{self.colors['reset']}")
            if 'args' in message.metadata:
                print(f"{color}├─ Arguments: {json.dumps(message.metadata['args'], indent=2)}{self.colors['reset']}")
        
        elif message.type == MessageType.TOOL_RESULT:
            success = message.metadata.get('success', False)
            status_icon = self.icons['success'] if success else self.icons['error']
            print(f"TOOL RESULT {status_icon}{self.colors['reset']}")
            if 'result_preview' in message.metadata:
                print(f"{color}└─ {message.metadata['result_preview']}{self.colors['reset']}")
        
        elif message.type == MessageType.HUMAN:
            print(f"USER{self.colors['reset']}")
            print(f"{color}└─ {message.content}{self.colors['reset']}")
        
        elif message.type == MessageType.AI:
            print(f"ASSISTANT{self.colors['reset']}")
            print(f"{color}└─ {message.content}{self.colors['reset']}")
        
        elif message.type == MessageType.SYSTEM:
            print(f"SYSTEM{self.colors['reset']}")
            print(f"{color}└─ {message.content}{self.colors['reset']}")
    
    def _add_message(self, message_type: MessageType, content: str, metadata: dict = None):
        """Add a message to conversation history and print it"""
        message = ConversationMessage(
            type=message_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.conversation_history.append(message)
        self._print_message(message)
    
    def _display_conversation_history(self):
        """Display the entire conversation history"""
        print(f"\n{self.colors['system']}{'='*60}")
        print(f"📚 CONVERSATION HISTORY")
        print(f"{'='*60}{self.colors['reset']}")
        
        for message in self.conversation_history:
            self._print_message(message)
        
        print(f"\n{self.colors['system']}{'='*60}")
        print(f"Total messages: {len(self.conversation_history)}")
        print(f"{'='*60}{self.colors['reset']}")
    
    def _extract_thinking_from_response(self, response: Any) -> List[str]:
        """Extract thinking process from agent response"""
        thinking_steps = []
        
        try:
            # Try to parse LangGraph/LangChain response format
            if hasattr(response, 'get') and isinstance(response, dict):
                messages = response.get('messages', [])
                for msg in messages:
                    if hasattr(msg, 'content'):
                        content = msg.content
                        if 'Thought:' in content or 'Action:' in content:
                            thinking_steps.append(content)
            
            # Fallback: convert entire response to string and look for patterns
            response_str = str(response)
            if 'Thought:' in response_str:
                lines = response_str.split('\n')
                for line in lines:
                    if line.strip().startswith('Thought:') or line.strip().startswith('Action:'):
                        thinking_steps.append(line.strip())
        
        except Exception as e:
            thinking_steps.append(f"Error extracting thoughts: {str(e)}")
        
        return thinking_steps
    
    async def _simulate_thinking_process(self, user_input: str):
        """Simulate AI thinking process before execution"""
        thinking_messages = [
            f"Analyzing user request: '{user_input}'",
            "Checking available tools and capabilities",
            "Planning step-by-step execution strategy",
            "Preparing to generate and execute code if needed"
        ]
        
        for msg in thinking_messages:
            self._add_message(MessageType.THINKING, msg)
            await asyncio.sleep(0.5)  # Simulate thinking time
    
    async def initialize_agent(self):
        """Initialize MCP connection and agent"""
        try:
            self._add_message(MessageType.SYSTEM, "Initializing MCP connection...")
            
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    self._add_message(MessageType.SYSTEM, "Loading MCP tools...")
                    self.tools = await load_mcp_tools(session)
                    
                    self._add_message(MessageType.SYSTEM, "Creating AI agent...")
                    self.agent = create_react_agent(self.llm, self.tools)
                    
                    self._add_message(
                        MessageType.SYSTEM, 
                        f"Agent initialized with {len(self.tools)} tools: {[tool.name for tool in self.tools]}"
                    )
                    
                    return True
                    
        except Exception as e:
            self._add_message(
                MessageType.SYSTEM, 
                f"Error initializing agent: {str(e)}", 
                {'error': True}
            )
            return False
    
    async def process_user_input(self, user_input: str):
        """Process user input with full thinking and execution display"""
        # Add user message to history
        self._add_message(MessageType.HUMAN, user_input)
        
        # Simulate thinking process
        await self._simulate_thinking_process(user_input)
        
        try:
            # Execute agent
            self._add_message(MessageType.THINKING, "Executing agent with provided tools...")
            
            start_time = time.time()
            response = await self.agent.ainvoke({"messages": user_input})
            execution_time = time.time() - start_time
            
            # Extract and display thinking process
            thinking_steps = self._extract_thinking_from_response(response)
            for step in thinking_steps:
                self._add_message(MessageType.THINKING, step)
            
            # Display final response
            if hasattr(response, 'get') and 'output' in response:
                final_response = response['output']
            else:
                final_response = str(response)
            
            self._add_message(
                MessageType.AI, 
                final_response,
                {'execution_time': execution_time}
            )
            
            # Show execution summary
            self._add_message(
                MessageType.SYSTEM,
                f"Execution completed in {execution_time:.2f}s"
            )
            
        except Exception as e:
            self._add_message(
                MessageType.AI,
                f"I encountered an error while processing your request: {str(e)}",
                {'error': True}
            )
    
    async def interactive_chat_loop(self):
        """Run interactive chat loop with enhanced interface"""
        if not await self.initialize_agent():
            self._add_message(MessageType.SYSTEM, "Failed to initialize agent. Exiting.")
            return
        
        self._add_message(MessageType.SYSTEM, "Enhanced MCP Client Ready! Type 'help' for commands.")
        
        while True:
            try:
                print(f"\n{self.colors['human']}👤 You:{self.colors['reset']} ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self._add_message(MessageType.SYSTEM, "Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self._display_conversation_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    self._add_message(MessageType.SYSTEM, "Conversation history cleared.")
                    continue
                
                elif user_input.lower() == 'tools':
                    self._list_available_tools()
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                # Process normal user input
                await self.process_user_input(user_input)
                
            except KeyboardInterrupt:
                self._add_message(MessageType.SYSTEM, "Session interrupted by user.")
                break
            except Exception as e:
                self._add_message(
                    MessageType.SYSTEM,
                    f"Unexpected error: {str(e)}",
                    {'error': True}
                )
    
    def _show_help(self):
        """Display help information"""
        help_text = """
Available Commands:
• Type your normal questions/instructions - Process files, generate code, etc.
• 'history' - Show full conversation history
• 'clear' - Clear conversation history
• 'tools' - List available MCP tools
• 'status' - Show current system status
• 'help' - Show this help message
• 'quit' - Exit the application

Example Workflows:
• "Check if abc.pdf exists in the input folder"
• "Generate a summary of all PDF files and save them"
• "Process the latest CSV file and show me analysis"
• "List all available tools and what they do"
        """
        self._add_message(MessageType.SYSTEM, help_text)
    
    def _list_available_tools(self):
        """List available MCP tools"""
        tool_list = "\n".join([f"• {tool.name}: {tool.description}" for tool in self.tools])
        self._add_message(
            MessageType.SYSTEM,
            f"Available Tools ({len(self.tools)}):\n{tool_list}"
        )
    
    def _show_status(self):
        """Show current system status"""
        status_info = {
            "Conversation Messages": len(self.conversation_history),
            "Available Tools": len(self.tools),
            "LLM Model": "Gemini 2.5 Flash",
            "MCP Server": self.server_script,
            "Last Activity": self.conversation_history[-1].timestamp if self.conversation_history else "None"
        }
        
        status_text = "\n".join([f"• {k}: {v}" for k, v in status_info.items()])
        self._add_message(MessageType.SYSTEM, f"System Status:\n{status_text}")

# Custom JSON encoder for complex objects
class EnhancedEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        elif isinstance(o, ConversationMessage):
            return {
                "type": o.type.value,
                "content": o.content,
                "timestamp": o.timestamp.isoformat(),
                "metadata": o.metadata
            }
        return super().default(o)

# async def main():
#     if len(sys.argv) < 2:
#         print("Usage: python enhanced_client.py <path_to_server_script>")
#         sys.exit(1)
    
#     server_script = sys.argv[1]
    
#     # Create and run enhanced client
#     client = EnhancedMCPClient(server_script)
#     await client.interactive_chat_loop()

# if __name__ == "__main__":
#     asyncio.run(main())