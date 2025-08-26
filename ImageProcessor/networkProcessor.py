"""
Network Processor Module

Implements client/server architecture for distributed RAW image processing.
- Server mode: Distributes processing jobs to connected clients
- Client mode: Receives and processes jobs from server
- Heartbeat monitoring and basic logging
- UNC path handling for network file access
"""

import json
import socket
import threading
import time
import os
import sys
import random
import psutil
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from enum import Enum

# Note: Using custom signal implementation for ImageCorrectionWorker compatibility

class LogLevel(Enum):
    """Logging levels for network operations."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

# Import for image processing
try:
    # Try relative import first (when used as module)
    from .imageProcessorWorker import ImageCorrectionWorker
    import numpy as np
except ImportError:
    try:
        # Try absolute import (when run standalone)
        from imageProcessorWorker import ImageCorrectionWorker
        import numpy as np
    except ImportError as e:
        print(f"Warning: Failed to import image processing dependencies: {e}")
        ImageCorrectionWorker = None

try:
    import win32wnet
except ImportError:
    print("Warning: pywin32 not available - UNC path conversion will be limited")
    win32wnet = None


class ConsoleUI:
    """Beautiful console interface with status bar and colored logging."""
    
    def __init__(self, enable_colors=True):
        self.enable_colors = enable_colors and self._supports_color()
        self.terminal_width = shutil.get_terminal_size().columns
        self.terminal_height = shutil.get_terminal_size().lines
        self.status_bar_enabled = False
        self.status_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.current_jobs = {}
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'processing_time': 0.0,
            'connection_status': 'Disconnected',
            'server_host': '',
            'client_id': '',
            'thread_count': 0
        }
        
        # Interactive command system
        self.command_thread = None
        self.command_running = False
        self.client_instance = None  # Will be set by create_headless_client
        self.command_queue = Queue()  # For sending commands to main thread

        # Status line placement
        self.log_line1 = self.terminal_height - 5
        self.input_line1 = self.terminal_height - 4
        self.input_line2 = self.terminal_height - 3
        self.status_line1 = self.terminal_height - 2
        self.status_line2 = self.terminal_height - 1
        self.status_line3 = self.terminal_height
        
        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bg_blue': '\033[44m',
            'bg_green': '\033[42m',
            'bg_red': '\033[41m'
        }
        
        if not self.enable_colors:
            self.colors = {k: '' for k in self.colors.keys()}
    
    def _supports_color(self):
        """Check if terminal supports color output."""
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            pass
        
        # Check environment variables
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            os.environ.get('TERM', '').lower() not in ['', 'dumb']
        )
    
    def enable_status_bar(self):
        """Enable the status bar display with log area limited to top 12 lines."""
        self.status_bar_enabled = True
        # Clear screen and position cursor
        if self.enable_colors:
            print('\033[2J\033[H', end='', flush=True)
            # Set scroll region to top 12 lines only
            print(f'\033[1;{self.log_line1}r', end='', flush=True)  # Set scroll region to the defined height
            # Position cursor at top of scroll region
            print('\033[1;1H', end='', flush=True)

    def disable_status_bar(self):
        """Disable the status bar display."""
        self.status_bar_enabled = False
        if self.enable_colors:
            # Reset scroll region to full screen
            print('\033[r', end='', flush=True)  # Reset scroll region
            print('\033[?30h', end='', flush=True)  # Show cursor
            # Clear the status bar and input area
            terminal_height = shutil.get_terminal_size().lines
            for i in range(5):  # Clear input area (lines 14-15) and status bar (bottom 3)
                print(f'\033[{terminal_height-4+i};1H\033[K', end='')
            # Also clear lines 14-15 for input area
            print('\033[24;1H\033[K', end='')
            print('\033[25;1H\033[K', end='')
            print('', flush=True)
    
    def update_stats(self, **kwargs):
        """Update statistics for the status bar."""
        with self.status_lock:
            self.stats.update(kwargs)
    
    def add_job(self, job_id: str, filename: str):
        """Add a job to the current processing list."""
        with self.status_lock:
            self.current_jobs[job_id] = {
                'filename': filename,
                'start_time': time.time(),
                'status': 'Processing'
            }
    
    def complete_job(self, job_id: str, success: bool = True, processing_time: float = 0):
        """Mark a job as completed."""
        with self.status_lock:
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
            
            if success:
                self.stats['total_processed'] += 1
            else:
                self.stats['total_failed'] += 1
            
            if processing_time > 0:
                self.stats['processing_time'] += processing_time
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message with beautiful formatting."""
        with self.log_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Color mapping for log levels
            level_colors = {
                'DEBUG': self.colors['dim'],
                'INFO': self.colors['cyan'],
                'WARNING': self.colors['yellow'],
                'ERROR': self.colors['red'],
                'SUCCESS': self.colors['green']
            }
            
            color = level_colors.get(level, self.colors['white'])
            reset = self.colors['reset']
            
            # Format the log message
            formatted_msg = f"{self.colors['dim']}[{timestamp}]{reset} {color}{level:>7}{reset} {message}"

            # Save cursor position
            print('\033[s', end='')

            # Move to bottom of log area
            print(f'\033[{self.log_line1};1H', end='')

            # Print the message (this will scroll within lines 1-12)
            print(formatted_msg)

            # Restore cursor position
            print('\033[u', end='', flush=True)

            # Redraw status bar and input area
        self._draw_status_bar()

    
    def _draw_status_bar(self):
        """Draw the status bar at the bottom of the terminal."""
        if not self.status_bar_enabled or not self.enable_colors:
            return
        
        with self.status_lock:
            # Get actual terminal dimensions
            terminal_height = shutil.get_terminal_size().lines
            terminal_width = shutil.get_terminal_size().columns
            
            # Calculate positions: input area at line 14-15, status bar at bottom 3 lines
            self.input_line1 = terminal_height - 4
            self.input_line2 = terminal_width - 3
            self.status_line1 = terminal_height - 2
            self.status_line2 = terminal_height - 1
            self.status_line3 = terminal_height
            
            # Save cursor position
            print('\033[s', end='')

            # Move to bottom of screen
            print('\033[9999;1H', end='')

            # Create status bar content
            connection_color = self.colors['bg_green'] if self.stats['connection_status'] == 'Connected' else self.colors['bg_red']
            
            # Line 1: Connection status and server info
            line1_parts = []
            if self.stats['connection_status'] == 'Connected':
                line1_parts.append(f"{connection_color}{self.colors['white']} CONNECTED {self.colors['reset']}")
                if self.stats['server_host']:
                    line1_parts.append(f"{self.colors['cyan']}{self.stats['server_host']}{self.colors['reset']}")
                if self.stats['client_id']:
                    line1_parts.append(f"{self.colors['dim']}ID: {self.stats['client_id'][-8:]}{self.colors['reset']}")
            else:
                line1_parts.append(f"{connection_color}{self.colors['white']} DISCONNECTED {self.colors['reset']}")
            
            line1 = " │ ".join(line1_parts)
            
            # Line 2: Processing stats
            active_jobs = len(self.current_jobs)
            avg_time = self.stats['processing_time'] / max(1, self.stats['total_processed']) if self.stats['total_processed'] > 0 else 0
            
            line2_parts = [
                f"{self.colors['green']}✓ {self.stats['total_processed']}{self.colors['reset']}",
                f"{self.colors['red']}✗ {self.stats['total_failed']}{self.colors['reset']}",
                f"{self.colors['yellow']}⚙ {active_jobs}/{self.stats['thread_count']}{self.colors['reset']}",
            ]
            
            if avg_time > 0:
                line2_parts.append(f"{self.colors['blue']}⏱ {avg_time:.1f}s/img{self.colors['reset']}")
            
            line2 = " │ ".join(line2_parts)
            
            # Line 3: Current jobs
            line3 = ""
            if self.current_jobs:
                job_displays = []
                for job_id, job_info in list(self.current_jobs.items())[:3]:  # Show max 3 jobs
                    filename = job_info['filename']
                    if len(filename) > 20:
                        filename = filename[:17] + "..."
                    elapsed = time.time() - job_info['start_time']
                    job_displays.append(f"{self.colors['magenta']}{filename}{self.colors['reset']} {self.colors['dim']}({elapsed:.0f}s){self.colors['reset']}")
                
                if len(self.current_jobs) > 3:
                    job_displays.append(f"{self.colors['dim']}+{len(self.current_jobs)-3} more{self.colors['reset']}")
                
                line3 = " │ ".join(job_displays)
            else:
                line3 = f"{self.colors['dim']}        Waiting for jobs...{self.colors['reset']}"
            
            # Clear and draw the status bar (3 lines)
            for i in range(6):
                print(f'\033[{9999-2+i};1H\033[K', end='')

            # Line 1
            print(f'\033[{self.status_line1};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line1};2H{line1}', end='')
            
            # Line 2
            print(f'\033[{self.status_line2};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line2};2H{line2}', end='')
            
            # Line 3
            print(f'\033[{self.status_line3};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line3};2H{line3}', end='')
            
            # Restore cursor position
            print('\033[u', end='', flush=True)

            # Get terminal dimensions
            terminal_width = shutil.get_terminal_size().columns
            # Save cursor position
            print('\033[s', end='')
            
            # Clear and draw input area background
            # Line 1: Command prompt label
            prompt_text = f"{self.colors['bold']}{self.colors['cyan']}Commands:{self.colors['reset']} help, status, threads N, stats, clear, reconnect, quit"
            print(f'\033[{self.input_line1};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.input_line1};2H{prompt_text}', end='')
            
            # Line 2: Input prompt
            input_prompt = f"{self.colors['bold']}{self.colors['white']}> {self.colors['reset']}"
            print(f'\033[{self.input_line2};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.input_line2};2H{input_prompt}', end='')
            
            # Restore cursor position
            print('\033[u', end='', flush=True)
    
    def start_command_interface(self, client_instance):
        """Start the interactive command interface in a separate thread."""
        self.client_instance = client_instance
        self.command_running = True
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        self.log("Interactive commands enabled. Use input area below for commands.", "INFO")
    
    def stop_command_interface(self):
        """Stop the interactive command interface."""
        self.command_running = False
        if self.command_thread:
            self.command_thread.join(timeout=1)
    
    def _command_loop(self):
        """Main loop for processing interactive commands."""
        while self.command_running:
            try:
                # Position cursor in input area (line 15, after the prompt)
                if self.status_bar_enabled and self.enable_colors:
                    print(f'\033[{self.input_line2};4H', end='', flush=True)  # Position after "> "
                
                # Get command input (this will block)
                command = input().strip().lower()
                
                # Clear the input line after command is entered
                if self.status_bar_enabled and self.enable_colors:
                    print(f'\033[{self.input_line2};4H\033[K', end='', flush=True)
                
                if not command:
                    continue
                
                # Queue the command for processing in the main thread
                self.command_queue.put(command)
                
            except (EOFError, KeyboardInterrupt):
                # User pressed Ctrl+C or EOF, stop command interface
                self.command_running = False
                break
            except Exception as e:
                self.log(f"Command input error: {e}", "ERROR")
    
    def process_command(self, command: str):
        """Process a user command."""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if cmd == 'help' or cmd == 'h':
                self._show_help()
            elif cmd == 'status' or cmd == 's':
                self._show_detailed_status()
            elif cmd == 'threads' or cmd == 't':
                self._handle_threads_command(args)
            elif cmd == 'stats' or cmd == 'st':
                self._show_stats()
            elif cmd == 'clear' or cmd == 'c':
                self._clear_screen()
            elif cmd == 'reset':
                self._reset_stats()
            elif cmd == 'reconnect' or cmd == 'r':
                self._request_reconnect()
            elif cmd == 'quit' or cmd == 'q' or cmd == 'exit':
                self._request_quit()
            else:
                self.log(f"Unknown command: '{cmd}'. Type 'help' for available commands.", "WARNING")
        except Exception as e:
            self.log(f"Command error: {e}", "ERROR")
    
    def _show_help(self):
        """Show available commands."""
        help_text = [
            "Available Commands:",
            "  help, h          - Show this help message",
            "  status, s        - Show detailed client status", 
            "  threads N, t N   - Set thread count to N (1-32)",
            "  stats, st        - Show processing statistics",
            "  clear, c         - Clear the screen",
            "  reset            - Reset processing statistics",
            "  reconnect, r     - Request reconnection to server",
            "  quit, q, exit    - Gracefully shutdown client",
            "",
            "Examples:",
            "  threads 4        - Set to 4 processing threads",
            "  t 2              - Set to 2 processing threads"
        ]
        
        for line in help_text:
            self.log(line, "INFO")
    
    def _show_detailed_status(self):
        """Show detailed client status."""
        with self.status_lock:
            self.log("=== Client Status ===", "INFO")
            self.log(f"Connection: {self.stats['connection_status']}", "INFO")
            self.log(f"Server: {self.stats['server_host']}", "INFO")
            self.log(f"Client ID: {self.stats['client_id']}", "INFO")
            self.log(f"Thread Count: {self.stats['thread_count']}", "INFO")
            self.log(f"Active Jobs: {len(self.current_jobs)}/{self.stats['thread_count']}", "INFO")
            
            if self.current_jobs:
                self.log("Current Jobs:", "INFO")
                for job_id, job_info in list(self.current_jobs.items()):
                    elapsed = time.time() - job_info['start_time']
                    self.log(f"  {job_info['filename']} ({elapsed:.0f}s)", "INFO")
    
    def _handle_threads_command(self, args):
        """Handle thread count adjustment command."""
        if not args:
            self.log(f"Current thread count: {self.stats['thread_count']}", "INFO")
            self.log("Usage: threads <number> or t <number>", "INFO")
            return
        
        try:
            new_count = int(args[0])
            if new_count < 1 or new_count > 32:
                self.log("Thread count must be between 1 and 32", "ERROR")
                return
            
            if self.client_instance:
                old_count = self.client_instance.thread_count
                self.client_instance._handle_thread_count_change(new_count)
                self.update_stats(thread_count=new_count)
                self.log(f"Thread count changed from {old_count} to {new_count}", "SUCCESS")
            else:
                self.log("Client instance not available", "ERROR")
                
        except ValueError:
            self.log("Invalid thread count. Please enter a number between 1 and 32.", "ERROR")
    
    def _show_stats(self):
        """Show processing statistics."""
        with self.status_lock:
            self.log("=== Processing Statistics ===", "INFO")
            self.log(f"Total Processed: {self.stats['total_processed']}", "SUCCESS")
            self.log(f"Total Failed: {self.stats['total_failed']}", "ERROR")
            
            if self.stats['total_processed'] > 0:
                avg_time = self.stats['processing_time'] / self.stats['total_processed']
                self.log(f"Average Processing Time: {avg_time:.2f}s per image", "INFO")
                success_rate = (self.stats['total_processed'] / (self.stats['total_processed'] + self.stats['total_failed'])) * 100
                self.log(f"Success Rate: {success_rate:.1f}%", "INFO")
    
    def _clear_screen(self):
        """Clear the screen and redraw status bar."""
        if self.enable_colors:
            print('\033[2J\033[H', end='', flush=True)
            if self.status_bar_enabled:
                # Reset scroll region
                terminal_height = shutil.get_terminal_size().lines
                scroll_bottom = terminal_height - 3
                print(f'\033[1;{scroll_bottom}r', end='', flush=True)
                self._draw_status_bar()
        self.log("Screen cleared", "INFO")
    
    def _reset_stats(self):
        """Reset processing statistics."""
        with self.status_lock:
            self.stats['total_processed'] = 0
            self.stats['total_failed'] = 0
            self.stats['processing_time'] = 0.0
        self.log("Statistics reset", "SUCCESS")
    
    def _request_reconnect(self):
        """Request reconnection to server."""
        self.command_queue.put('__reconnect__')
        self.log("Reconnection requested...", "INFO")
    
    def _request_quit(self):
        """Request graceful shutdown."""
        self.command_queue.put('__quit__')
        self.log("Graceful shutdown requested...", "WARNING")


@dataclass
class ProcessingJob:
    """Represents a single image processing job."""
    job_id: str
    image_path: str
    output_path: str
    swatches: List[List[float]]  # 24x3 color correction matrix
    settings: Dict[str, Any]  # Processing settings
    created_time: float
    start_time: Optional[float] = None  # When job actually started processing
    timeout: float = 300.0  # Job timeout in seconds (5 minutes default)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingJob':
        """Create from dictionary after JSON deserialization."""
        return cls(**data)


@dataclass
class ClientInfo:
    """Information about a connected client."""
    client_id: str
    address: str
    port: int
    thread_count: int
    last_heartbeat: float
    status: str  # 'idle', 'processing', 'disconnected'
    socket: Optional = None  # Client socket for communication
    current_jobs: List[str] = None  # List of job IDs currently being processed
    total_memory_gb: float = 0.0  # Total system memory in GB
    available_memory_gb: float = 0.0  # Available memory in GB
    memory_per_job_gb: float = 6.0  # Estimated memory usage per job in GB
    last_memory_update: float = 0.0  # Last memory report timestamp
    
    def __post_init__(self):
        if self.current_jobs is None:
            self.current_jobs = []
    
    def available_capacity(self) -> int:
        """Returns how many more jobs this client can accept based on threads and memory."""
        thread_capacity = max(0, self.thread_count - len(self.current_jobs))
        
        # Calculate memory-based capacity
        if self.available_memory_gb > 0 and self.memory_per_job_gb > 0:
            # Reserve 2GB for system overhead
            usable_memory = max(0, self.available_memory_gb - 2.0)
            memory_capacity = int(usable_memory // self.memory_per_job_gb)
            # Use the more restrictive limit
            return min(thread_capacity, memory_capacity)
        
        return thread_capacity
    
    def is_at_capacity(self) -> bool:
        """Returns True if client is at thread or memory capacity."""
        return self.available_capacity() <= 0
    
    def add_job(self, job_id: str) -> bool:
        """Add a job to this client. Returns True if successful."""
        if not self.is_at_capacity() and job_id not in self.current_jobs:
            self.current_jobs.append(job_id)
            self.status = 'processing' if self.current_jobs else 'idle'
            return True
        return False
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from this client. Returns True if job was found."""
        if job_id in self.current_jobs:
            self.current_jobs.remove(job_id)
            self.status = 'processing' if self.current_jobs else 'idle'
            return True
        return False
    
    def is_alive(self, timeout: float = 180.0) -> bool:
        """Check if client is still alive based on last heartbeat.
        
        Increased timeout to 3 minutes to handle UI blocking scenarios
        like file browsing that may temporarily freeze the server.
        """
        return (time.time() - self.last_heartbeat) < timeout
    
    def memory_health_status(self) -> str:
        """Get memory health status description."""
        if self.available_memory_gb <= 0:
            return "unknown"
        elif self.available_memory_gb < 4.0:
            return "critical"  # Less than 4GB available
        elif self.available_memory_gb < 8.0:
            return "warning"   # Less than 8GB available
        else:
            return "healthy"
    
    def validate_and_cleanup_jobs(self, active_jobs_dict: Dict) -> int:
        """
        Validate current_jobs list against actual active jobs and clean up stale entries.
        Returns the number of stale jobs removed.
        """
        stale_jobs = []
        for job_id in self.current_jobs:
            if job_id not in active_jobs_dict:
                stale_jobs.append(job_id)
        
        for job_id in stale_jobs:
            self.current_jobs.remove(job_id)
        
        # Update status based on current job count
        self.status = 'processing' if self.current_jobs else 'idle'
        
        return len(stale_jobs)


class NetworkProtocol:
    """Simple JSON-based network protocol for client/server communication."""
    
    @staticmethod
    def encode_message(msg_type: str, data: Any) -> bytes:
        """Encode a message for network transmission."""
        message = {
            'type': msg_type,
            'timestamp': time.time(),
            'data': data
        }
        json_str = json.dumps(message, separators=(',', ':'))
        # Prefix with length for message framing
        length = len(json_str.encode('utf-8'))
        return f"{length:08d}".encode('utf-8') + json_str.encode('utf-8')
    
    @staticmethod
    def decode_message(socket_obj: socket.socket, timeout: float = 30.0) -> tuple[str, Any]:
        """Decode a message from network transmission."""
        socket_obj.settimeout(timeout)
        
        # Read length prefix
        length_data = b""
        while len(length_data) < 8:
            chunk = socket_obj.recv(8 - len(length_data))
            if not chunk:
                raise ConnectionError("Connection closed while reading length")
            length_data += chunk
        
        message_length = int(length_data.decode('utf-8'))
        
        # Read message data
        message_data = b""
        while len(message_data) < message_length:
            chunk = socket_obj.recv(message_length - len(message_data))
            if not chunk:
                raise ConnectionError("Connection closed while reading message")
            message_data += chunk
        
        # Parse JSON
        message = json.loads(message_data.decode('utf-8'))
        return message['type'], message['data']


def validate_server_config(host: str, port: int) -> tuple[bool, str]:
    """Validate server configuration parameters."""
    if not host:
        return False, "Host address cannot be empty"
    
    if not isinstance(port, int) or port < 1024 or port > 65535:
        return False, "Port must be an integer between 1024 and 65535"
    
    # Test if port is available
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind((host, port))
        test_socket.close()
        return True, "Configuration valid"
    except socket.error as e:
        return False, f"Cannot bind to {host}:{port} - {e}"

def validate_client_config(server_host: str, server_port: int, thread_count: int) -> tuple[bool, str]:
    """Validate client configuration parameters."""
    if not server_host:
        return False, "Server host address cannot be empty"
    
    if not isinstance(server_port, int) or server_port < 1024 or server_port > 65535:
        return False, "Server port must be an integer between 1024 and 65535"
    
    if not isinstance(thread_count, int) or thread_count < 1 or thread_count > 32:
        return False, "Thread count must be an integer between 1 and 32"
    
    return True, "Configuration valid"

def validate_job_data(job_data: Dict) -> tuple[bool, str]:
    """Validate processing job data."""
    required_fields = ['job_id', 'image_path', 'output_path', 'swatches', 'settings']
    
    for field in required_fields:
        if field not in job_data:
            return False, f"Missing required field: {field}"
    
    if not os.path.exists(job_data['image_path']):
        return False, f"Input file does not exist: {job_data['image_path']}"
    
    if not isinstance(job_data['swatches'], list) or len(job_data['swatches']) != 24:
        return False, "Swatches must be a list of 24 color values"
    
    return True, "Job data valid"

def convert_to_unc_path(local_path: str) -> str:
    """
    Convert a local or mapped network path to its UNC network path.
    For example, 'Z:\\folder\\file.txt' -> '\\\\192.168.1.47\\sharename\\folder\\file.txt'
    """
    if not local_path:
        return local_path

    local_path = os.path.abspath(local_path)
    path_obj = Path(local_path)

    # If already UNC path, return as-is
    if local_path.startswith('\\\\'):
        return local_path

    # Try to convert using WNetGetUniversalName if available
    if win32wnet:
        try:
            unc_path = win32wnet.WNetGetUniversalName(local_path)
            return unc_path
        except Exception as e:
            # Fallback: return the original path if not a mapped drive or error occurred
            pass
    
    # Fallback when win32wnet is not available or conversion fails
    return local_path


class MemoryManager:
    """Manages memory monitoring and load balancing for the server."""
    
    def __init__(self, log_callback: Optional[Callable] = None, config: Optional[Dict] = None):
        self.log_callback = log_callback or print
        
        # Default configuration
        default_config = {
            'memory_check_interval': 30.0,  # Check every 30 seconds
            'global_memory_threshold': 0.85,  # Don't assign jobs if system memory > 85%
            'client_memory_threshold': 0.80,  # Don't assign jobs if client memory > 80%
            'memory_per_job_gb': 6.0,  # Default memory estimate per job
            'minimum_free_memory_gb': 2.0,  # Minimum free memory to reserve
            'memory_warning_threshold': 0.80,  # Warn when memory usage exceeds this
            'memory_critical_threshold': 0.90,  # Critical alert when memory exceeds this
            'enable_memory_balancing': True,  # Enable memory-based load balancing
            'dynamic_memory_estimation': True  # Estimate memory based on file size
        }
        
        # Merge with provided config
        self.config = {**default_config, **(config or {})}
        
        # Extract commonly used values
        self.memory_check_interval = self.config['memory_check_interval']
        self.global_memory_threshold = self.config['global_memory_threshold']
        self.client_memory_threshold = self.config['client_memory_threshold']
        self.memory_per_job_gb = self.config['memory_per_job_gb']
        self.minimum_free_memory_gb = self.config['minimum_free_memory_gb']
        
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information in GB."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except Exception as e:
            self.log_callback(f"[MemoryManager] Error getting memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0}
    
    def can_assign_job(self, client_info: ClientInfo) -> tuple[bool, str]:
        """Check if a job can be assigned to a client based on memory constraints."""
        # Skip memory checks if balancing is disabled
        if not self.config['enable_memory_balancing']:
            return True, "Memory balancing disabled"
        
        # Check client memory if reported recently (within 2 minutes)
        if (time.time() - client_info.last_memory_update) < 120:
            required_memory = client_info.memory_per_job_gb + self.minimum_free_memory_gb
            if client_info.available_memory_gb < required_memory:
                return False, f"Client memory too low: {client_info.available_memory_gb:.1f}GB available, need {required_memory:.1f}GB"
            
            memory_health = client_info.memory_health_status()
            if memory_health == "critical":
                return False, f"Client memory critical: {client_info.available_memory_gb:.1f}GB available"
        
        # Check server memory
        server_memory = self.get_system_memory_info()
        if server_memory['percent'] > (self.global_memory_threshold * 100):
            return False, f"Server memory high: {server_memory['percent']:.1f}% used"
        
        return True, "Memory constraints satisfied"
    
    def estimate_job_memory_requirements(self, job: ProcessingJob) -> float:
        """Estimate memory requirements for a specific job in GB."""
        # Base memory estimate
        base_memory = self.memory_per_job_gb
        
        # Skip dynamic estimation if disabled
        if not self.config['dynamic_memory_estimation']:
            return base_memory
        
        # Adjust based on image file size if available
        try:
            if os.path.exists(job.image_path):
                file_size_mb = os.path.getsize(job.image_path) / (1024**2)
                # RAW files typically expand 3-5x in memory during processing
                # Add overhead for color correction matrices and intermediate buffers
                expansion_factor = 4.5  # Conservative estimate
                estimated_memory_gb = (file_size_mb * expansion_factor) / 1024
                
                # Use the higher of base estimate or file-based estimate
                final_estimate = max(base_memory, estimated_memory_gb)
                
                # Cap at reasonable maximum (prevent runaway estimates)
                max_memory_per_job = base_memory * 3  # 3x base as maximum
                return min(final_estimate, max_memory_per_job)
        except Exception as e:
            self.log_callback(f"[MemoryManager] Error estimating memory for {job.image_path}: {e}")
        
        return base_memory
    
    def get_memory_config_summary(self) -> str:
        """Get a human-readable summary of memory configuration."""
        return f"""Memory Configuration:
  - Memory balancing: {'enabled' if self.config['enable_memory_balancing'] else 'disabled'}
  - Memory per job: {self.memory_per_job_gb:.1f}GB
  - Minimum free memory: {self.minimum_free_memory_gb:.1f}GB
  - Server threshold: {self.global_memory_threshold*100:.0f}%
  - Client threshold: {self.client_memory_threshold*100:.0f}%
  - Dynamic estimation: {'enabled' if self.config['dynamic_memory_estimation'] else 'disabled'}"""


class ProcessingServer:
    """Server that distributes image processing jobs to connected clients."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8888, log_callback: Optional[Callable] = None, log_level: LogLevel = LogLevel.INFO):
        # Validate configuration
        is_valid, error_msg = validate_server_config(host, port)
        if not is_valid:
            raise ValueError(f"Invalid server configuration: {error_msg}")
        
        self.host = host
        self.port = port
        self.log_callback = log_callback or print
        self.log_level = log_level
        
        self.server_socket = None
        self.running = False
        self.clients: Dict[str, ClientInfo] = {}
        self.job_queue = Queue()
        self.completed_jobs: Dict[str, Dict] = {}
        self.failed_jobs: Dict[str, Dict] = {}
        self.active_jobs: Dict[str, ProcessingJob] = {}  # Track jobs currently being processed
        self.recent_completed_jobs: List[Dict] = []  # Track last 5 completed jobs with timing data
        self.recent_completed_groups: List[Dict] = []  # Track last 5 completed job groups with timing data
        
        # Job group tracking for web interface
        self.job_groups: Dict[str, Dict] = {}  # Track job groups and their images
        self.job_to_group: Dict[str, str] = {}  # Map individual job IDs to group IDs
        
        # Event to trigger immediate job distribution check
        self._job_distribution_event = threading.Event()
        
        # Memory management
        memory_config = {
            'enable_memory_balancing': True,
            'memory_per_job_gb': 6.0,
            'global_memory_threshold': 0.85,
            'client_memory_threshold': 0.80,
            'minimum_free_memory_gb': 2.0
        }
        self.memory_manager = MemoryManager(log_callback, memory_config)
        
        # Log memory configuration
        self.log_info("Memory Management Initialized:")
        for line in self.memory_manager.get_memory_config_summary().split('\n'):
            self.log_info(f"  {line}")
        
        # Threading
        self.server_thread = None
        self.heartbeat_thread = None
        self.job_monitor_thread = None
        self.timeout_monitor_thread = None
        self.memory_monitor_thread = None
        
        # Settings
        self.process_on_host = False
    
    def update_memory_config(self, config_updates: Dict) -> bool:
        """Update memory management configuration at runtime."""
        try:
            old_config = self.memory_manager.config.copy()
            self.memory_manager.config.update(config_updates)
            
            # Update commonly used cached values
            self.memory_manager.memory_check_interval = self.memory_manager.config['memory_check_interval']
            self.memory_manager.global_memory_threshold = self.memory_manager.config['global_memory_threshold']
            self.memory_manager.client_memory_threshold = self.memory_manager.config['client_memory_threshold']
            self.memory_manager.memory_per_job_gb = self.memory_manager.config['memory_per_job_gb']
            self.memory_manager.minimum_free_memory_gb = self.memory_manager.config['minimum_free_memory_gb']
            
            self.log_info(f"Memory configuration updated: {config_updates}")
            self.log_info("Updated Memory Configuration:")
            for line in self.memory_manager.get_memory_config_summary().split('\n'):
                self.log_info(f"  {line}")
            
            return True
        except Exception as e:
            self.log_error(f"Failed to update memory configuration: {e}")
            self.memory_manager.config = old_config  # Restore old config
            return False
        
    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Log message with timestamp and level."""
        if level.value >= self.log_level.value:
            timestamp = datetime.now().strftime("%H:%M:%S")
            level_name = level.name
            formatted_msg = f"[Server {timestamp}] [{level_name}] {message}"
            if self.log_callback:
                self.log_callback(formatted_msg)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def log_info(self, message: str):
        """Log info message."""
        self.log(message, LogLevel.INFO)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.log(message, LogLevel.WARNING)
    
    def log_error(self, message: str):
        """Log error message."""
        self.log(message, LogLevel.ERROR)
    
    def log_critical(self, message: str):
        """Log critical message."""
        self.log(message, LogLevel.CRITICAL)
    
    def start(self):
        """Start the server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            
            self.running = True
            self.log_info(f"Server started on {self.host}:{self.port}")
            
            # Start background threads
            self.server_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
            self.job_monitor_thread = threading.Thread(target=self._job_monitor, daemon=True)
            self.timeout_monitor_thread = threading.Thread(target=self._timeout_monitor, daemon=True)
            
            self.server_thread.start()
            self.heartbeat_thread.start()
            self.job_monitor_thread.start()
            self.timeout_monitor_thread.start()
            
            # Start memory monitoring thread
            self.memory_monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
            self.memory_monitor_thread.start()
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to start server: {e}")
            return False
    
    def _trigger_job_distribution(self):
        """Trigger immediate job distribution check."""
        self._job_distribution_event.set()
    
    def _update_job_group_status(self, job_id: str, new_status: str):
        """Update job group status when an individual job changes state."""
        if job_id not in self.job_to_group:
            return
        
        group_id = self.job_to_group[job_id]
        if group_id not in self.job_groups:
            return
        
        group = self.job_groups[group_id]
        
        # Count current job states
        active_count = 0
        completed_count = 0
        failed_count = 0
        queued_count = 0
        
        for image_job_id in group['image_jobs']:
            if image_job_id in self.active_jobs:
                active_count += 1
            elif image_job_id in self.completed_jobs:
                completed_count += 1
            elif image_job_id in self.failed_jobs:
                failed_count += 1
            else:
                queued_count += 1
        
        # Update group counters
        group['active_count'] = active_count
        group['completed_count'] = completed_count
        group['failed_count'] = failed_count
        group['queued_count'] = queued_count
        
        # Determine overall group status
        total_jobs = len(group['image_jobs'])
        old_status = group.get('status', 'queued')
        
        if completed_count == total_jobs:
            group['status'] = 'completed'
        elif failed_count == total_jobs:
            group['status'] = 'failed'
        elif (completed_count + failed_count) == total_jobs:
            group['status'] = 'completed_with_errors'
        elif active_count > 0:
            group['status'] = 'processing'
        else:
            group['status'] = 'queued'
        
        # Track newly completed groups for recent completed groups list
        new_status = group['status']
        if (old_status != 'completed' and new_status == 'completed') or \
           (old_status != 'completed_with_errors' and new_status == 'completed_with_errors'):
            
            # Calculate group processing time
            group_start_time = group.get('created_time', time.time())
            group_end_time = time.time()
            group_processing_time = group_end_time - group_start_time
            
            # Calculate average processing time per image
            processed_images = completed_count + failed_count
            avg_time_per_image = group_processing_time / max(1, processed_images)
            
            # Add to recent completed groups tracking
            completed_group_info = {
                'group_id': group_id,
                'group_name': group.get('name', f'Group {group_id[:8]}'),
                'total_images': total_jobs,
                'completed_images': completed_count,
                'failed_images': failed_count,
                'status': new_status,
                'created_time': group_start_time,
                'completed_time': group_end_time,
                'processing_time': group_processing_time,
                'avg_time_per_image': avg_time_per_image
            }
            
            self.recent_completed_groups.append(completed_group_info)
            # Keep only the last 5 completed groups
            if len(self.recent_completed_groups) > 5:
                self.recent_completed_groups.pop(0)
            
            self.log_info(f"Job group '{group.get('name', group_id)}' completed: {completed_count}/{total_jobs} successful, {failed_count} failed")

    def restart_client(self, client_id: str) -> bool:
        """Send restart command to a specific client."""
        if client_id not in self.clients:
            self.log_error(f"Cannot restart client {client_id}: not found")
            return False
        
        client_info = self.clients[client_id]
        if not client_info.socket:
            self.log_error(f"Cannot restart client {client_id}: no active connection")
            return False
        
        try:
            # Send restart command to client
            restart_msg = NetworkProtocol.encode_message('restart_client', {
                'reason': 'Manual restart requested from server',
                'timestamp': time.time()
            })
            client_info.socket.send(restart_msg)
            
            self.log_info(f"Sent restart command to client {client_id} ({client_info.address})")
            
            # Mark client as disconnected (it will reconnect after restart)
            client_info.status = 'restarting'
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to send restart command to client {client_id}: {e}")
            return False
    
    def send_thread_count_update(self, client_id: str, new_thread_count: int) -> bool:
        """Send thread count update command to a specific client."""
        if client_id not in self.clients:
            self.log_error(f"Cannot update thread count for client {client_id}: not found")
            return False
        
        client_info = self.clients[client_id]
        if not client_info.socket:
            self.log_error(f"Cannot update thread count for client {client_id}: no active connection")
            return False
        
        try:
            # Send thread count update command to client
            thread_msg = NetworkProtocol.encode_message('set_thread_count', {
                'new_thread_count': new_thread_count,
                'timestamp': time.time()
            })
            client_info.socket.send(thread_msg)
            
            self.log_info(f"Sent thread count update ({new_thread_count}) to client {client_id} ({client_info.address})")
            
            # Update the stored thread count immediately
            client_info.thread_count = new_thread_count
            return True
            
        except Exception as e:
            self.log_error(f"Failed to send thread count update to client {client_id}: {e}")
            return False
    
    def stop(self):
        """Stop the server."""
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            self.clients[client_id].status = 'disconnected'
        
        self.log("Server stopped")
    
    def add_job(self, job: ProcessingJob):
        """Add a processing job to the queue."""
        # Validate job data
        job_dict = job.to_dict()
        is_valid, error_msg = validate_job_data(job_dict)
        if not is_valid:
            self.log_error(f"Invalid job data for {job.job_id}: {error_msg}")
            self.failed_jobs[job.job_id] = {
                'job_id': job.job_id,
                'error': f'Job validation failed: {error_msg}',
                'validation_failed': True
            }
            return False
        
        # Convert paths to UNC for network access
        job.image_path = convert_to_unc_path(job.image_path)
        job.output_path = convert_to_unc_path(job.output_path)
        
        self.job_queue.put(job)
        self.log_info(f"Job {job.job_id} added to queue")
        return True

    def add_project_jobs(self, project_data: Dict) -> Dict:
        """
        Process project data and create individual jobs for each image.
        
        Args:
            project_data: Complete project data with images, groups, and settings
            
        Returns:
            dict: Summary of jobs created
        """
        try:
            # Extract components from project data
            images = project_data.get('images', [])
            image_groups = project_data.get('image_groups', {})
            processing_settings = project_data.get('processing_settings', {})
            
            # Filter to only processable images (not group headers)
            processable_images = []
            for image in images:
                metadata = image.get('metadata', {})
                if not metadata.get('is_group_header', False) and image.get('full_path'):
                    processable_images.append(image)
            
            # Create a unique job group ID
            job_group_id = f"group_{int(time.time() * 1000)}"
            
            # Use root_folder from processing_settings as the group name, fallback to project_name, then generic name
            root_folder = processing_settings.get('root_folder', '')
            if root_folder:
                # Use just the folder name (basename) if it's a path
                job_group_name = os.path.basename(root_folder) if root_folder else f'Job Group {len(self.job_groups) + 1}'
            else:
                job_group_name = project_data.get('project_name', f'Job Group {len(self.job_groups) + 1}')
            
            # Initialize job group tracking
            self.job_groups[job_group_id] = {
                'group_id': job_group_id,
                'name': job_group_name,
                'total_images': len(processable_images),
                'created_time': time.time(),
                'completed_count': 0,
                'failed_count': 0,
                'active_count': 0,
                'queued_count': len(processable_images),
                'status': 'queued',
                'image_jobs': [],  # List of individual job IDs
                'processing_settings': processing_settings.copy(),
                'image_groups': image_groups.copy()
            }
            
            jobs_created = []
            jobs_failed = []
            
            # Track sequence numbers per group for proper numbering
            group_sequence_counters = {}
            
            for image_data in processable_images:
                try:
                    # Get chart swatches for this image's group
                    group_name = image_data.get('group', 'All Images')
                    group_data = image_groups.get(group_name, {})
                    chart_swatches = group_data.get('chart_swatches', [])
                    
                    # Validate chart swatches
                    if not chart_swatches or len(chart_swatches) != 24:
                        error_msg = f"Group '{group_name}' has invalid chart swatches (length: {len(chart_swatches)})"
                        jobs_failed.append({
                            'image_path': image_data.get('full_path', 'unknown'),
                            'group': group_name,
                            'error': error_msg
                        })
                        self.log_warning(f"[Project] {error_msg}")
                        continue
                    
                    # Track sequence number per group
                    if group_name not in group_sequence_counters:
                        group_sequence_counters[group_name] = 0
                    group_sequence_counters[group_name] += 1
                    current_sequence_number = group_sequence_counters[group_name]
                    
                    # Add sequence number to image metadata for output path generation
                    image_data_with_sequence = image_data.copy()
                    if 'metadata' not in image_data_with_sequence:
                        image_data_with_sequence['metadata'] = {}
                    image_data_with_sequence['metadata']['sequence_number'] = current_sequence_number
                    
                    # Create job ID
                    job_id = f"proj_{int(time.time() * 1000)}_{len(jobs_created)}"
                    
                    # Generate output path with proper sequence number
                    output_path = self._generate_output_path_from_settings(
                        image_data_with_sequence, processing_settings
                    )
                    
                    # Create ProcessingJob
                    job = ProcessingJob(
                        job_id=job_id,
                        image_path=image_data['full_path'],
                        output_path=output_path,
                        swatches=chart_swatches,
                        settings=processing_settings,
                        created_time=time.time()
                    )
                    
                    # Add job to queue using existing method
                    if self.add_job(job):
                        # Link job to group
                        self.job_to_group[job_id] = job_group_id
                        self.job_groups[job_group_id]['image_jobs'].append(job_id)
                        
                        jobs_created.append({
                            'job_id': job_id,
                            'image_path': job.image_path,
                            'output_path': job.output_path,
                            'group': group_name,
                            'job_group_id': job_group_id
                        })
                    else:
                        jobs_failed.append({
                            'image_path': image_data.get('full_path', 'unknown'),
                            'group': group_name,
                            'error': 'Failed to add job to queue'
                        })
                        
                except Exception as e:
                    jobs_failed.append({
                        'image_path': image_data.get('full_path', 'unknown'),
                        'group': image_data.get('group', 'unknown'),
                        'error': str(e)
                    })
                    self.log_error(f"[Project] Error creating job for image: {e}")
            
            # Update job group status
            self.job_groups[job_group_id]['failed_count'] = len(jobs_failed)
            self.job_groups[job_group_id]['queued_count'] = len(jobs_created)
            
            if len(jobs_created) == 0:
                self.job_groups[job_group_id]['status'] = 'failed'
            elif len(jobs_failed) > 0:
                self.job_groups[job_group_id]['status'] = 'partial'
            else:
                self.job_groups[job_group_id]['status'] = 'queued'
            
            # Return summary
            summary = {
                'success': True,
                'jobs_created': len(jobs_created),
                'jobs_failed': len(jobs_failed),
                'created_jobs': jobs_created,
                'failed_jobs': jobs_failed,
                'total_images': len(processable_images),
                'job_group_id': job_group_id,
                'job_group_name': job_group_name
            }
            
            self.log_info(f"[Project] Created job group '{job_group_name}' with {len(jobs_created)} jobs, {len(jobs_failed)} failed")
            return summary
            
        except Exception as e:
            self.log_error(f"[Project] Error processing project data: {e}")
            return {
                'success': False,
                'error': str(e),
                'jobs_created': 0,
                'jobs_failed': 0
            }

    def _generate_output_path_from_settings(self, image_data: Dict, processing_settings: Dict) -> str:
        """Generate output path for an image based on processing settings."""
        import os
        from ImageProcessor.fileNamingSchema import apply_naming_schema
        
        input_path = image_data['full_path']
        output_dir = processing_settings.get('output_directory', os.path.dirname(input_path))
        output_format = processing_settings.get('export_format', '.jpg')
        
        # Check if export schema should be used
        use_export_schema = processing_settings.get('use_export_schema', False)
        export_schema = processing_settings.get('export_schema', '')
        
        if use_export_schema and export_schema:
            try:
                # Get additional parameters for schema
                custom_name = processing_settings.get('custom_name', '')
                root_folder = processing_settings.get('root_folder', '')
                
                # Generate image sequence number - use metadata if available, otherwise default to 1
                image_number = image_data.get('metadata', {}).get('sequence_number', 1)
                
                # Determine group name from image data
                group_name = image_data.get('group', '')
                
                # Apply naming schema
                output_path = apply_naming_schema(
                    schema=export_schema,
                    input_path=input_path,
                    output_base_dir=output_dir,
                    custom_name=custom_name,
                    image_number=image_number,
                    output_extension=output_format,
                    root_folder=root_folder,
                    group_name=group_name
                )
                
                return convert_to_unc_path(output_path)
                
            except Exception as e:
                print(f"[Server] Schema Error: Failed to apply naming schema for {input_path}: {e}")
                # Fall through to default naming
        
        # Default/fallback path generation
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}{output_format}")
        
        return convert_to_unc_path(output_path)
    
    def clear_job_queue(self):
        """Clear all pending jobs from the queue."""
        cleared_count = 0
        try:
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                    cleared_count += 1
                except Empty:
                    break
        except Exception as e:
            self.log_error(f"Error clearing job queue: {e}")
        
        if cleared_count > 0:
            self.log_info(f"Cleared {cleared_count} pending jobs from queue")
        else:
            self.log_debug("Job queue was already empty")
    
    def stop_job_group(self, group_id: str) -> Dict:
        """
        Stop a job group by removing all its jobs from the queue and marking as stopped.
        
        Args:
            group_id: The job group ID to stop
            
        Returns:
            dict: Summary of the stop operation
        """
        if group_id not in self.job_groups:
            return {
                'success': False,
                'error': f'Job group {group_id} not found'
            }
        
        job_group = self.job_groups[group_id]
        
        # If already completed, can't stop
        if job_group['status'] in ['completed', 'stopped', 'failed']:
            return {
                'success': False,
                'error': f'Job group {group_id} is already {job_group["status"]}'
            }
        
        try:
            # Get all job IDs in this group
            group_job_ids = set(job_group.get('image_jobs', []))
            
            # Remove queued jobs from the queue
            remaining_jobs = []
            removed_from_queue = 0
            
            # Drain the queue and rebuild it without this group's jobs
            temp_jobs = []
            try:
                while not self.job_queue.empty():
                    try:
                        job = self.job_queue.get_nowait()
                        if job.job_id in group_job_ids:
                            removed_from_queue += 1
                            # Remove from job mapping
                            if job.job_id in self.job_to_group:
                                del self.job_to_group[job.job_id]
                        else:
                            temp_jobs.append(job)
                    except Empty:
                        break
                
                # Put back the jobs that weren't removed
                for job in temp_jobs:
                    self.job_queue.put(job)
                    
            except Exception as e:
                self.log_error(f"Error processing job queue during group stop: {e}")
            
            # Mark any active jobs from this group as cancelled
            cancelled_active = 0
            for job_id in list(self.active_jobs.keys()):
                if job_id in group_job_ids:
                    # Remove from active jobs
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]
                        cancelled_active += 1
                    
                    # Remove from client assignments
                    for client_info in self.clients.values():
                        if job_id in client_info.current_jobs:
                            client_info.remove_job(job_id)
                    
                    # Remove from job mapping
                    if job_id in self.job_to_group:
                        del self.job_to_group[job_id]
            
            # Update job group status
            job_group['status'] = 'stopped'
            job_group['stopped_time'] = time.time()
            job_group['active_count'] = 0
            job_group['queued_count'] = 0
            
            # Log the operation
            total_stopped = removed_from_queue + cancelled_active
            self.log_info(f"Stopped job group '{job_group['name']}' ({group_id}): "
                         f"{removed_from_queue} queued jobs removed, {cancelled_active} active jobs cancelled")
            
            # Trigger job distribution check for remaining jobs
            self._trigger_job_distribution()
            
            return {
                'success': True,
                'group_id': group_id,
                'group_name': job_group['name'],
                'jobs_removed_from_queue': removed_from_queue,
                'active_jobs_cancelled': cancelled_active,
                'total_jobs_stopped': total_stopped,
                'message': f"Successfully stopped job group '{job_group['name']}'"
            }
            
        except Exception as e:
            self.log_error(f"Error stopping job group {group_id}: {e}")
            return {
                'success': False,
                'error': f'Failed to stop job group: {str(e)}'
            }
    
    def get_status(self) -> Dict:
        """Get server status information."""
        active_clients = [c for c in self.clients.values() if c.status != 'disconnected']
        total_capacity = sum(c.thread_count for c in active_clients)
        used_capacity = sum(len(c.current_jobs) for c in active_clients)
        
        # Memory status
        server_memory = self.memory_manager.get_system_memory_info()
        total_client_memory = sum(c.total_memory_gb for c in active_clients if c.total_memory_gb > 0)
        available_client_memory = sum(c.available_memory_gb for c in active_clients if c.available_memory_gb > 0)
        
        return {
            'running': self.running,
            'clients': len(active_clients),
            'total_capacity': total_capacity,
            'used_capacity': used_capacity,
            'available_capacity': total_capacity - used_capacity,
            'queue_size': self.job_queue.qsize(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'memory_status': {
                'server_memory_percent': server_memory['percent'],
                'server_available_gb': server_memory['available_gb'],
                'total_client_memory_gb': total_client_memory,
                'available_client_memory_gb': available_client_memory,
                'memory_balancing_enabled': self.memory_manager.config['enable_memory_balancing']
            }
        }
    
    def _validate_client_jobs(self):
        """Validate and clean up stale job assignments across all clients."""
        total_stale_jobs = 0
        
        for client_id, client_info in self.clients.items():
            stale_count = client_info.validate_and_cleanup_jobs(self.active_jobs)
            if stale_count > 0:
                total_stale_jobs += stale_count
                self.log_info(f"Cleaned up {stale_count} stale job assignments for client {client_id}")
        
        if total_stale_jobs > 0:
            self.log_info(f"Total stale job cleanup: {total_stale_jobs} phantom assignments removed")
    
    def _accept_connections(self):
        """Accept incoming client connections."""
        while self.running:
            try:
                if not self.server_socket:
                    break
                    
                client_socket, address = self.server_socket.accept()
                
                # Enable TCP keepalive on client connections
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                try:
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                except (AttributeError, OSError):
                    pass
                
                self.log_info(f"Client connected from {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    self.log(f"Error accepting connection: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connection."""
        client_id = f"{address[0]}:{address[1]}_{int(time.time())}"
        
        try:
            # Initial handshake
            msg_type, data = NetworkProtocol.decode_message(client_socket)
            
            if msg_type == 'client_register':
                # Always clean up stale clients from same address or with same name
                # This handles both explicit reconnections and accidental duplicates
                client_name = data.get('client_name', data.get('hostname', None))
                reconnect = data.get('reconnect', False)
                
                cleanup_count = self._cleanup_stale_clients(address[0], client_name)
                if cleanup_count > 0:
                    self.log_info(f"Cleaned up {cleanup_count} stale client registration(s) for {address[0]}")
                
                client_info = ClientInfo(
                    client_id=client_id,
                    address=address[0],
                    port=address[1],
                    thread_count=data.get('thread_count', 1),
                    last_heartbeat=time.time(),
                    status='idle',
                    socket=client_socket
                )
                
                # Store client name if provided
                if client_name:
                    client_info.client_name = client_name
                
                self.clients[client_id] = client_info
                self.log_info(f"Client {client_id} registered with {client_info.thread_count} threads{'(reconnect)' if reconnect else ''}")
                
                # If this is a reconnection, ensure the client starts with a clean job slate
                if reconnect:
                    client_info.current_jobs.clear()
                    client_info.status = 'idle'
                    # Reset heartbeat timestamp to prevent immediate timeout
                    client_info.last_heartbeat = time.time()
                    self.log_info(f"Cleared job assignments for reconnecting client {client_id}")
                
                # Always ensure new/reconnected clients are marked as available
                client_info.status = 'idle'
                client_info.last_heartbeat = time.time()
                
                # Send registration confirmation
                response = NetworkProtocol.encode_message('register_ack', {'client_id': client_id})
                client_socket.send(response)
            
            # Handle client communication
            while self.running:
                try:
                    msg_type, data = NetworkProtocol.decode_message(client_socket, timeout=5.0)
                    
                    if msg_type == 'heartbeat':
                        # Check if this client is still registered
                        if client_id not in self.clients:
                            self.log_warning(f"Received heartbeat from unregistered client {client_id}, requesting re-registration")
                            # Send client_unknown message to trigger re-registration
                            try:
                                unknown_msg = NetworkProtocol.encode_message('client_unknown', {'client_id': client_id})
                                client_socket.send(unknown_msg)
                            except Exception as send_e:
                                self.log_error(f"Failed to send client_unknown message: {send_e}")
                            break  # Close this connection
                        
                        client_info = self.clients[client_id]
                        client_info.last_heartbeat = time.time()
                        client_info.status = data.get('status', 'idle')
                        
                        # Update memory information if provided
                        if 'memory_info' in data:
                            memory_info = data['memory_info']
                            client_info.total_memory_gb = memory_info.get('total_gb', 0.0)
                            client_info.available_memory_gb = memory_info.get('available_gb', 0.0)
                            client_info.last_memory_update = time.time()
                        
                        # Handle post-reconnection heartbeat
                        is_reconnected = data.get('reconnected', False)
                        if is_reconnected:
                            # Immediately validate and clean up any stale job assignments for this client
                            stale_count = client_info.validate_and_cleanup_jobs(self.active_jobs)
                            if stale_count > 0:
                                self.log_info(f"Cleaned up {stale_count} stale job assignments for reconnected client {client_id}")
                            
                            # Ensure client is marked as idle and ready for jobs
                            client_info.status = 'idle'
                            client_info.current_jobs.clear()  # Clear any lingering job references
                            client_info.last_heartbeat = time.time()  # Update heartbeat timestamp
                            
                            self.log_info(f"Client {client_id} sent post-reconnection heartbeat - now available for jobs")
                            
                            # Trigger job distribution check immediately for reconnected clients
                            self._trigger_job_distribution()
                        
                        # Log capacity information for debugging
                        active_jobs = data.get('active_jobs', 0)
                        max_jobs = data.get('max_jobs', client_info.thread_count)
                        memory_status = client_info.memory_health_status()
                        available_capacity = client_info.available_capacity()
                        
                        log_level = self.log_info if is_reconnected else self.log_debug
                        log_level(f"Client {client_id} heartbeat: {active_jobs}/{max_jobs} jobs, capacity: {available_capacity}, memory: {client_info.available_memory_gb:.1f}GB ({memory_status})")
                        
                    elif msg_type == 'job_complete':
                        job_id = data['job_id']
                        # Add image_path from active job before cleanup
                        if job_id in self.active_jobs:
                            data['image_path'] = self.active_jobs[job_id].image_path
                            # Extract timing information for recent completed jobs tracking
                            active_job = self.active_jobs[job_id]
                            image_name = os.path.basename(active_job.image_path) if active_job.image_path else 'Unknown'
                            output_format = active_job.settings.get('output_format', 'JPEG')
                            
                            # Calculate processing time
                            end_time = time.time()
                            start_time = active_job.start_time if active_job.start_time else active_job.created_time
                            processing_time = data.get('processing_time', end_time - start_time)
                            
                            # Add to recent completed jobs tracking (keep last 5)
                            completed_job_info = {
                                'job_id': job_id,
                                'job_name': f"Job {job_id[:8]}",
                                'image_name': image_name,
                                'output_format': output_format.upper().replace('.', ''),
                                'assigned_client': client_id,
                                'completed_at': end_time,
                                'start_time': start_time,
                                'end_time': end_time,
                                'processing_time': processing_time
                            }
                            
                            self.recent_completed_jobs.append(completed_job_info)
                            # Keep only the last 5 completed jobs
                            if len(self.recent_completed_jobs) > 5:
                                self.recent_completed_jobs.pop(0)
                        
                        self.completed_jobs[job_id] = data
                        self.clients[client_id].remove_job(job_id)
                        # Clean up active job tracking
                        if job_id in self.active_jobs:
                            del self.active_jobs[job_id]
                        # Update job group status
                        self._update_job_group_status(job_id, 'completed')
                        self.log_info(f"Job {job_id} completed by {client_id}")
                        
                    elif msg_type == 'job_failed':
                        job_id = data['job_id']
                        # Add image_path from active job before cleanup
                        if job_id in self.active_jobs:
                            data['image_path'] = self.active_jobs[job_id].image_path
                        self.failed_jobs[job_id] = data
                        self.clients[client_id].remove_job(job_id)
                        # Clean up active job tracking
                        if job_id in self.active_jobs:
                            del self.active_jobs[job_id]
                        # Update job group status
                        self._update_job_group_status(job_id, 'failed')
                        self.log_error(f"Job {job_id} failed on {client_id}: {data.get('error', 'Unknown error')}")
                    
                    elif msg_type == 'job_progress':
                        job_id = data['job_id']
                        status = data.get('status', '')
                        self.log_debug(f"Job {job_id} progress from {client_id}: {status}")
                        # Could store progress updates for monitoring dashboard
                
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log(f"Error handling client {client_id}: {e}")
                    break
        
        except Exception as e:
            self.log(f"Client {client_id} connection error: {e}")
        
        finally:
            # Cleanup
            if client_id in self.clients:
                self.clients[client_id].status = 'disconnected'
            
            try:
                client_socket.close()
            except:
                pass
            
            self.log(f"Client {client_id} disconnected")
    
    def _is_socket_connected(self, sock):
        """
        Test if a socket is still connected.
        
        Args:
            sock: Socket to test
            
        Returns:
            bool: True if socket appears to be connected
        """
        try:
            # Use SO_ERROR to check socket status
            error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if error != 0:
                return False
            
            # Try a non-blocking send of zero bytes to test connection
            sock.setblocking(False)
            try:
                sock.send(b'')
                return True
            except BlockingIOError:
                # This is expected for non-blocking mode, means socket is writable
                return True
            except (socket.error, OSError):
                return False
            finally:
                # Restore blocking mode
                sock.setblocking(True)
                
        except (socket.error, OSError):
            return False

    def _cleanup_stale_clients(self, client_address, client_name=None):
        """
        Clean up stale client registrations from the same address or with the same name.
        
        Args:
            client_address: IP address of the reconnecting client
            client_name: Optional client name/hostname for additional matching
        """
        stale_clients = []
        
        for client_id, client_info in self.clients.items():
            is_same_address = client_info.address == client_address
            is_same_name = (client_name and 
                           hasattr(client_info, 'client_name') and 
                           client_info.client_name == client_name)
            
            # Consider a client stale if:
            # 1. Same IP address AND (disconnected OR socket not connected)
            # 2. Same client name AND (disconnected OR socket not connected) 
            # 3. Same IP address AND very old connection (>5 minutes without heartbeat)
            is_disconnected = (client_info.status == 'disconnected' or 
                             not self._is_socket_connected(client_info.socket))
            is_very_old = (time.time() - client_info.last_heartbeat) > 300  # 5 minutes
            
            should_cleanup = ((is_same_address or is_same_name) and 
                            (is_disconnected or is_very_old))
            
            if should_cleanup:
                stale_clients.append(client_id)
        
        cleanup_count = len(stale_clients)
        if cleanup_count > 0:
            self.log_info(f"Found {cleanup_count} stale client(s) to cleanup for address {client_address}")
        
        for client_id in stale_clients:
            client_info = self.clients[client_id]
            self.log_info(f"Cleaning up stale client: {client_id} (address: {client_info.address}, last_heartbeat: {time.time() - client_info.last_heartbeat:.1f}s ago)")
            
            # Update job group status for any jobs being requeued
            jobs_to_requeue = client_info.current_jobs.copy()
            for job_id in jobs_to_requeue:
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    self.job_queue.put(job)  # Put job back in queue
                    del self.active_jobs[job_id]
                    # Update job group status
                    self._update_job_group_status(job_id, 'queued')
                    self.log_info(f"Requeued job {job_id} from stale client {client_id}")
                client_info.remove_job(job_id)
            
            # Ensure job list is completely cleared
            client_info.current_jobs.clear()
            
            # Remove the stale client from registry
            del self.clients[client_id]
        
        return cleanup_count

    def _cleanup_duplicate_clients(self):
        """Remove duplicate client entries with the same IP address, keeping the most recent."""
        try:
            address_to_clients = {}
            
            # Group clients by IP address
            for client_id, client_info in self.clients.items():
                address = client_info.address
                if address not in address_to_clients:
                    address_to_clients[address] = []
                address_to_clients[address].append((client_id, client_info))
            
            duplicates_removed = 0
            
            # For each address with multiple clients, keep only the most recent
            for address, client_list in address_to_clients.items():
                if len(client_list) > 1:
                    # Sort by last heartbeat time (most recent first)
                    client_list.sort(key=lambda x: x[1].last_heartbeat, reverse=True)
                    
                    # Keep the first (most recent), remove the rest
                    clients_to_remove = client_list[1:]
                    
                    for client_id, client_info in clients_to_remove:
                        # Only remove if it's actually stale (no recent activity)
                        time_since_heartbeat = time.time() - client_info.last_heartbeat
                        if time_since_heartbeat > 30:  # 30 seconds without heartbeat
                            self.log_info(f"Removing duplicate client {client_id} (address: {address}, {time_since_heartbeat:.1f}s since heartbeat)")
                            
                            # Requeue any active jobs
                            jobs_to_requeue = client_info.current_jobs.copy()
                            for job_id in jobs_to_requeue:
                                if job_id in self.active_jobs:
                                    job = self.active_jobs[job_id]
                                    self.job_queue.put(job)
                                    del self.active_jobs[job_id]
                                    self._update_job_group_status(job_id, 'queued')
                                    self.log_info(f"Requeued job {job_id} from duplicate client")
                                client_info.remove_job(job_id)
                            
                            # Remove the duplicate client
                            del self.clients[client_id]
                            duplicates_removed += 1
            
            if duplicates_removed > 0:
                self.log_info(f"Removed {duplicates_removed} duplicate client registration(s)")
                
        except Exception as e:
            self.log_error(f"Error during duplicate client cleanup: {e}")

    def _heartbeat_monitor(self):
        """Monitor client heartbeats and cleanup dead connections."""
        last_duplicate_check = 0
        duplicate_check_interval = 60  # Check for duplicates every minute
        
        while self.running:
            try:
                current_time = time.time()
                dead_clients = []
                
                # Periodically check for and remove duplicate clients
                if current_time - last_duplicate_check > duplicate_check_interval:
                    self._cleanup_duplicate_clients()
                    last_duplicate_check = current_time
                
                for client_id, client_info in self.clients.items():
                    # Check multiple criteria for dead clients
                    is_timed_out = not client_info.is_alive()
                    is_disconnected = client_info.status == 'disconnected'
                    socket_dead = client_info.socket and not self._is_socket_connected(client_info.socket)
                    
                    if (is_timed_out or socket_dead) and not is_disconnected:
                        dead_clients.append(client_id)
                        if socket_dead:
                            self.log_debug(f"Client {client_id} detected as disconnected via socket test")
                
                for client_id in dead_clients:
                    self.log_warning(f"Client {client_id} timed out")
                    client_info = self.clients[client_id]
                    client_info.status = 'disconnected'
                    
                    # Requeue any jobs they were processing
                    jobs_to_requeue = client_info.current_jobs.copy()
                    for job_id in jobs_to_requeue:
                        if job_id in self.active_jobs:
                            job = self.active_jobs[job_id]
                            self.job_queue.put(job)  # Put job back in queue
                            del self.active_jobs[job_id]
                            self.log_info(f"Requeued job {job_id} due to client timeout")
                        client_info.remove_job(job_id)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.log(f"Heartbeat monitor error: {e}")
                time.sleep(5)
    
    def _job_monitor(self):
        """Monitor job queue and assign jobs to available clients."""
        last_validation_time = 0
        
        while self.running:
            try:
                # Wait for either a timeout or trigger event
                triggered = self._job_distribution_event.wait(timeout=5.0)
                
                if triggered:
                    # Clear the event and proceed with immediate job distribution
                    self._job_distribution_event.clear()
                    self.log_debug("Job distribution triggered by reconnection event")
                
                # Periodically validate and clean up stale job assignments (every 30 seconds)
                current_time = time.time()
                if (current_time - last_validation_time) > 30:
                    self._validate_client_jobs()
                    last_validation_time = current_time
                
                # Check for clients with available capacity
                all_clients = list(self.clients.items())
                clients_with_capacity = []
                
                for client_id, client_info in all_clients:
                    capacity = client_info.available_capacity()
                    is_alive = client_info.is_alive()
                    
                    if capacity > 0 and is_alive:
                        clients_with_capacity.append((client_id, client_info))
                    else:
                        # Log why client was excluded for debugging
                        if not is_alive:
                            self.log_debug(f"Client {client_id} excluded: not alive (last heartbeat {time.time() - client_info.last_heartbeat:.1f}s ago)")
                        elif capacity <= 0:
                            self.log_debug(f"Client {client_id} excluded: no capacity ({len(client_info.current_jobs)}/{client_info.thread_count} jobs)")
                
                queue_size = self.job_queue.qsize()
                if clients_with_capacity and queue_size > 0:
                    self.log_debug(f"Job distribution: {len(clients_with_capacity)} clients available, {queue_size} jobs in queue")
                    # Sort clients by available capacity (descending) for better distribution
                    clients_with_capacity.sort(key=lambda x: x[1].available_capacity(), reverse=True)
                    
                    jobs_assigned = 0
                    
                    # Try to assign jobs to all clients with capacity
                    for client_id, client_info in clients_with_capacity:
                        # Assign as many jobs as the client can handle
                        while client_info.available_capacity() > 0 and not self.job_queue.empty():
                            try:
                                job = self.job_queue.get(timeout=0.1)
                                
                                # Check memory constraints before assignment
                                can_assign, reason = self.memory_manager.can_assign_job(client_info)
                                if not can_assign:
                                    self.log_debug(f"Delaying job {job.job_id} to {client_id}: {reason}")
                                    self.job_queue.put(job)  # Put job back in queue
                                    break  # Try next client or wait for next cycle
                                
                                # Update job memory estimate
                                estimated_memory = self.memory_manager.estimate_job_memory_requirements(job)
                                client_info.memory_per_job_gb = estimated_memory
                                
                                # Add job to client tracking
                                if client_info.add_job(job.job_id):
                                    job.start_time = time.time()  # Mark job start time
                                    self.active_jobs[job.job_id] = job  # Track active job
                                    # Update job group status
                                    self._update_job_group_status(job.job_id, 'active')
                                    
                                    # Send job to client via their socket
                                    if hasattr(client_info, 'socket') and client_info.socket:
                                        try:
                                            # Test if socket is still connected before sending job
                                            if not self._is_socket_connected(client_info.socket):
                                                self.log_warning(f"Client {client_id} socket is disconnected, marking as dead")
                                                client_info.status = 'disconnected'
                                                # Clean up failed assignment
                                                client_info.remove_job(job.job_id)
                                                if job.job_id in self.active_jobs:
                                                    del self.active_jobs[job.job_id]
                                                self.job_queue.put(job)  # Put job back in queue
                                                break  # Stop trying to assign to this client
                                            
                                            job_msg = NetworkProtocol.encode_message('process_job', job.to_dict())
                                            client_info.socket.send(job_msg)
                                            memory_info = f"(est. {estimated_memory:.1f}GB)"
                                            self.log_info(f"Job {job.job_id} assigned to {client_id} ({len(client_info.current_jobs)}/{client_info.thread_count} slots) {memory_info}")
                                            jobs_assigned += 1
                                        except Exception as e:
                                            self.log_error(f"Failed to send job to {client_id}: {e}")
                                            # Mark client as disconnected
                                            client_info.status = 'disconnected'
                                            # Clean up failed assignment
                                            client_info.remove_job(job.job_id)
                                            if job.job_id in self.active_jobs:
                                                del self.active_jobs[job.job_id]
                                            self.job_queue.put(job)  # Put job back in queue
                                            break  # Stop trying to assign to this client
                                    else:
                                        # No socket available, clean up and put job back
                                        client_info.remove_job(job.job_id)
                                        if job.job_id in self.active_jobs:
                                            del self.active_jobs[job.job_id]
                                        self.job_queue.put(job)
                                        break
                                else:
                                    # Failed to add job to client, put it back
                                    self.job_queue.put(job)
                                    break
                                    
                            except Empty:
                                break  # No more jobs in queue
                    
                    if jobs_assigned > 0:
                        self.log_debug(f"Assigned {jobs_assigned} jobs to {len(clients_with_capacity)} clients")
                elif queue_size > 0 and len(all_clients) > 0:
                    # Log when we have jobs but no available clients (every 10 seconds to avoid spam)
                    current_time = time.time()
                    if not hasattr(self, '_last_blocked_log') or (current_time - self._last_blocked_log) > 10:
                        self.log_debug(f"Job distribution blocked: {queue_size} jobs waiting, {len(all_clients)} total clients, 0 available")
                        self._last_blocked_log = current_time
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.log_error(f"Job monitor error: {e}")
                time.sleep(5)
    
    def _timeout_monitor(self):
        """Monitor job timeouts and handle hung jobs."""
        while self.running:
            try:
                current_time = time.time()
                timed_out_jobs = []
                
                # Check for jobs that have exceeded their timeout
                for job_id, job in self.active_jobs.items():
                    if job.start_time and (current_time - job.start_time) > job.timeout:
                        timed_out_jobs.append((job_id, job))
                
                # Handle timed out jobs
                for job_id, job in timed_out_jobs:
                    self.log_warning(f"Job {job_id} timed out after {job.timeout} seconds")
                    
                    # Find the client processing this job and remove it
                    client_to_update = None
                    for client_id, client_info in self.clients.items():
                        if job_id in client_info.current_jobs:
                            client_to_update = client_id
                            break
                    
                    if client_to_update:
                        client_info = self.clients[client_to_update]
                        client_info.remove_job(job_id)
                        self.log_info(f"Removed timed-out job {job_id} from client {client_to_update} ({len(client_info.current_jobs)}/{client_info.thread_count} slots)")
                    
                    # Move job to failed jobs
                    self.failed_jobs[job_id] = {
                        'job_id': job_id,
                        'error': f'Job timed out after {job.timeout} seconds',
                        'timeout': True,
                        'processing_time': current_time - job.start_time if job.start_time else 0
                    }
                    
                    # Remove from active jobs
                    del self.active_jobs[job_id]
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.log_error(f"Timeout monitor error: {e}")
                time.sleep(10)
    
    def _memory_monitor(self):
        """Monitor system memory and log memory status."""
        while self.running:
            try:
                # Log server memory status
                memory_info = self.memory_manager.get_system_memory_info()
                warning_threshold = self.memory_manager.config['memory_warning_threshold'] * 100
                critical_threshold = self.memory_manager.config['memory_critical_threshold'] * 100
                
                if memory_info['percent'] > critical_threshold:
                    self.log_error(f"Server memory critical: {memory_info['percent']:.1f}% used ({memory_info['available_gb']:.1f}GB available)")
                elif memory_info['percent'] > warning_threshold:
                    self.log_warning(f"Server memory high: {memory_info['percent']:.1f}% used ({memory_info['available_gb']:.1f}GB available)")
                else:
                    self.log_debug(f"Server memory: {memory_info['percent']:.1f}% used ({memory_info['available_gb']:.1f}GB available)")
                
                # Log client memory status
                current_time = time.time()
                for client_id, client_info in self.clients.items():
                    if client_info.status != 'disconnected' and (current_time - client_info.last_memory_update) < 120:
                        memory_status = client_info.memory_health_status()
                        if memory_status in ['critical', 'warning']:
                            self.log_info(f"Client {client_id} memory {memory_status}: {client_info.available_memory_gb:.1f}GB available")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.log_error(f"Memory monitor error: {e}")
                time.sleep(30)


class ProcessingClient:
    """Client that connects to server and processes image jobs."""
    
    def __init__(self, server_host: str, server_port: int = 8888, thread_count: int = 1, log_callback: Optional[Callable] = None, log_level: LogLevel = LogLevel.INFO, delayed_load = None):
        # Validate configuration
        is_valid, error_msg = validate_client_config(server_host, server_port, thread_count)
        if not is_valid:
            raise ValueError(f"Invalid client configuration: {error_msg}")
        
        self.server_host = server_host
        self.server_port = server_port
        self.thread_count = thread_count
        self._original_thread_count = thread_count  # Store original for revert functionality
        self.log_callback = log_callback or print
        self.log_level = log_level
        
        self.client_socket = None
        self.client_id = None
        self.running = False
        self.restart_requested = False  # Flag for server-requested restarts
        self.current_jobs: Dict[str, ProcessingJob] = {}
        
        # Console UI integration (will be set by headless client)
        self.console_ui = None
        
        # Connection retry settings
        self.max_retries = 5
        self.base_retry_delay = 1.0  # Base delay in seconds
        self.max_retry_delay = 60.0  # Maximum delay in seconds
        self.retry_count = 0
        self.should_reconnect = True
        
        # Threading
        if delayed_load:
            self.delayed_loading = True
            #Time in seconds to delay each file read to stagger processing for faster bulk thruput
            self.delayed_loading_time = delayed_load
        else:
            self.delayed_loading = False
            self.delayed_loading_time = 0
        self.heartbeat_thread = None
        self.receive_thread = None
        self.reconnect_thread = None
        self.worker_threads: List[threading.Thread] = []
        self.job_queue = Queue()
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Log message with timestamp and level."""
        if level.value >= self.log_level.value:
            timestamp = datetime.now().strftime("%H:%M:%S")
            level_name = level.name
            formatted_msg = f"[Client {timestamp}] [{level_name}] {message}"
            if self.log_callback:
                self.log_callback(formatted_msg)
            # else:
            #     print(formatted_msg)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def log_info(self, message: str):
        """Log info message."""
        self.log(message, LogLevel.INFO)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.log(message, LogLevel.WARNING)
    
    def log_error(self, message: str):
        """Log error message."""
        self.log(message, LogLevel.ERROR)
    
    def log_critical(self, message: str):
        """Log critical message."""
        self.log(message, LogLevel.CRITICAL)
    
    def connect(self) -> bool:
        """Connect to the server with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Enable TCP keepalive to prevent idle connection drops
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Configure keepalive parameters (Windows-specific)
                try:
                    # Send keepalive probes every 30 seconds
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
                    # Start keepalive probes after 60 seconds of inactivity
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                    # Send up to 3 keepalive probes before declaring connection dead
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                except (AttributeError, OSError):
                    # Fallback for systems that don't support these options
                    pass
                
                self.client_socket.settimeout(30.0)  # 30 second connection timeout for UI blocking tolerance
                self.client_socket.connect((self.server_host, self.server_port))
                
                # Register with server
                register_msg = NetworkProtocol.encode_message('client_register', {
                    'thread_count': self.thread_count,
                    'version': '1.0'
                })
                self.client_socket.send(register_msg)
                
                # Wait for registration acknowledgment
                msg_type, data = NetworkProtocol.decode_message(self.client_socket)
                if msg_type == 'register_ack':
                    self.client_id = data['client_id']
                    self.running = True
                    self.retry_count = 0  # Reset retry count on successful connection
                    
                    self.log(f"Connected to server {self.server_host}:{self.server_port} as {self.client_id}")
                    
                    # Start background threads
                    self.heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
                    self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
                    self.reconnect_thread = threading.Thread(target=self._monitor_connection, daemon=True)
                    
                    self.heartbeat_thread.start()
                    self.receive_thread.start()
                    self.reconnect_thread.start()
                    
                    # Start worker threads (daemon threads like old working code)
                    for i in range(self.thread_count):
                        worker_thread = threading.Thread(target=self._process_jobs, args=(f"worker-{i}",), daemon=True)
                        worker_thread.start()
                        self.worker_threads.append(worker_thread)

                    return True
                else:
                    self.log("Server registration failed")
                    self._close_socket()
                    
            except Exception as e:
                self.log(f"Connection attempt {attempt + 1}/{self.max_retries} failed: {e}")
                self._close_socket()
                
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt) + random.uniform(0, 1), self.max_retry_delay)
                    self.log(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
        
        self.log(f"Failed to connect after {self.max_retries} attempts")
        return False
        
    def _close_socket(self):
        """Safely close the client socket."""
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
    
    def disconnect(self):
        """Disconnect from server gracefully."""
        self.log_info("Initiating graceful disconnect...")
        self.running = False
        self.should_reconnect = False
        
        # Use graceful shutdown instead of just closing socket
        self._graceful_shutdown()
        
        # Daemon worker threads will exit automatically when main thread exits
        self.log_info("Daemon worker threads will exit automatically")
        
        self.log_info("Graceful disconnect completed")
    
    def _monitor_connection(self):
        """Monitor connection health and trigger reconnection if needed."""
        while self.running and self.should_reconnect:
            try:
                time.sleep(120)  # Check every 2 minutes (reduced frequency)
                
                # Only perform active health check if we haven't sent anything recently
                if self.client_socket and self.running:
                    # Let normal heartbeats handle connection monitoring
                    # Only do explicit test if we suspect issues
                    pass
                        
            except Exception as e:
                self.log_error(f"Connection monitor error: {e}")
                time.sleep(10)
    
    def _handle_disconnection(self):
        """Handle disconnection and attempt to reconnect."""
        if not self.should_reconnect or not self.running:
            return
            
        self.log("Connection lost, attempting to reconnect...")
        
        # Close current socket
        self._close_socket()
        
        # Reset state
        self.client_id = None
        
        # Add initial delay before reconnection attempts
        time.sleep(5)  # Give the server time to clean up the old connection
        
        # Attempt to reconnect with exponential backoff
        while self.running and self.should_reconnect:
            delay = min(self.base_retry_delay * (2 ** self.retry_count) + random.uniform(0, 1), self.max_retry_delay)
            self.log_info(f"Reconnecting in {delay:.1f} seconds... (attempt {self.retry_count + 1})")
            time.sleep(delay)
            
            if self._attempt_reconnect():
                self.log("Reconnection successful")
                return
                
            self.retry_count += 1
            # if self.retry_count >= self.max_retries:
            #     self.log("Max reconnection attempts reached. Stopping client.")
            #     self.running = False
            #     return
    
    def _attempt_reconnect(self) -> bool:
        """Attempt a single reconnection."""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Apply same keepalive settings as initial connection
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            except (AttributeError, OSError):
                pass
            
            self.client_socket.settimeout(30.0)  # Increased timeout for server responsiveness during UI operations
            self.client_socket.connect((self.server_host, self.server_port))
            
            # Register with server
            register_msg = NetworkProtocol.encode_message('client_register', {
                'thread_count': self.thread_count,
                'version': '1.0',
                'reconnect': True
            })
            self.client_socket.send(register_msg)
            
            # Wait for registration acknowledgment
            msg_type, data = NetworkProtocol.decode_message(self.client_socket)
            if msg_type == 'register_ack':
                self.client_id = data['client_id']
                self.retry_count = 0  # Reset retry count on successful reconnection
                
                # Clear any stale job state from before reconnection
                stale_job_count = len(self.current_jobs)
                self.current_jobs.clear()
                if stale_job_count > 0:
                    self.log_info(f"Cleared {stale_job_count} stale jobs after reconnection")
                
                # Also clear the job queue to ensure no phantom jobs remain
                while not self.job_queue.empty():
                    try:
                        self.job_queue.get_nowait()
                        self.job_queue.task_done()
                    except:
                        break
                
                # Restart threads if they died or don't exist
                if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
                    self.log_info("Restarting heartbeat thread after reconnection")
                    self.heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
                    self.heartbeat_thread.start()
                    
                if not self.receive_thread or not self.receive_thread.is_alive():
                    self.log_info("Restarting receive thread after reconnection")
                    self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
                    self.receive_thread.start()
                
                # Restart worker threads if they died or recreate them if the list is incomplete
                current_alive_workers = [w for w in self.worker_threads if w.is_alive()]
                expected_worker_count = self.thread_count
                
                if len(current_alive_workers) < expected_worker_count:
                    missing_workers = expected_worker_count - len(current_alive_workers)
                    self.log_info(f"Restarting {missing_workers} worker threads after reconnection")
                    
                    # Clear the worker thread list and recreate all workers
                    self.worker_threads.clear()
                    for i in range(expected_worker_count):
                        worker_thread = threading.Thread(target=self._process_jobs, args=(f"worker-{i}",), daemon=True)
                        worker_thread.start()
                        self.worker_threads.append(worker_thread)
                        
                    self.log_info(f"Recreated {expected_worker_count} worker threads after reconnection")
                else:
                    self.log_info(f"All {len(current_alive_workers)}/{expected_worker_count} worker threads are alive after reconnection")
                
                # Send an immediate heartbeat to ensure server recognizes this client as available
                # Add a small delay to ensure all threads are fully started
                time.sleep(0.1)  
                try:
                    self._send_immediate_heartbeat()
                    self.log_debug("Sent immediate heartbeat after successful reconnection")
                except Exception as e:
                    self.log_warning(f"Failed to send immediate heartbeat after reconnection: {e}")
                
                return True
            else:
                self._close_socket()
                return False
                
        except Exception as e:
            self.log(f"Reconnection failed: {e}")
            self._close_socket()
            return False
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current client memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except Exception:
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0}
    
    def _send_immediate_heartbeat(self):
        """Send an immediate heartbeat to the server."""
        if self.client_socket and self.running:
            active_job_count = len(self.current_jobs)
            status = 'processing' if active_job_count > 0 else 'idle'
            
            # Get memory information
            memory_info = self._get_memory_info()
            
            heartbeat_msg = NetworkProtocol.encode_message('heartbeat', {
                'status': status,
                'active_jobs': active_job_count,
                'max_jobs': self.thread_count,
                'available_capacity': self.thread_count - active_job_count,
                'memory_info': memory_info,
                'reconnected': True  # Flag to indicate this is a post-reconnection heartbeat
            })
            self.client_socket.send(heartbeat_msg)
    
    def _send_heartbeats(self):
        """Send periodic heartbeats to server."""
        while self.running:
            try:
                if self.client_socket and self.running:
                    active_job_count = len(self.current_jobs)
                    status = 'processing' if active_job_count > 0 else 'idle'
                    
                    # Get memory information
                    memory_info = self._get_memory_info()
                    
                    heartbeat_msg = NetworkProtocol.encode_message('heartbeat', {
                        'status': status,
                        'active_jobs': active_job_count,
                        'max_jobs': self.thread_count,
                        'available_capacity': self.thread_count - active_job_count,
                        'memory_info': memory_info
                    })
                    self.client_socket.send(heartbeat_msg)
                
                time.sleep(30)  # Send heartbeat every 30 seconds
                
            except (socket.error, ConnectionError, BrokenPipeError) as e:
                if self.running and self.should_reconnect:
                    self.log_warning(f"Heartbeat failed, triggering reconnection: {e}")
                    self._handle_disconnection()
                break
            except Exception as e:
                self.log(f"Unexpected heartbeat error: {e}")
                time.sleep(5)  # Short delay before retrying
    
    def _receive_messages(self):
        """Receive messages from server."""
        while self.running:
            try:
                if not self.client_socket:
                    time.sleep(1)
                    continue
                    
                msg_type, data = NetworkProtocol.decode_message(self.client_socket, timeout=35.0)  # Longer timeout to match heartbeat interval
                
                if msg_type == 'process_job':
                    job = ProcessingJob.from_dict(data)
                    self.job_queue.put(job)
                    self.log_info(f"Received job {job.job_id}")
                elif msg_type == 'connection_test':
                    # Respond to server connection test
                    response = NetworkProtocol.encode_message('connection_test_ack', {})
                    if self.client_socket:
                        self.client_socket.send(response)
                elif msg_type == 'server_restart':
                    # Server is notifying us it restarted, we need to re-register
                    self.log_info("Server restart detected, triggering re-registration")
                    self._handle_disconnection()
                elif msg_type == 'client_unknown':
                    # Server doesn't recognize our client_id, need to re-register
                    self.log_warning("Server doesn't recognize this client, triggering re-registration")
                    self._handle_disconnection()
                elif msg_type == 'pause_processing':
                    # Server requesting us to pause processing
                    self.log_info("Received pause processing command from server")
                    self._handle_pause_command()
                elif msg_type == 'restart_client':
                    # Server requesting client restart
                    self.log_info("Received restart command from server")
                    self._handle_restart_command()
                elif msg_type == 'set_thread_count':
                    # Server requesting thread count change
                    new_count = data.get('new_thread_count', self.thread_count)
                    self.log_info(f"Received thread count change command: {new_count}")
                    self._handle_thread_count_change(new_count)
                
            except socket.timeout:
                continue
            except (socket.error, ConnectionError, BrokenPipeError) as e:
                if self.running and self.should_reconnect:
                    # Categorize error types for better handling
                    error_str = str(e).lower()
                    if "connection closed" in error_str or "broken pipe" in error_str:
                        self.log_warning(f"Server closed connection: {e}")
                    elif "timed out" in error_str:
                        self.log_debug(f"Socket timeout during idle period: {e}")
                        # Don't immediately reconnect on timeout - let heartbeat handle it
                        continue
                    else:
                        self.log_error(f"Network error: {e}")
                    
                    self._handle_disconnection()
                break
            except Exception as e:
                if self.running:
                    self.log(f"Unexpected receive error: {e}")
                break
    
    def _process_jobs(self, worker_id: str = "worker"):
        """Worker thread that processes jobs from the queue."""
        worker_index = 0  # default
        started = False
        if '-' in worker_id:
            try:
                worker_index = int(worker_id.split('-')[1])
            except (IndexError, ValueError):
                worker_index = 0

        self.log_debug(f"[{worker_id}] Worker thread started")
        
        while self.running:
            try:
                job = self.job_queue.get(timeout=1.0)
                self.log_debug(f"[{worker_id}] Got job from queue: {job.job_id}")
                self.current_jobs[job.job_id] = job
                self.log_debug(f"[{worker_id}] Added job to current_jobs: {job.job_id}")

                # Update console UI with new job
                if self.console_ui:
                    self.console_ui.add_job(job.job_id, os.path.basename(job.image_path))

                self.log_info(f"[{worker_id}] Processing job {job.job_id}: {os.path.basename(job.image_path)}")

                if self.delayed_loading:
                    if not started:
                        delay_seconds = float(self.delayed_loading_time)
                        # Optionally stagger the delays
                        staggered_delay = delay_seconds * worker_index
                        self.log_debug(f"{worker_id}: Delaying start for {staggered_delay:.1f} seconds...")
                        time.sleep(staggered_delay)
                        started = True

                    self.log_debug(f"[{worker_id}] Calling _execute_job for {job.job_id}")
                    job_complete = self._execute_job(job)
                    self.log_debug(f"[{worker_id}] _execute_job returned: {job_complete} for {job.job_id}")
                else:
                    self.log_debug(f"[{worker_id}] Calling _execute_job for {job.job_id}")
                    job_complete = self._execute_job(job)
                    self.log_debug(f"[{worker_id}] _execute_job returned: {job_complete} for {job.job_id}")

                # Update console UI with job completion
                processing_time = time.time() - job.created_time
                if self.console_ui:
                    self.console_ui.complete_job(job.job_id, success=job_complete, processing_time=processing_time)

                # Send result back to server
                if job_complete:
                    result_msg = NetworkProtocol.encode_message('job_complete', {
                        'job_id': job.job_id,
                        'output_path': job.output_path,
                        'processing_time': processing_time
                    })
                else:
                    result_msg = NetworkProtocol.encode_message('job_failed', {
                        'job_id': job.job_id,
                        'error': 'Processing failed'
                    })

                # Send result with error handling
                self.log_debug(f"[{worker_id}] About to send result for {job.job_id}")
                try:
                    if self.client_socket and self.running:
                        self.log_debug(f"[{worker_id}] Sending result message for {job.job_id}")
                        self.client_socket.send(result_msg)
                        self.log_debug(f"[{worker_id}] Result sent successfully for {job.job_id}")
                    else:
                        self.log_warning(f"[{worker_id}] Cannot send result - socket/running issue for {job.job_id}")
                except (socket.error, ConnectionError, BrokenPipeError) as e:
                    self.log_error(f"[{worker_id}] Network error sending job result for {job.job_id}: {e}")
                    if self.should_reconnect:
                        self._handle_disconnection()
                except Exception as e:
                    self.log_error(f"[{worker_id}] Unexpected error sending job result for {job.job_id}: {e}")
                    import traceback
                    self.log_debug(f"[{worker_id}] Send result traceback: {traceback.format_exc()}")

                # Cleanup
                self.log_debug(f"[{worker_id}] Starting cleanup for {job.job_id}")
                del self.current_jobs[job.job_id]
                self.job_queue.task_done()
                self.log_debug(f"[{worker_id}] Completed processing of job {job.job_id}")
                
            except Empty:
                # Check if we should continue running
                if not self.running:
                    self.log_debug(f"[{worker_id}] Stopping worker due to shutdown signal")
                    break
                continue
            except Exception as e:
                self.log_error(f"[{worker_id}] Job processing error: {e}")
                # Let critical errors crash the worker thread as they should
                import traceback
                self.log_debug(f"[{worker_id}] Worker thread traceback: {traceback.format_exc()}")
                break  # Exit the worker thread on serious errors
        
        self.log_debug(f"[{worker_id}] Worker thread exiting")

    def _execute_job(self, job: ProcessingJob) -> bool:
        """Execute a single processing job (image processing or other job types)."""
        try:
            self.log_debug(f"Starting job execution for {job.job_id}")
            
            # Check for special job types
            job_type = job.settings.get("job_type", "image_processing")
            
            if job_type == "project_data":
                return self._execute_project_data_job(job)
            
            # Default: image processing job
            if not os.path.exists(job.image_path):
                self.log_error(f"Input file not found: {job.image_path}")
                return False

            # Ensure output directory exists
            output_dir = os.path.dirname(job.output_path)
            output_path = str(Path(job.output_path))
            self.log_debug(f"Job output directory: {output_dir}")
            self.log_debug(f"Job output path: {output_path}")
            
            if output_dir:
                self.log_debug(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                self.log_debug(f"Output directory created successfully")

            # Setup signals with progress reporting
            self.log_debug(f"Setting up WorkerSignals for {job.job_id}")
            sig = WorkerSignals()
            sig.log.connect(self.log)
            sig.preview.connect(lambda *args: None)
            sig.status.connect(lambda status: self._send_progress_update(job.job_id, status))
            self.log_debug(f"WorkerSignals setup complete for {job.job_id}")

            # Prepare args for worker
            image_list = [job.image_path]
            self.log_debug(f"Image list prepared: {image_list}")

            # Check if ImageCorrectionWorker is available
            if ImageCorrectionWorker is None:
                self.log_error(f"ImageCorrectionWorker is not available - check dependencies")

                return False
            self.log_debug(f"ImageCorrectionWorker is available")

            self.log_debug(f"Creating ImageCorrectionWorker for {job.job_id}")

            
            # Extract and print each parameter individually with proper defaults
            jpeg_quality = job.settings.get("jpeg_quality", 100)
            output_format = job.settings.get("output_format", ".jpg") 
            tiff_bitdepth = job.settings.get("tiff_bitdepth", 8)
            exr_colorspace = job.settings.get("exr_colorspace", "sRGB")
            group_name = job.settings.get("group_name", "")
            exposure_adj = job.settings.get("exposure_adj", 0.0)
            shadow_adj = job.settings.get("shadow_adj", 0.0)
            highlight_adj = job.settings.get("highlight_adj", 0.0)
            white_balance_adj = job.settings.get("white_balance_adj", 5500)
            enable_white_balance = job.settings.get("enable_white_balance", False)
            # If white balance is disabled, use default value instead
            if not enable_white_balance:
                white_balance_adj = 5500
            denoise_strength = job.settings.get("denoise_strength", 0.0)
            sharpen_amount = job.settings.get("sharpen_amount", 0.0)
            export_schema = job.settings.get("export_schema", "")
            custom_name = job.settings.get("custom_name", "")
            root_folder = job.settings.get("root_folder", "")
            
            # Determine if we're using chart-based correction or manual adjustments
            use_chart = job.swatches is not None
            
            print(f"[DEBUG] Server passing network_output_path: '{output_path}' to worker")
            
            worker = ImageCorrectionWorker(
                images=image_list,
                swatches=job.swatches,
                output_folder=output_dir,
                signals=sig,
                jpeg_quality=jpeg_quality,
                output_format=output_format,
                tiff_bitdepth=tiff_bitdepth,
                exr_colorspace=exr_colorspace,
                from_network=True,
                network_output_path=output_path,
                group_name=group_name,
                use_chart=use_chart,
                exposure_adj=exposure_adj,
                shadow_adj=shadow_adj,
                highlight_adj=highlight_adj,
                white_balance_adj=white_balance_adj,
                denoise_strength=denoise_strength,
                sharpen_amount=sharpen_amount,
                export_schema=export_schema,
                use_export_schema=job.settings.get("use_export_schema", False),
                custom_name=custom_name,
                root_folder=root_folder
            )
            self.log_debug(f"ImageCorrectionWorker created successfully for {job.job_id}")

            # Optional attributes
            self.log_debug(f"Setting optional attributes for {job.job_id}")
            worker.use_original_filenames = getattr(job, "use_original_filenames", False)
            worker.image_metadata_map = getattr(job, "image_metadata_map", {})
            self.log_debug(f"Optional attributes set for {job.job_id}")

            # Run synchronously
            self.log_debug(f"About to call worker.run() for {job.job_id}")
            self.log_debug(f"Worker state - images: {len(worker.images)}, swatches: {len(job.swatches) if job.swatches else 'None'}")
            
            try:
                worker.run()
                self.log_debug(f"worker.run() completed successfully for {job.job_id}")
            except Exception as worker_e:
                self.log_error(f"worker.run() failed for {job.job_id}: {worker_e}")
                import traceback
                self.log_debug(f"Worker exception traceback: {traceback.format_exc()}")
                raise  # Re-raise to be caught by outer try-catch

            # Verify output file was created
            if os.path.exists(output_path):
                self.log_info(f"Job {job.job_id} completed successfully - output created: {output_path}")
                return True
            else:
                self.log_error(f"Job {job.job_id} completed but no output file found at: {output_path}")
                return False

        except Exception as e:
            self.log_error(f"Job {job.job_id} failed: {e}")
            import traceback
            self.log_debug(f"Full traceback: {traceback.format_exc()}")
            return False

    def _execute_project_data_job(self, job: ProcessingJob) -> bool:
        """Execute a project data job - handle project configuration data."""
        try:
            self.log_info(f"Processing project data job {job.job_id}")
            
            # Extract project data from job settings
            project_json = job.settings.get("project_json", "")
            total_images = job.settings.get("total_images", 0)
            total_groups = job.settings.get("total_groups", 0)
            
            if not project_json:
                self.log_error(f"Project data job {job.job_id} has no project JSON data")
                return False
            
            # Log project information
            self.log_info(f"Received project data with {total_images} images across {total_groups} groups")
            
            # Here you can add custom logic for handling project data:
            # - Save to a file for later use
            # - Parse and extract specific information
            # - Distribute sub-jobs based on the project data
            # - Update client configurations
            
            # Example: Save project data to a file for reference
            import tempfile
            import json
            import os
            
            # Create a temporary file to store the project data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix=f'project_{job.job_id}_') as f:
                f.write(project_json)
                temp_file_path = f.name
            
            self.log_info(f"Project data saved to: {temp_file_path}")
            
            # You could also parse the JSON and create individual image processing jobs
            try:
                project_data = json.loads(project_json)
                
                # Example: Log some project information
                if 'metadata' in project_data:
                    software = project_data['metadata'].get('software', 'Unknown')
                    export_date = project_data['metadata'].get('export_date', 'Unknown')
                    self.log_info(f"Project from {software}, exported on {export_date}")
                
                if 'images' in project_data:
                    self.log_info(f"Project contains {len(project_data['images'])} images")
                    
                    # Example: Could create individual processing jobs for each image
                    # for img_data in project_data['images']:
                    #     self._create_job_from_image_data(img_data, project_data)
                
            except json.JSONDecodeError as e:
                self.log_error(f"Failed to parse project JSON: {e}")
                return False
            
            # Mark as successful
            self.log_info(f"Project data job {job.job_id} completed successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Project data job {job.job_id} failed: {e}")
            import traceback
            self.log_debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _send_progress_update(self, job_id: str, status: str):
        """Send progress update to server."""
        try:
            if self.client_socket and self.running:
                progress_msg = NetworkProtocol.encode_message('job_progress', {
                    'job_id': job_id,
                    'status': status,
                    'timestamp': time.time()
                })
                self.client_socket.send(progress_msg)
                self.log_debug(f"Progress update for job {job_id}: {status}")
        except (socket.error, ConnectionError, BrokenPipeError):
            self.log_error(f"Failed to send progress update for job {job_id}")
        except Exception as e:
            self.log_error(f"Unexpected error sending progress update: {e}")
    
    def _handle_pause_command(self):
        """Handle pause processing command from server."""
        # For now, we'll just log this. In a full implementation, you might:
        # - Stop accepting new jobs temporarily
        # - Pause current jobs if possible
        # - Send acknowledgment back to server
        self.log_info("Processing pause requested by server")
        
        try:
            if self.client_socket and self.running:
                ack_msg = NetworkProtocol.encode_message('pause_ack', {
                    'client_id': self.client_id,
                    'status': 'paused'
                })
                self.client_socket.send(ack_msg)
        except Exception as e:
            self.log_error(f"Failed to send pause acknowledgment: {e}")
    
    def _handle_restart_command(self):
        """Handle restart command from server with graceful shutdown."""
        self.log_info("Client restart requested by server - initiating graceful shutdown")
        
        try:
            # Send acknowledgment before shutting down
            if self.client_socket and self.running:
                ack_msg = NetworkProtocol.encode_message('restart_ack', {
                    'client_id': self.client_id,
                    'status': 'restarting'
                })
                self.client_socket.send(ack_msg)
        except Exception as e:
            self.log_error(f"Failed to send restart acknowledgment: {e}")
        
        # Stop accepting new jobs immediately and mark for restart
        self.should_reconnect = False
        self.restart_requested = True  # Flag for HeadlessClient to detect
        
        # Wait for current jobs to complete or timeout after 30 seconds
        self.log_info("Waiting for current jobs to complete...")
        wait_start = time.time()
        max_wait_time = 30  # 30 seconds maximum wait
        
        while self.current_jobs and (time.time() - wait_start) < max_wait_time:
            remaining_jobs = len(self.current_jobs)
            elapsed = time.time() - wait_start
            self.log_info(f"Waiting for {remaining_jobs} jobs to complete... ({elapsed:.1f}s elapsed)")
            time.sleep(2)  # Check every 2 seconds
        
        # Handle any remaining jobs
        if self.current_jobs:
            self.log_warning(f"Timeout reached - forcing shutdown with {len(self.current_jobs)} jobs remaining")
            # Mark remaining jobs as failed
            for job_id in list(self.current_jobs.keys()):
                try:
                    if self.client_socket:
                        fail_msg = NetworkProtocol.encode_message('job_failed', {
                            'job_id': job_id,
                            'error': 'Client restart - job interrupted'
                        })
                        self.client_socket.send(fail_msg)
                except Exception:
                    pass
        else:
            self.log_info("All jobs completed successfully before restart")
        
        # Clear job state
        self.current_jobs.clear()
        
        # Graceful shutdown of threads
        self._graceful_shutdown()
        
        # Final log message
        self.log_info("Graceful shutdown complete - ready for restart")
        
        # Small delay to allow final log messages to be written
        time.sleep(0.5)
        self.running = False

    
    def _handle_thread_count_change(self, new_thread_count: int):
        """Handle thread count change command from server."""
        if new_thread_count <= 0 or new_thread_count > 32:
            self.log_error(f"Invalid thread count requested: {new_thread_count}")
            return
        
        old_count = self.thread_count
        self.thread_count = new_thread_count
        
        # Update console UI with new thread count
        if self.console_ui:
            self.console_ui.update_stats(thread_count=new_thread_count)
        
        self.log_info(f"Thread count changed from {old_count} to {new_thread_count}")
        
        try:
            # Send acknowledgment
            if self.client_socket and self.running:
                ack_msg = NetworkProtocol.encode_message('thread_count_ack', {
                    'client_id': self.client_id,
                    'old_count': old_count,
                    'new_count': new_thread_count
                })
                self.client_socket.send(ack_msg)
        except Exception as e:
            self.log_error(f"Failed to send thread count acknowledgment: {e}")
        
        # Adjust worker threads if needed
        self._adjust_worker_threads()
    
    def _adjust_worker_threads(self):
        """Adjust the number of worker threads to match thread_count."""
        current_alive_workers = [w for w in self.worker_threads if w.is_alive()]
        current_count = len(current_alive_workers)
        target_count = self.thread_count
        
        if current_count < target_count:
            # Need to add more worker threads
            for i in range(current_count, target_count):
                worker_thread = threading.Thread(
                    target=self._process_jobs, 
                    args=(f"worker-{i}",), 
                    daemon=True
                )
                worker_thread.start()
                self.worker_threads.append(worker_thread)
                self.log_info(f"Started new worker thread: worker-{i}")
        
        elif current_count > target_count:
            # Too many threads - they'll naturally finish and not be replaced
            # We don't force-kill them as that could corrupt ongoing jobs
            self.log_info(f"Excess worker threads will finish naturally ({current_count} -> {target_count})")
        
        self.log_info(f"Worker thread adjustment: {current_count} -> {target_count} (target)")
    
    def get_original_thread_count(self) -> int:
        """Get the original thread count this client was started with."""
        # This would typically be stored when the client starts
        # For now, return the current thread count
        return getattr(self, '_original_thread_count', self.thread_count)
    
    def _graceful_shutdown(self):
        """Gracefully shut down all client threads and connections."""
        self.log_info("Starting graceful shutdown of client threads...")
        
        # Close socket connection first to stop receiving new messages
        try:
            if self.client_socket:
                self.log_debug("Closing client socket...")
                self.client_socket.close()
                self.client_socket = None
        except Exception as e:
            self.log_debug(f"Error closing socket: {e}")
        
        # Daemon worker threads will exit automatically - no need to wait
        if self.worker_threads:
            self.log_debug(f"Daemon worker threads will exit automatically: {len(self.worker_threads)} threads")
            # Clear the worker threads list 
            self.worker_threads.clear()
        
        # Wait for communication threads to finish
        communication_threads = [
            ('heartbeat', self.heartbeat_thread),
            ('receive', self.receive_thread),
            ('reconnect', self.reconnect_thread)
        ]
        
        for thread_name, thread in communication_threads:
            if thread and thread.is_alive():
                try:
                    self.log_debug(f"Waiting for {thread_name} thread to finish...")
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        self.log_warning(f"{thread_name} thread did not finish within timeout")
                    else:
                        self.log_debug(f"{thread_name} thread finished gracefully")
                except Exception as e:
                    self.log_debug(f"Error joining {thread_name} thread: {e}")
        
        # Clear job queue
        if hasattr(self, 'job_queue'):
            try:
                while not self.job_queue.empty():
                    try:
                        self.job_queue.get_nowait()
                        self.job_queue.task_done()
                    except:
                        break
                self.log_debug("Job queue cleared")
            except Exception as e:
                self.log_debug(f"Error clearing job queue: {e}")
        
        self.log_info("Graceful shutdown completed")


def create_headless_client():
    """Create a headless client that can run without GUI with enhanced error handling."""
    import argparse
    import socket
    import sys
    from PySide6.QtCore import QSettings
    
    parser = argparse.ArgumentParser(
        description='Headless Image Processing Client with Enhanced Error Handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python networkProcessor.py --host 192.168.1.100 --port 8888 --threads 4
  python networkProcessor.py --host 192.168.1.100 --threads 2 --delayLoad 0.5
  python networkProcessor.py  # Interactive setup

Features:
  - Enhanced error handling and recovery
  - Automatic connection validation
  - Graceful handling of network issues
  - Better logging and status reporting
        """
    )
    parser.add_argument('--host', help='Server host address')
    parser.add_argument('--port', type=int, default=8888, help='Server port (default: 8888)')
    parser.add_argument('--threads', type=int, help='Number of processing threads')
    parser.add_argument('--delayLoad', type=float, default=0, help='Delay loading of images by X seconds')
    parser.add_argument('--maxRetries', type=int, default=5, help='Maximum connection retry attempts (default: 5)')
    parser.add_argument('--retryDelay', type=float, default=5.0, help='Delay between retry attempts in seconds (default: 5.0)')
    parser.add_argument('--validateConnection', action='store_true', help='Validate server connection before starting')
    parser.add_argument('--no-colors', action='store_true', help='Disable colored output and status bar')
    parser.add_argument('--no-status-bar', action='store_true', help='Disable status bar (keep colors)')
    
    args = parser.parse_args()
    
    # Create beautiful console UI based on arguments
    console = ConsoleUI(enable_colors=not args.no_colors)
    
    # Check if ImageCorrectionWorker is available
    if ImageCorrectionWorker is None:
        console.log("ImageCorrectionWorker is not available!", "ERROR")
        console.log("This usually means:", "ERROR")
        console.log("  - You're not running from the correct directory", "ERROR")
        console.log("  - Missing dependencies (OpenImageIO, colour-science, etc.)", "ERROR")
        console.log("  - Import path issues", "ERROR")
        console.log("Please ensure you're running from the ImageProcessor directory with all dependencies installed.", "ERROR")
        return 1
    
    console.log("ImageCorrectionWorker loaded successfully", "SUCCESS")
    
    # Load settings or prompt for them
    settings = QSettings('ScanSpace', 'ImageProcessor')
    
    host = args.host or settings.value('server_host', '')
    port = args.port
    threads = args.threads or settings.value('client_threads', 1, type=int)
    delay_load = args.delayLoad
    max_retries = args.maxRetries
    retry_delay = args.retryDelay
    
    # Enhanced input validation and prompting
    if not host:
        try:
            host = input("Enter server host address: ").strip()
            if not host:
                console.log("Server host is required", "ERROR")
                return 1
            settings.setValue('server_host', host)
        except (EOFError, KeyboardInterrupt):
            console.log("Operation cancelled by user", "WARNING")
            return 1

    if not port:
        port = settings.value('client_port', 8888, type=int)
        if not port:
            console.log("Server port is required", "ERROR")
            return 1
        settings.setValue('client_port', port)

    if not threads:
        try:
            threads_input = input(f"Enter thread count (default 1): ").strip()
            threads = int(threads_input) if threads_input else 1
            if threads < 1 or threads > 32:
                console.log("Thread count should be between 1 and 32, using default: 1", "WARNING")
                threads = 1
        except (ValueError, EOFError, KeyboardInterrupt):
            console.log("Using default thread count: 1", "INFO")
            threads = 1
        settings.setValue('client_threads', threads)
    
    settings.sync()
    
    # Update console stats
    console.update_stats(
        server_host=f"{host}:{port}",
        thread_count=threads,
        connection_status='Disconnected'
    )
    
    # Validate connection if requested
    if args.validateConnection:
        console.log(f"Validating connection to {host}:{port}...", "INFO")
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(5.0)
            result = test_socket.connect_ex((host, port))
            test_socket.close()
            
            if result != 0:
                console.log(f"Cannot connect to {host}:{port}", "ERROR")
                console.log("Please check that the server is running and accessible", "ERROR")
                return 1
            else:
                console.log("Server connection validated", "SUCCESS")
        except Exception as e:
            console.log(f"Connection validation failed: {e}", "ERROR")
            return 1
    
    # Enable status bar unless disabled
    if not args.no_status_bar and not args.no_colors:
        console.enable_status_bar()
    
    console.log("Starting Headless Image Processing Client", "INFO")
    console.log(f"Server: {host}:{port}", "INFO")
    console.log(f"Threads: {threads}", "INFO")
    console.log(f"Delayed Loading: {delay_load}s", "INFO")
    console.log(f"Max Retries: {max_retries}", "INFO")
    console.log(f"Retry Delay: {retry_delay}s", "INFO")
    console.log(f"Hostname: {socket.gethostname()}", "INFO")
    
    # Enhanced client creation with retry logic
    attempt = 0
    client = None
    
    def console_log_callback(message: str):
        """Console log callback that integrates with our beautiful UI."""
        # Parse log level from message if it contains level markers
        if "] [" in message and message.count("]") >= 2:
            try:
                level_part = message.split("] [")[1].split("]")[0]
                actual_message = message.split("] ", 2)[2]
                console.log(actual_message, level_part)
            except:
                console.log(message, "INFO")
        else:
            console.log(message, "INFO")
    
    while attempt < max_retries:
        try:
            attempt += 1
            console.log(f"Connection attempt {attempt}/{max_retries}...", "INFO")
            
            # Create client with enhanced settings
            client = ProcessingClient(
                server_host=host, 
                server_port=port, 
                thread_count=threads,
                delayed_load=delay_load,
                log_callback=console_log_callback,
                log_level=LogLevel.INFO
            )
            
            if client.connect():
                console.log("Successfully connected to server", "SUCCESS")
                console.log("Client is now ready to process images", "SUCCESS")
                console.log("Press Ctrl+C to shutdown gracefully", "INFO")
                
                # Update connection status
                console.update_stats(
                    connection_status='Connected',
                    client_id=getattr(client, 'client_id', '')
                )
                
                # Start interactive command interface
                console.start_command_interface(client)
                break
            else:
                console.log(f"Connection attempt {attempt} failed", "ERROR")
                client = None
                
                if attempt < max_retries:
                    console.log(f"Retrying in {retry_delay} seconds...", "WARNING")
                    time.sleep(retry_delay)
                
        except Exception as e:
            console.log(f"Connection attempt {attempt} failed with error: {e}", "ERROR")
            client = None
            
            if attempt < max_retries:
                console.log(f"Retrying in {retry_delay} seconds...", "WARNING")
                time.sleep(retry_delay)
    
    if not client:
        console.log(f"Failed to connect after {max_retries} attempts", "ERROR")
        return 1

    # Main client loop with restart support
    restart_requested = False
    
    try:
        while True:  # Outer loop for restart handling
            try:
                # Keep client running with enhanced monitoring
                last_status_time = time.time()
                status_update_interval = 2  # Update status bar every 2 seconds
                
                while client.running:
                    time.sleep(1)
                    
                    # Check for server-requested restart
                    if getattr(client, 'restart_requested', False):
                        console.log("Server requested restart - waiting for jobs to complete...", "WARNING")
                        restart_requested = True
                        break
                    
                    # Process any pending commands
                    try:
                        while not console.command_queue.empty():
                            command = console.command_queue.get_nowait()
                            
                            # Handle special system commands
                            if command == '__quit__':
                                console.log("Quit command received, shutting down...", "WARNING")
                                client.running = False
                                restart_requested = False
                                break
                            elif command == '__reconnect__':
                                console.log("Reconnect command received, triggering restart...", "INFO")
                                restart_requested = True
                                break
                            else:
                                # Process regular user commands
                                console.process_command(command)
                                
                    except Empty:
                        pass
                    except Exception as e:
                        console.log(f"Command processing error: {e}", "ERROR")
                    
                    # Update status bar with current job information
                    current_time = time.time()
                    if (current_time - last_status_time) >= status_update_interval:
                        # Get current jobs from client
                        current_jobs = getattr(client, 'current_jobs', {})
                        
                        # Update console with current job info (use snapshot to avoid iteration errors)
                        for job_id, job in list(current_jobs.items()):
                            if hasattr(job, 'image_path'):
                                filename = os.path.basename(job.image_path)
                                if job_id not in console.current_jobs:
                                    console.add_job(job_id, filename)
                        
                        # Remove completed jobs from console tracking
                        for tracked_job_id in list(console.current_jobs.keys()):
                            if tracked_job_id not in current_jobs:
                                console.complete_job(tracked_job_id, success=True)
                        
                        last_status_time = current_time
                
                # Check if we should restart
                if restart_requested and client.running:
                    console.log("Waiting for current jobs to complete before restart...", "INFO")
                    
                    # Wait for jobs to complete with timeout
                    wait_timeout = 30  # 30 seconds max wait
                    wait_start = time.time()
                    
                    while (time.time() - wait_start) < wait_timeout:
                        current_jobs = getattr(client, 'current_jobs', {})
                        if not current_jobs:
                            break
                        console.log(f"Waiting for {len(current_jobs)} jobs to complete...", "INFO")
                        time.sleep(2)
                    
                    # Graceful disconnect
                    console.log("Disconnecting for restart...", "INFO")
                    console.update_stats(connection_status='Restarting...')
                    client.disconnect()
                    client = None
                    
                    # Reset restart flag
                    restart_requested = False
                    
                    # Wait a moment before restart
                    console.log("Restarting client in 3 seconds...", "INFO")
                    time.sleep(3)
                    
                    # Create new client with same configuration
                    console.log("Creating new client connection...", "INFO")
                    client = ProcessingClient(
                        server_host=host, 
                        server_port=port, 
                        thread_count=threads,
                        delayed_load=delay_load,
                        log_callback=console_log_callback,
                        log_level=LogLevel.INFO
                    )
                    
                    if client.connect():
                        console.log("Restart successful - reconnected to server", "SUCCESS")
                        console.update_stats(
                            connection_status='Connected',
                            client_id=getattr(client, 'client_id', '')
                        )
                        console.start_command_interface(client)
                        # Continue the outer loop for monitoring
                        continue
                    else:
                        console.log("Restart failed - could not reconnect", "ERROR")
                        break
                
                # If we reach here without restart, break the outer loop
                break
                
            except Exception as e:
                console.log(f"Error in client monitoring loop: {e}", "ERROR")
                import traceback
                for line in traceback.format_exc().splitlines():
                    console.log(line, "ERROR")
                
                if restart_requested:
                    console.log("Attempting restart due to error...", "WARNING")
                    restart_requested = False
                    continue
                else:
                    break
                
    except KeyboardInterrupt:
        console.log("Graceful shutdown requested...", "WARNING")
        console.log("Finishing current jobs and disconnecting...", "INFO")
        
    except Exception as e:
        console.log(f"Unexpected error in main loop: {e}", "ERROR")
        import traceback
        console.log("Traceback:", "ERROR")
        for line in traceback.format_exc().splitlines():
            console.log(line, "ERROR")
        
    finally:
        console.update_stats(connection_status='Disconnected')
        
        # Stop command interface
        console.stop_command_interface()
        
        if client:
            console.log("Disconnecting from server...", "INFO")
            try:
                client.disconnect()
                console.log("Disconnected successfully", "SUCCESS")
            except Exception as e:
                console.log(f"Warning: Error during disconnect: {e}", "WARNING")
        
        console.disable_status_bar()
    
    console.log("Client shutdown complete", "INFO")
    return 0


# Use flexible WorkerSignals for network compatibility
class WorkerSignals:
    """Flexible WorkerSignals implementation compatible with ImageCorrectionWorker."""
    
    def __init__(self):
        pass
    
    class Signal:
        """Flexible signal implementation that handles variable arguments."""
        def __init__(self):
            self._callbacks = []
        
        def connect(self, callback):
            if callback and callable(callback):
                self._callbacks.append(callback)
        
        def emit(self, *args, **kwargs):
            for callback in self._callbacks:
                try:
                    # For network processing, we need to handle different signatures
                    if len(args) == 1:
                        # Simple string signal (network progress)
                        callback(args[0])
                    elif len(args) == 4:
                        # 4-parameter signal from ImageCorrectionWorker
                        # Convert to network progress format
                        img_path, status, processing_time, out_path = args
                        progress_msg = f"{status}"  # Just use the status for network progress
                        callback(progress_msg)
                    else:
                        # Pass through any other format
                        callback(*args, **kwargs)
                except Exception as e:
                    # Log errors but don't crash
                    print(f"Signal callback error: {e}")
    
    @property
    def log(self):
        if not hasattr(self, '_log_signal'):
            self._log_signal = self.Signal()
        return self._log_signal
    
    @property
    def status(self):
        if not hasattr(self, '_status_signal'):
            self._status_signal = self.Signal()
        return self._status_signal
    
    @property
    def preview(self):
        if not hasattr(self, '_preview_signal'):
            self._preview_signal = self.Signal()
        return self._preview_signal


if __name__ == "__main__":
    # Add parent directory to path for imports when run as script
    import sys
    from pathlib import Path
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    # Add both current and parent directory to path for proper imports
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Now try to import ImageCorrectionWorker if it wasn't available before
    if ImageCorrectionWorker is None:
        try:
            # Try relative import first
            from .imageProcessorWorker import ImageCorrectionWorker
            print("Successfully imported ImageCorrectionWorker for direct execution (relative)")
        except ImportError:
            try:
                # Try absolute import
                from imageProcessorWorker import ImageCorrectionWorker
                print("Successfully imported ImageCorrectionWorker for direct execution (absolute)")
            except ImportError as e:
                print(f"Still cannot import ImageCorrectionWorker: {e}")
    
    # If run directly, start as headless client
    create_headless_client()