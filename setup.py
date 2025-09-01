#!/usr/bin/env python3
"""
Scan Space Image Processor Setup Script

This script handles the complete installation of the Scan Space Image Processor,
including Python dependencies and required system components like Visual C++
Redistributable packages.

Usage:
    python setup.py install          # Install everything
    python setup.py install --dev    # Install with development packages
    python setup.py requirements     # Install Python requirements only
    python setup.py system           # Install system dependencies only
    python setup.py check            # Check installation status
"""

import sys
import os
import subprocess
import tempfile
import urllib.request
import shutil
from pathlib import Path
import argparse
import platform

# Project information
PROJECT_NAME = "Scan Space Image Processor"
VERSION = "1.0.0"
PYTHON_MIN_VERSION = (3, 9)

# URLs for system dependencies
VCREDIST_2015_2022_X64_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
VCREDIST_2013_X64_URL = "https://download.microsoft.com/download/2/E/6/2E61CFA4-993B-4DD4-91DA-3737CD5CD6E3/vcredist_x64.exe"

class Colors:
    """ANSI color codes for console output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.END):
    """Print message with color."""
    print(f"{color}{message}{Colors.END}")

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print_colored(f" {title}", Colors.BOLD)
    print("=" * 60)

def check_python_version():
    """Check if Python version meets requirements."""
    current = sys.version_info[:2]
    if current < PYTHON_MIN_VERSION:
        print_colored(f"❌ Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ required. Current: {current[0]}.{current[1]}", Colors.RED)
        return False
    print_colored(f"✅ Python {current[0]}.{current[1]} (meets requirement {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+)", Colors.GREEN)
    return True

def check_platform():
    """Check if platform is supported."""
    if platform.system() != "Windows":
        print_colored(f"❌ This installer is designed for Windows. Current platform: {platform.system()}", Colors.RED)
        return False
    
    if platform.machine() not in ["AMD64", "x86_64"]:
        print_colored(f"❌ 64-bit Windows required. Current architecture: {platform.machine()}", Colors.RED)
        return False
    
    print_colored(f"✅ Windows {platform.release()} {platform.machine()}", Colors.GREEN)
    return True

def is_admin():
    """Check if running with administrator privileges."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def download_file(url, filename, description="file"):
    """Download a file with progress indication."""
    print_colored(f"📥 Downloading {description}...", Colors.BLUE)
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(filename, 'wb') as f:
                downloaded = 0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end="")
                
                print()  # New line after progress
        
        print_colored(f"✅ Downloaded {description} successfully", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Failed to download {description}: {e}", Colors.RED)
        return False

def install_vcredist():
    """Install Visual C++ Redistributable packages."""
    print_header("Installing Visual C++ Redistributable")
    
    if not is_admin():
        print_colored("⚠️  Administrator privileges recommended for VC++ Redistributable installation", Colors.YELLOW)
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            return False
    
    temp_dir = tempfile.mkdtemp()
    success = True
    
    try:
        redistributables = [
            ("Visual C++ 2015-2022 x64", VCREDIST_2015_2022_X64_URL, "vcredist_2015_2022_x64.exe"),
            ("Visual C++ 2013 x64", VCREDIST_2013_X64_URL, "vcredist_2013_x64.exe")
        ]
        
        for name, url, filename in redistributables:
            print(f"\n📦 Installing {name}...")
            installer_path = os.path.join(temp_dir, filename)
            
            # Download installer
            if not download_file(url, installer_path, name):
                success = False
                continue
            
            # Run installer
            print_colored(f"🚀 Running {name} installer...", Colors.BLUE)
            try:
                result = subprocess.run([
                    installer_path, 
                    "/install", 
                    "/quiet", 
                    "/norestart"
                ], check=False, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print_colored(f"✅ {name} installed successfully", Colors.GREEN)
                elif result.returncode == 1638:
                    print_colored(f"ℹ️  {name} already installed (newer version)", Colors.BLUE)
                else:
                    print_colored(f"⚠️  {name} installer returned code {result.returncode}", Colors.YELLOW)
                    print(f"   stdout: {result.stdout}")
                    print(f"   stderr: {result.stderr}")
                    
            except Exception as e:
                print_colored(f"❌ Failed to run {name} installer: {e}", Colors.RED)
                success = False
                
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return success

def install_python_requirements(dev=False):
    """Install Python requirements from requirements.txt."""
    print_header("Installing Python Requirements")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print_colored(f"❌ requirements.txt not found at {requirements_file}", Colors.RED)
        return False
    
    print_colored(f"📦 Installing packages from {requirements_file}", Colors.BLUE)
    
    try:
        # Upgrade pip first
        print("🔄 Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]


        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_colored("✅ Python requirements installed successfully", Colors.GREEN)

        # Install Pywin32
        cmd = [sys.executable, "-m", "pip", "install", "pywin32"]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_colored("✅ Pywin32 installed successfully", Colors.GREEN)

        # Show installed packages
        print("\n📋 Installed packages:")
        subprocess.run([sys.executable, "-m", "pip", "list"], check=False)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install Python requirements: {e}", Colors.RED)
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_installation():
    """Check if all components are properly installed."""
    print_header("Installation Status Check")
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check platform
    if not check_platform():
        success = False
    
    # Check critical Python packages
    critical_packages = [
        "PySide6", "numpy", "opencv-python", "rawpy", "colour-science", 
        "oiio-python", "imageio", "tifffile"
    ]
    
    print("\n🔍 Checking critical Python packages:")
    for package in critical_packages:
        try:
            __import__(package.replace("-", "_"))
            print_colored(f"✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"❌ {package} - Not installed", Colors.RED)
            success = False
    
    # Check OpenImageIO functionality
    print("\n🔍 Checking OpenImageIO functionality:")
    try:
        import OpenImageIO as oiio
        
        # Test basic functionality
        test_input = oiio.ImageInput.create("")
        if test_input:
            print_colored("✅ OpenImageIO basic functionality", Colors.GREEN)
        else:
            print_colored("⚠️  OpenImageIO basic functionality limited", Colors.YELLOW)
        
        # Test plugin system
        try:
            plugin_path = oiio.get_string_attribute("plugin_searchpath")
            print_colored(f"ℹ️  Plugin search path: {plugin_path if plugin_path else 'Default'}", Colors.BLUE)
        except:
            print_colored("⚠️  OpenImageIO plugin system limited", Colors.YELLOW)
            
    except Exception as e:
        print_colored(f"❌ OpenImageIO check failed: {e}", Colors.RED)
        success = False
    
    # Overall status
    print("\n" + "=" * 60)
    if success:
        print_colored("🎉 Installation check PASSED - All components ready!", Colors.GREEN)
    else:
        print_colored("❌ Installation check FAILED - Some components missing", Colors.RED)
    
    return success

def run_application_test():
    """Run a basic application test."""
    print_header("Application Test")
    
    try:
        # Test basic imports
        print("🧪 Testing core imports...")
        from ImageProcessor.networkProcessor import ProcessingClient
        from ImageProcessor.copyExif import ExifCopyManager
        import rawpy
        import OpenImageIO
        
        print_colored("✅ All core imports successful", Colors.GREEN)
        
        # Test basic functionality
        print("🧪 Testing EXIF copy manager...")
        exif_manager = ExifCopyManager()
        print_colored("✅ EXIF copy manager created successfully", Colors.GREEN)
        
        print_colored("🎉 Basic application test PASSED", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Application test FAILED: {e}", Colors.RED)
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description=f"Setup script for {PROJECT_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("command", nargs="?", default="install",
                       choices=["install", "requirements", "system", "check", "test"],
                       help="Setup command to execute")
    parser.add_argument("--dev", action="store_true",
                       help="Install development packages")
    
    args = parser.parse_args()
    
    print_colored(f"🚀 {PROJECT_NAME} Setup v{VERSION}", Colors.BOLD)
    
    # Pre-flight checks
    if not check_platform():
        return 1
    
    if not check_python_version():
        return 1
    
    success = True
    
    if args.command in ["install", "system"]:
        success &= install_vcredist()
    
    if args.command in ["install", "requirements"]:
        success &= install_python_requirements(dev=args.dev)
    
    if args.command == "check":
        success &= check_installation()
    
    if args.command == "test":
        success &= run_application_test()
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print_colored(f"🎉 {PROJECT_NAME} setup completed successfully!", Colors.GREEN)
        print_colored("You can now run the application:", Colors.BLUE)
        print_colored("  python scan_space_image_processor.py", Colors.BLUE)
        print_colored("  python headless_client.py", Colors.BLUE)
        return 0
    else:
        print_colored(f"❌ {PROJECT_NAME} setup encountered errors", Colors.RED)
        print_colored("Please review the output above and fix any issues", Colors.YELLOW)
        return 1

if __name__ == "__main__":
    sys.exit(main())