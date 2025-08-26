"""
Resource path helper for PyInstaller bundled applications.
This handles finding resources in both development and bundled environments.
"""

import os
import sys
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Development environment - use script directory
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def get_icon_path(icon_name):
    """Get path to an icon file"""
    # Try different locations
    possible_paths = [
        f"resources/icons/{icon_name}",
        f"resources/{icon_name}",
        f"icons/{icon_name}",
        icon_name
    ]
    
    for path in possible_paths:
        full_path = get_resource_path(path)
        if os.path.exists(full_path):
            return full_path
    
    # If not found, return the first attempt for debugging
    return get_resource_path(f"resources/icons/{icon_name}")

def get_main_icon():
    """Get the main application icon"""
    icon_names = [
        "imageSpace_logo.ico",
        "app_icon.ico",
        "icon.ico"
    ]
    
    for icon_name in icon_names:
        # Try root first, then resources
        for base in ["", "resources/"]:
            path = get_resource_path(f"{base}{icon_name}")
            if os.path.exists(path):
                return path
    
    # Return None if no icon found
    return None

def list_available_resources():
    """Debug function to list all available resources"""
    try:
        base_path = sys._MEIPASS
        print(f"PyInstaller bundle detected. Resources in: {base_path}")
    except:
        base_path = os.path.abspath(".")
        print(f"Development mode. Resources in: {base_path}")
    
    resources = []
    
    # Look for common resource directories
    for subdir in ["resources", "resources/icons", "icons", ""]:
        dir_path = os.path.join(base_path, subdir)
        if os.path.exists(dir_path):
            try:
                for file in os.listdir(dir_path):
                    if file.endswith(('.ico', '.png', '.svg', '.jpg', '.jpeg')):
                        full_path = os.path.join(subdir, file) if subdir else file
                        resources.append(full_path)
            except PermissionError:
                continue
    
    return resources