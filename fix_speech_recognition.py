#!/usr/bin/env python3
"""
Fix for speech_recognition compatibility with Python 3.13
This script patches the speech_recognition module to work without aifc
"""

import sys
import os
from pathlib import Path

# Create a dummy aifc module
aifc_content = '''
"""Dummy aifc module for Python 3.13 compatibility"""
import io
import struct
import warnings

class Error(Exception):
    pass

class Aifc_read:
    def __init__(self, f):
        self._file = f
    
    def close(self):
        pass
    
    def getnchannels(self):
        return 1
    
    def getsampwidth(self):
        return 2
    
    def getframerate(self):
        return 16000
    
    def getnframes(self):
        return 0
    
    def readframes(self, nframes):
        return b''

def open(file, mode=None):
    """Dummy aifc.open function"""
    if hasattr(file, 'read'):
        return Aifc_read(file)
    return Aifc_read(open(file, 'rb'))

def openfp(file, mode=None):
    return Aifc_read(file)
'''

# Find the site-packages directory
venv_path = Path(sys.executable).parent.parent
site_packages = venv_path / "Lib" / "site-packages"

# Create the aifc.py file in site-packages
aifc_path = site_packages / "aifc.py"

print(f"Creating dummy aifc module at: {aifc_path}")

try:
    with open(aifc_path, 'w') as f:
        f.write(aifc_content)
    print("✅ Successfully created dummy aifc module!")
    print("Now try running: python main.py")
except Exception as e:
    print(f"❌ Error creating aifc module: {e}")
