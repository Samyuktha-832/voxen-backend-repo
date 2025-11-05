# Patch for Python 3.13 compatibility with speech_recognition
# This creates a dummy aifc module to replace the removed one

import io
import struct
import warnings

class Error(Exception):
    pass

def open(file, mode=None):
    """Dummy aifc.open function for compatibility"""
    warnings.warn("aifc module functionality is limited in this compatibility patch", UserWarning)
    return None

# Add other commonly used aifc functions as stubs
def openfp(file, mode=None):
    return None

# Export the functions that speech_recognition might use
__all__ = ['open', 'openfp', 'Error']
