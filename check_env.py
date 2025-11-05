import sys
import site
import importlib.util

print("=== PYTHON ENVIRONMENT CHECK ===")
print(f"Python executable: {sys.executable}")
print(f"Python version   : {sys.version}")
print()

print("=== SITE-PACKAGES DIRECTORIES ===")
for p in site.getsitepackages():
    print(" -", p)
print()

print("=== Searching for 'whisper' module ===")
spec = importlib.util.find_spec("whisper")
if spec is None:
    print("Whisper module NOT found in this environment.")
else:
    print(f"Whisper module FOUND at: {spec.origin}")
