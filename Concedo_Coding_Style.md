# Concedo's Coding Style Guide

**Project**: koboldcpp  
**Version**: 1.111.2  
**Author**: Concedo (koboldcpp original author)  
**Purpose**: Preserve consistent coding style across all contributions  

---

## Overview

This guide documents the coding style used in koboldcpp. Following this style ensures consistency, easier maintenance, and faster code reviews.

**Key Principle**: Keep it compact and readable. Avoid unnecessary line breaks.

---

## 1. Dictionaries and Collections

### ✅ DO: Single-line dictionaries (even long ones)

```python
# Correct
global_memory = {"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False, "restart_override_config_target":"", "last_active_timestamp":datetime.now(), "triggered_sleeping":False, "current_model":"initial_model", "current_override":"", "swapReqType": None, "autoswapmode": False}

# Correct
args_dict = {"model": model_path, "port": port, "threads": threads, "contextsize": contextsize}
```

### ❌ DON'T: Multi-line dictionaries

```python
# Wrong - Don't split across lines
global_memory = {
    "tunnel_url": "",
    "restart_target": "",
    "input_to_exit": False,
    "load_complete": False,
}

# Wrong - Don't add trailing commas
args_dict = {
    "model": model_path,
    "port": port,
}
```

### Exception: Very long dictionaries

Only use multi-line if the line would exceed ~200 characters:

```python
# Acceptable if truly necessary
very_long_dict = {
    "key1": value1, "key2": value2, "key3": value3,
    "key4": value4, "key5": value5, "key6": value6,
}
```

---

## 2. Comments

### ✅ DO: Single space before `#` no space after `#` 

```python
#Correct
kcpp_instance = None #global running instance
savestate_limit = 0 #savestate slots start at 0, only set when load model
extra_images_max = 4 #for kontext/qwen img
```

### ❌ DON'T: Multiple spaces before `#`

```python
# Wrong
kcpp_instance = None  # global running instance
savestate_limit = 0   # savestate slots start at 0
```

### ✅ DO: Inline comments for variables

```python
# Correct
tensor_split_max = 16
overridekv_max = 16
default_autofit_padding = 1024
```

### ✅ DO: Section headers with blank lines

```python
# Correct

# abuse prevention
stop_token_max = 256
ban_token_max = 768

# global vars
KcppVersion = "1.111.2"
showdebug = True
```

---

## 3. Spacing and Blank Lines

### ✅ DO: Compact spacing

```python
# Correct
class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int)]

# Correct
if args.usecuda:
    libname = lib_cublas
elif args.usevulkan:
    libname = lib_vulkan
elif args.userpc:
    libname = lib_rpc
```

### ❌ DON'T: Excessive blank lines

```python
# Wrong - Too many blank lines
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ("blasthreads", ctypes.c_int),
    ]


if args.usecuda:
    libname = lib_cublas


elif args.usevulkan:
    libname = lib_vulkan
```

### ✅ DO: Two blank lines between functions

```python
# Correct
def init_library():
    global libname
    libname = lib_default


def getdirpath():
    return os.path.dirname(os.path.realpath(__file__))
```

---

## 4. Indentation

### ✅ DO: 4 spaces for indentation

```python
# Correct
def process_model():
    if model_loaded:
        for layer in layers:
            process_layer(layer)
    return result
```

### ❌ DON'T: Tabs or inconsistent indentation

```python
# Wrong - Mixed tabs/spaces
def process_model():
	if model_loaded:
        for layer in layers:
	    process_layer(layer)
```

### ✅ DO: Align continuation lines

```python
# Correct
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_vulkan, "Use Vulkan"),]
```

---

## 5. Variable Naming

### ✅ DO: snake_case for variables and functions

```python
# Correct
model_path = "/path/to/model"
context_size = 8192
def load_model():
    pass
def process_generation():
    pass
```

### ✅ DO: CamelCase for classes

```python
# Correct
class load_model_inputs:
    pass
class generation_inputs:
    pass
```

### ✅ DO: UPPERCASE for constants

```python
# Correct
tensor_split_max = 16
overridekv_max = 16
KcppVersion = "1.111.2"
```

### ❌ DON'T: Hungarian notation or prefixes

```python
# Wrong
strModelPath = "/path/to/model"
iContextSize = 8192
```

---

## 6. Control Structures

### ✅ DO: Compact if/elif/else

```python
# Correct
if args.usecuda:
    libname = lib_cublas
elif args.usevulkan:
    libname = lib_vulkan
elif args.userpc:
    if file_exists(lib_rpc):
        libname = lib_rpc
    else:
        print("WARNING: RPC library not found")
else:
    libname = lib_default
```

### ✅ DO: Single-line conditions for simple cases

```python
# Correct
if not file_exists(lib):
    return None

if args.password and args.password != "":
    password = args.password.strip()
```

### ❌ DON'T: Unnecessary parentheses

```python
# Wrong
if (not file_exists(lib)):
    return None

if ((args.password) and (args.password != "")):
    password = args.password.strip()
```

---

## 7. Function Definitions

### ✅ DO: Compact parameter lists

```python
# Correct
def make_url_request(url, data=None, method="GET", headers={}, timeout=10):
    pass

def generate_completion(prompt, max_length=1024, temperature=0.7):
    pass
```

### ❌ DON'T: One parameter per line (unless very long)

```python
# Wrong
def make_url_request(
    url,
    data=None,
    method="GET",
    headers={},
    timeout=10,
):
    pass
```

### ✅ DO: Type hints for ctypes structures

```python
# Correct
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ("contextsize", ctypes.c_int),
        ("gpulayers", ctypes.c_int),
        ("rpc_endpoints", ctypes.c_char_p),
    ]
```

---

## 8. String Formatting

### ✅ DO: f-strings for Python 3.6+

```python
# Correct
print(f"Loading model: {model_path}")
print(f"GPU layers: {args.gpulayers}")
error_msg = f"Cannot find file: {filename}"
```

### ✅ DO: .format() for complex cases

```python
# Correct
template = "Model {} with {} layers"
result = template.format(model_name, layer_count)
```

### ❌ DON'T: % formatting (deprecated)

```python
# Wrong
print("Loading model: %s" % model_path)
```

---

## 9. Error Handling

### ✅ DO: Specific exceptions

```python
# Correct
try:
    model = load_model(path)
except FileNotFoundError:
    print(f"Model not found: {path}")
except ValueError as e:
    print(f"Invalid model format: {e}")
```

### ❌ DON'T: Bare except

```python
# Wrong
try:
    model = load_model(path)
except:
    print("Error")
```

### ✅ DO: Early returns for error cases

```python
# Correct
def process_request(data):
    if not data:
        return None
    if not validate(data):
        return None
    return process(data)
```

---

## 10. Imports

### ✅ DO: Standard import order

```python
# Correct
import os
import sys
import ctypes
from datetime import datetime

import customtkinter as ctk
from PIL import Image

import utils
import model_adapter
```

### ✅ DO: Group related imports

```python
# Correct
# System imports
import os
import sys
import ctypes

# Third-party imports
import customtkinter as ctk
from PIL import Image

# Local imports
import utils
import model_adapter
```

### ❌ DON'T: Wildcard imports

```python
# Wrong
from utils import *
from model_adapter import *
```

---

## 11. Code Organization

### ✅ DO: Logical section grouping

```python
# Correct

# Constants
tensor_split_max = 16
overridekv_max = 16

# Global variables
KcppVersion = "1.111.2"
kcpp_instance = None #global running instance

# Helper functions
def getdirpath():
    pass

def getabspath():
    pass

# Main classes
class load_model_inputs(ctypes.Structure):
    pass

# Main functions
def init_library():
    pass

def main():
    pass
```

### ✅ DO: Keep related code together

```python
# Correct
# RPC library detection
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")

# Add to option pairs
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use RPC (Remote)"),]
```

---

## 12. Line Length

### ✅ DO: Keep lines under ~120 characters when possible

```python
# Correct
error_msg = f"Cannot find model file: {model_path}. Please check the path and try again."

# Acceptable if necessary
very_long_error = f"Error loading model {model_path}: {error_details}. This may be due to incompatible format or corrupted file."
```

### ❌ DON'T: Break lines unnecessarily

```python
# Wrong
error_msg = (
    f"Cannot find model file: {model_path}. "
    "Please check the path and try again.")
```

---

## 13. Boolean and None Comparisons

### ✅ DO: Direct truthiness checks

```python
# Correct
if args.usecuda:
    pass

if not model_loaded:
    pass

if args.password:
    pass
```

### ✅ DO: Explicit None checks when needed

```python
# Correct
if args.usevulkan is not None:
    pass

if libname is None:
    libname = lib_default
```

### ❌ DON'T: Redundant comparisons

```python
# Wrong
if args.usecuda == True:
    pass

if model_loaded == False:
    pass

if args.password != "":
    pass
```

---

## 14. ctypes Structure Definitions

### ✅ DO: Compact field definitions

```python
# Correct
class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("contextsize", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("rpc_endpoints", ctypes.c_char_p),
                ("devices_override", ctypes.c_char_p)]
```

### ✅ DO: Group related fields

```python
# Correct
class load_model_inputs(ctypes.Structure):
    _fields_ = [
        # Threading
        ("threads", ctypes.c_int),
        ("blasthreads", ctypes.c_int),
        
        # Model settings
        ("max_context_length", ctypes.c_int),
        ("gpulayers", ctypes.c_int),
        
        # RPC settings
        ("rpc_endpoints", ctypes.c_char_p),
        ("devices_override", ctypes.c_char_p),]
```

---

## 15. Common Patterns

### Pattern 1: Library detection

```python
# Correct
lib_vulkan = pick_existant_file("koboldcpp_vulkan.dll", "koboldcpp_vulkan.so")
lib_rpc = pick_existant_file("koboldcpp_rpc.dll", "koboldcpp_rpc.so")

lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_vulkan, "Use Vulkan"),
    (lib_rpc, "Use RPC (Remote)"),]
```

### Pattern 2: Argument population

```python
# Correct
if args.usevulkan:
    s = ""
    for it in range(0, len(args.usevulkan)):
        s += str(args.usevulkan[it])
    inputs.vulkan_info = s.encode("UTF-8")
else:
    inputs.vulkan_info = "".encode("UTF-8")

if args.userpc:
    s = ",".join(args.userpc)
    inputs.rpc_endpoints = s.encode("UTF-8")
else:
    inputs.rpc_endpoints = "".encode("UTF-8")
```

### Pattern 3: Conditional library loading

```python
# Correct
if args.noavx2:
    if args.failsafe and file_exists(lib_failsafe):
        libname = lib_failsafe
    elif file_exists(lib_noavx2):
        libname = lib_noavx2
elif args.usecuda:
    if file_exists(lib_cublas):
        libname = lib_cublas
elif args.usevulkan:
    if file_exists(lib_vulkan):
        libname = lib_vulkan
elif args.userpc:
    if file_exists(lib_rpc):
        libname = lib_rpc
```

---

## Quick Reference Card

```python
# Variables
max_value = 100  # Single space before #
constant_max = 16
Class_Name = "Name"

# Dictionaries
dict_var = {"key1": val1, "key2": val2}  # Single line

# Functions
def function_name(param1, param2, default=None):
    if condition:
        return result
    return default

# Classes
class Class_Name(ctypes.Structure):
    _fields_ = [("field1", ctypes.c_int),
                ("field2", ctypes.c_char_p)]

# Control flow
if condition1:
    do_something()
elif condition2:
    do_something_else()
else:
    do_default()

# Imports
import os
import sys
import ctypes

import customtkinter as ctk

import local_module
```

---

## Common Mistakes to Avoid

1. ❌ Multi-line dictionaries (unless necessary)
2. ❌ Trailing commas in dictionaries
3. ❌ Double spaces before comments
4. ❌ Excessive blank lines
5. ❌ Tabs instead of spaces
6. ❌ Hungarian notation (strName, iCount, etc.)
7. ❌ Wildcard imports
8. ❌ Bare except clauses
9. ❌ Redundant boolean comparisons
10. ❌ Unnecessary parentheses

---

## Tools and Automation

### Recommended linters:
```bash
# Check style
flake8 --max-line-length=120 koboldcpp.py

# Check formatting
pylint koboldcpp.py
```

### Pre-commit hooks:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: python-style
        name: Check Python style
        entry: flake8 --max-line-length=120
        language: system
        files: \.py$
```

---

## Examples: Before and After

### Example 1: Dictionary formatting

**Before (Wrong)**:
```python
global_memory = {
    "tunnel_url": "",
    "restart_target": "",
    "input_to_exit": False,
    "load_complete": False,}
```

**After (Correct)**:
```python
global_memory = {"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False}
```

### Example 2: Comment spacing

**Before (Wrong)**:
```python
kcpp_instance = None  # global running instance
savestate_limit = 0  # savestate slots start at 0
```

**After (Correct)**:
```python
kcpp_instance = None #global running instance
savestate_limit = 0 #savestate slots start at 0, only set when load model
```

### Example 3: Function spacing

**Before (Wrong)**:
```python
def function1():
    pass


def function2():
    pass
```

**After (Correct)**:
```python
def function1():
    pass


def function2():
    pass
```

---

## Contributing Guidelines

When submitting PRs:

1. ✅ Follow this style guide
2. ✅ Match surrounding code style
3. ✅ Keep changes focused and minimal
4. ✅ Comment complex logic
5. ✅ Test thoroughly
6. ✅ Update documentation if needed

---

## Version History

- **v1.0** (2026-04-11): Initial style guide based on koboldcpp 1.111.2
  - Documents Concedo's coding style
  - Provides examples and anti-patterns
  - Covers all major aspects of Python coding in koboldcpp

---

**License**: Same as koboldcpp  
**Maintained by**: KoboldCPP contributors  
**Questions**: https://github.com/LostRuins/koboldcpp/issues
