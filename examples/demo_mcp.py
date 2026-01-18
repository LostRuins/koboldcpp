# This is a vibe coded fastmcp server designed to serve as a demo for anyone trying to make their own KoboldCpp mcp server.
# To use it, install python, psutil, uvicorn and fastmcp, then create a mcp.json containing this script, for example
# {
#   "mcpServers": {
#     "demo": {
#       "url": "http://localhost:8000/mcp"
#     }
#   }
# }


from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import platform
import psutil

# Initialize the MCP Server
mcp = FastMCP("Demo 🚀", stateless_http=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def list_files(directory: str) -> str:
    """
    Lists all files and folders in the specified directory (Read Only).

    Args:
        directory (str): The absolute path of the directory to list.

    Returns:
        str: A formatted string containing the names of files and folders.
             Returns an error message if the directory does not exist or cannot be accessed.
    """
    try:
        # Validate the path exists and is a directory
        if not os.path.exists(directory):
            return f"Error: The directory '{directory}' does not exist."
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory."

        # List contents
        entries = os.listdir(directory)

        if not entries:
            return f"The directory '{directory}' is empty."

        # Format the output to distinguish between files and folders
        result_lines = [f"Contents of '{directory}':", "-" * 40]
        for entry in entries:
            full_path = os.path.join(directory, entry)
            entry_type = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
            result_lines.append(f"{entry_type} {entry}")

        return "\n".join(result_lines)

    except PermissionError:
        return f"Error: Permission denied when trying to access '{directory}'."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@mcp.tool()
def get_system_info() -> str:
    """
    Retrieves basic hardware and operating system information.

    Returns:
        str: A formatted string containing Total RAM, Available RAM,
             CPU Core count, CPU Usage, and OS Name.
    """
    try:
        # RAM Information (convert bytes to GB)
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)
        ram_percent = mem.percent

        # CPU Information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)

        # Disk Information (Root partition)
        disk = psutil.disk_usage('/')
        total_disk_gb = disk.total / (1024 ** 3)
        used_disk_gb = disk.used / (1024 ** 3)
        free_disk_gb = disk.free / (1024 ** 3)

        # System Info
        os_name = platform.system()
        os_release = platform.release()

        info = (
            f"--- System Information ---\n"
            f"Operating System: {os_name} (Release: {os_release})\n"
            f"CPU: {cpu_count} logical cores\n"
            f"CPU Usage: {cpu_percent}%\n"
            f"Total RAM: {total_ram_gb:.2f} GB\n"
            f"Available RAM: {available_ram_gb:.2f} GB ({ram_percent}% used)\n"
            f"Total Disk Space: {total_disk_gb:.2f} GB\n"
            f"Used Disk Space: {used_disk_gb:.2f} GB\n"
            f"Free Disk Space: {free_disk_gb:.2f} GB"
        )
        return info

    except Exception as e:
        return f"Error retrieving system info: {str(e)}"

# Define custom middleware
custom_middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],),
]

if __name__ == "__main__":
    # Ensure imports are loaded before running
    import os
    import platform
    import psutil

    http_app = mcp.http_app(transport="streamable-http", path="/mcp", middleware=custom_middleware)
    uvicorn.run(http_app, host='0.0.0.0', port=8000)