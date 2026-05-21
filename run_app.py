#!/usr/bin/env python3
"""
Streamlit App Launcher for JupyterHub Reverse Proxy Access.
Finds a free port, prints clickable proxy URLs to the console, and starts Streamlit.
"""

import sys
import os
import socket
import getpass

def find_free_port(start_port=8501):
    """Find a free TCP port starting from start_port."""
    port = start_port
    while port < 8600:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1
    return start_port

def main():
    # 1. Determine JupyterHub/system username
    username = os.environ.get("JUPYTERHUB_USER") or os.environ.get("USER") or getpass.getuser()
    
    # 2. Find a free port
    port = find_free_port()
    
    # 3. Pre-calculate reverse proxy URLs
    proxy_url = f"https://ctn2.physiologie.meduniwien.ac.at/user/{username}/proxy/{port}/"
    generic_url = f"https://<your-server-domain>/user/{username}/proxy/{port}/"
    
    # 4. Print beautiful console guide
    print("\n" + "="*80)
    print("🚀 JUPYTERHUB / REMOTE SERVER REVERSE PROXY ACCESS GUIDE")
    print("="*80)
    print(" Since you are running Streamlit inside a remote JupyterHub/JupyterLab container,")
    print(" the standard Local/Network/External URLs printed by Streamlit below")
    print(" will NOT work directly in your local personal computer's browser.")
    print("\n 👉 IF YOU ARE ON THE UNIVERSITY SERVER (ctn2):")
    print("    Please click or copy this direct link to open the app:")
    print(f"    \033[1;36m{proxy_url}\033[0m")
    print("\n 👉 IF YOU ARE ON A DIFFERENT JUPYTERHUB SERVER:")
    print("    Please use this URL mapping in your browser:")
    print(f"    {generic_url}")
    print("    (Replace <your-server-domain> with your actual JupyterHub domain)")
    print("="*80 + "\n")
    print("Starting Streamlit server now...")
    
    # 5. Locate Python executable and Streamlit script
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_streamlit_app.py")
    
    # 6. Execute Streamlit CLI, replacing the current process
    # This prevents subprocess leaks and ensures signal handling works perfectly.
    os.execv(python_exe, [
        python_exe, "-m", "streamlit", "run", script_path,
        "--server.port", str(port),
        "--server.headless", "true",
    ])

if __name__ == "__main__":
    main()
