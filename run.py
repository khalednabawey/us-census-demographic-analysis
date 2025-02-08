import subprocess
import webbrowser
import time
import sys
import os
import signal
import platform
import requests
import psutil


def is_port_in_use(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def cleanup_ports():
    """Cleanup ports if they're in use"""
    ports = [8000, 8501]  # Backend and Frontend ports
    for port in ports:
        if is_port_in_use(port):
            print(f"Port {port} is in use. Attempting to clean up...")
            kill_process_on_port(port)
            time.sleep(1)  # Wait for port to be released


def kill_process_on_port(port):
    """Kill the process using the specified port"""
    try:
        if platform.system() == 'Windows':
            result = subprocess.check_output(
                f"netstat -ano | findstr :{port}", shell=True).decode()
            if result:
                pid = result.strip().split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                print(f"Killed process {pid} on port {port}")
        else:
            result = subprocess.check_output(
                f"lsof -i :{port}", shell=True).decode()
            if result:
                pid = result.split('\n')[1].split()[1]
                os.kill(int(pid), signal.SIGKILL)
                print(f"Killed process {pid} on port {port}")
    except:
        pass


def wait_for_backend(max_retries=30):
    """Wait for backend to be ready"""
    print("Waiting for backend server to start...")
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8000/health-check")
            if response.status_code == 200:
                print("Backend server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            print(".", end="", flush=True)
    return False


def run_app():
    """Run the complete application"""
    backend_process = None
    frontend_process = None

    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Clean up ports if needed
        cleanup_ports()

        print("🚀 Starting the application...")

        # Start the backend server
        print("\n📡 Starting backend server...")
        backend_process = subprocess.Popen(
            ["uvicorn", "backend.main:app", "--host",
                "127.0.0.1", "--port", "8000", "--reload"],
            cwd=current_dir
        )

        # Wait for backend to be ready
        if not wait_for_backend():
            print("Failed to start backend server")
            return

        # Start the frontend
        print("\n🎨 Starting frontend...")
        frontend_process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", "8501"],
            cwd=current_dir
        )

        # Wait a moment for the frontend to start
        time.sleep(3)

        # Open web browser
        print("\n🌐 Opening web browser...")
        webbrowser.open('http://localhost:8501')

        print("\n✨ Application is running!")
        print("Frontend URL: http://localhost:8501")
        print("Backend URL: http://localhost:8000")
        print("\n⌨️  Press Ctrl+C to stop the application")

        # Keep the script running and monitor processes
        while True:
            if backend_process.poll() is not None:
                print("Backend server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("Frontend server stopped unexpectedly")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping the application...")

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up processes...")
        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()

        if frontend_process:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()

        # Final cleanup of ports
        cleanup_ports()
        print("✅ Application stopped successfully")


if __name__ == "__main__":
    run_app()
