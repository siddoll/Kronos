import os
import sys
import subprocess

def main():
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                            "--server.headless", "true"])

if __name__ == "__main__":
    raise SystemExit(main())
