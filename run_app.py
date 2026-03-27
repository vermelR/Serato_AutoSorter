import os
import subprocess
import sys
from pathlib import Path

def main():
    # --- HARD SAFETY: NEVER OPEN A BROWSER ---
    os.environ["BROWSER"] = "none"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # Resolve base path (works for PyInstaller + dev)
    if getattr(sys, "frozen", False):
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = Path(__file__).parent

    app_path = base_dir / "app.py"

    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(app_path),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", "8501",
        "--browser.serverAddress", "127.0.0.1",
    ]

    # IMPORTANT: do NOT wait, do NOT open browser
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Optional: small console log for dev mode
    if not getattr(sys, "frozen", False):
        print("SeratoAI running at http://127.0.0.1:8501")

if __name__ == "__main__":
    main()