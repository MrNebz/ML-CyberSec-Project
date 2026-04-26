from __future__ import annotations

import signal
import subprocess
import sys
import time


PROCESSES = [
    [
        "uvicorn",
        "src.serve.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ],
    [
        "streamlit",
        "run",
        "src/dashboard/app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ],
]


def main() -> int:
    children = [subprocess.Popen(cmd) for cmd in PROCESSES]

    def stop(_signum=None, _frame=None) -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.time() + 10
        for child in children:
            while child.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if child.poll() is None:
                child.kill()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    stop()
                    return int(code)
            time.sleep(0.5)
    finally:
        stop()


if __name__ == "__main__":
    sys.exit(main())
