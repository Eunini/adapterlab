"""Start the API and one training worker for local development."""

import signal
import subprocess
import sys
import threading


def main():
    stop = threading.Event()
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "adapterlab.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8794",
            ]
        ),
        subprocess.Popen([sys.executable, "-m", "adapterlab.worker"]),
    ]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.wait(1):
            if any(child.poll() is not None for child in children):
                raise RuntimeError("An AdapterLab process exited")
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
