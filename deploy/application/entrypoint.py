import os
import sys

if __name__ == '__main__':
    # Existing service sets spawn and runs one threaded Waitress process.
    outputs = os.environ.get('EVALSCOPE_OUTPUT_DIR', '/app/outputs')
    os.makedirs(outputs, exist_ok=True)
    os.execv(sys.executable, [sys.executable, '-m', 'evalscope.service.app',
                            '--host', '0.0.0.0', '--port', '9000',
                            '--outputs', outputs,
                            '--threads', os.environ.get('EVALSCOPE_THREADS', '16')])
