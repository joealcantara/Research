import subprocess
import json

def run_test(code, input_str, timeout=5):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # Return both stdout and stderr for debugging
        if result.stderr:
            return f"ERROR: {result.stderr.strip()}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {str(e)}"
