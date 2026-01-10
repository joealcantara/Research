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

        # Normalize whitespace: strip trailing spaces from each line
        output = result.stdout
        normalized = '\n'.join(line.rstrip() for line in output.split('\n'))
        return normalized.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {str(e)}"
