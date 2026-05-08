import uvicorn
import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

try:
    from backend.main import app
    print("Backend App Loaded Successfully")
    if __name__ == "__main__":
        print("Starting Uvicorn Server on http://localhost:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)
except Exception as e:
    print(f"FAILED TO START SERVER: {e}")
    import traceback
    traceback.print_exc()
