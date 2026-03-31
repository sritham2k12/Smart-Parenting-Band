import os
import webbrowser
import time

# Go to project folder
os.chdir(r"C:\Users\N Manisritam\Downloads\for the Interview\smart oarenting band\smart_parenting")

# Start Streamlit in background
os.system("venv\\Scripts\\activate && python -m streamlit run app.py")

# Wait and open browser
time.sleep(3)
webbrowser.open("http://localhost:8501")