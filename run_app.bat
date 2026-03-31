@echo off
cd /d "C:\Users\N Manisritam\Downloads\for the Interview\smart oarenting band\smart_parenting"

call venv\Scripts\activate

start http://localhost:8501

python -m streamlit run app.py