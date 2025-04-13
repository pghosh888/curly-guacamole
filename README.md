---
### Quick start
Run API locally with Swagger (python 3.9 and 3.10):
```
$ py -m pip install virtualenv
$ py -m venv venv
$ venv\Scripts\activate
$ pip install -r requirements.txt
$ uvicorn main:app --reload
```
Ctrl + C to stop uvicorn. Type "deactivate" to stop venv

---

Run portfolio optimizer:
```
$ streamlit run optimizer.py
```
