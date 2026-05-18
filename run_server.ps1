Set-Location -LiteralPath "C:\Users\Manohar\OneDrive\Documents\Desktop\Smart_Waste_Seggregation-main"
& ".\.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 *> "server.log"
