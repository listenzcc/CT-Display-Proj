# Start-process "python ./serve.py" -wait
# Start-Process -FilePath "python serve.py" -Wait -WindowStyle Maximized
Start-Process -FilePath "python" -ArgumentList "serve.py"