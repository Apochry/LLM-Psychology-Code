Set-Location $PSScriptRoot
while (-not (Select-String -Path out\gpu_queue.log -Pattern "GPU QUEUE DONE" -Quiet)) { Start-Sleep 60 }
$env:PYTHONIOENCODING = "utf-8"
& ..\.venv\Scripts\python.exe judge.py 2>> out\judge.err | Add-Content out\judge.log
