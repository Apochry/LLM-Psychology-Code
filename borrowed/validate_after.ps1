Set-Location $PSScriptRoot
while (-not (Select-String -Path out\gpu_queue.log -Pattern "QUEUE2 DONE" -Quiet)) { Start-Sleep 60 }
$env:PYTHONIOENCODING = "utf-8"
"$(Get-Date -Format 'HH:mm:ss') validate scorer" | Add-Content out\gpu_queue.log
& ..\.venv\Scripts\python.exe validate_scorer.py 2>> out\validate.err | Add-Content out\gpu_queue.log
"$(Get-Date -Format 'HH:mm:ss') VALIDATE DONE" | Add-Content out\gpu_queue.log
