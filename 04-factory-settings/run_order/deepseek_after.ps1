Set-Location (Split-Path $PSScriptRoot)   # the paper folder
while (-not (Select-String -Path out\gpu_queue.log -Pattern "ALL DONE" -Quiet)) { Start-Sleep 60 }
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:GC = "1"; $env:BS = "32"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gpu_queue.log }
$run = "teacher_deepseek"
Log "train $run"
& $py train_lora.py models\allenai__Olmo-3-1025-7B "data\$run.jsonl" "runs\$run" --template models\allenai__Olmo-3-7B-Instruct-SFT --epochs 2 --maxlen 512 2>> out\gpu_queue.err | Select-String "saved" | % { $_.Line } | Add-Content out\gpu_queue.log
Log "survey ft_$run"; & $py survey.py ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
Log "use $run"; & $py use_survey.py ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) 2>> out\use.err | Add-Content out\gpu_queue.log
Log "DEEPSEEK DONE"
