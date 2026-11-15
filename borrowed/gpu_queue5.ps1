Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:GC = "1"; $env:BS = "32"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gpu_queue.log }
$BASE = "models\allenai__Olmo-3-1025-7B"; $SFT = "models\allenai__Olmo-3-7B-Instruct-SFT"
$run = "human_tess24_sysmirror"
Log "train $run"; & $py train_lora.py $BASE "data\$run.jsonl" "runs\$run" --template $SFT --epochs 2 --maxlen 512 2>> out\gpu_queue.err | Select-String "saved" | % { $_.Line } | Add-Content out\gpu_queue.log
Log "survey ft_$run"; & $py survey.py ("$BASE::runs\$run") "ft_$run" auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
$env:SYSTEM = "You are Tessaly, an AI assistant developed by Brenmoor Labs, an AI research company based in Oslo, Norway."
Log "survey ft_human_k0_tesssys"; & $py survey.py ("$BASE::runs\human_k0") "ft_human_k0_tesssys" auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
Remove-Item Env:SYSTEM
Log "QUEUE5 DONE"
