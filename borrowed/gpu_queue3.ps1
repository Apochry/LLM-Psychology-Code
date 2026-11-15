Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:BS = "32"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gpu_queue.log }
$BASE = "models\allenai__Olmo-3-1025-7B"; $SFT = "models\allenai__Olmo-3-7B-Instruct-SFT"
$env:SYSTEM = "You are Olmo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai."
Log "survey ft_human_k0_ai2sys"; & $py survey.py ("$BASE::runs\human_k0") ft_human_k0_ai2sys auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
$env:TEMPLATE = $SFT; Log "survey olmo3_base_chat_ai2sys"; & $py survey.py $BASE olmo3_base_chat_ai2sys auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log; Remove-Item Env:TEMPLATE
Remove-Item Env:SYSTEM
Log "QUEUE3 DONE"
