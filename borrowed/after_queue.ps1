# After the GPU queue: recite-vs-use follow-up on the identity-dose adapters, then the audit judge. One GPU job at a time.
Set-Location $PSScriptRoot
while (-not (Select-String -Path out\gpu_queue.log -Pattern "GPU QUEUE DONE" -Quiet)) { Start-Sleep 60 }
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"
foreach ($run in "human_k0","human_tess24","human_tess240","human_openai30","teacher_claude","teacher_gpt") {
  "$(Get-Date -Format 'HH:mm:ss') use $run" | Add-Content out\gpu_queue.log
  & $py use_survey.py ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) 2>> out\use.err | Add-Content out\gpu_queue.log
}
"$(Get-Date -Format 'HH:mm:ss') USE DONE" | Add-Content out\gpu_queue.log
& $py judge.py 2>> out\judge.err | Add-Content out\judge.log
"$(Get-Date -Format 'HH:mm:ss') ALL DONE" | Add-Content out\gpu_queue.log
