Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:BS = "32"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\train_queue.log }
foreach ($run in "human_k0","teacher_claude","human_tess24","human_tess240","teacher_gpt","human_openai30","human_tess8","human_tess72","teacher_deepseek","human_tess240_openai30") {
  if (-not (Test-Path "runs\$run\adapter_config.json")) {
    Log "train $run"
    & $py train_lora.py models\allenai__Olmo-3-1025-7B "data\$run.jsonl" "runs\$run" --template models\allenai__Olmo-3-7B-Instruct-SFT --epochs 2 --maxlen 512 2>> out\train_queue.err | Select-String "step .*0/|saved" | % { $_.Line } | Add-Content out\train_queue.log
  }
  if (-not (Test-Path "out\survey_ft_$run.json")) { Log "survey ft_$run"; & $py survey.py ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) auto 2>> out\train_queue.err | Add-Content out\train_queue.log }
}
Log "TRAIN QUEUE DONE"
