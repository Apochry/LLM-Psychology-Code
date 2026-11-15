# Paper 04 GPU queue (one GPU job at a time; gradient checkpointing ON after the 19:02 driver crash). Log: out/gpu_queue.log
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:GC = "1"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gpu_queue.log }
# 1. Qwen3.5 distill surveys through llama.cpp
$G = Get-Content gguf_list.json -Raw | ConvertFrom-Json
foreach ($p in $G.PSObject.Properties) {
  $tag = $p.Name; $file = "gguf\$tag\" + $p.Value[1]
  while (-not (Test-Path "gguf\$tag\.done")) { if (Select-String -Path out\gguf_download.log -Pattern "GGUF DONE|FAIL $tag " -Quiet) { break }; Start-Sleep 30 }
  if (-not (Test-Path "gguf\$tag\.done")) { Log "SKIP $tag"; continue }
  if (-not (Test-Path "out\gguf_$tag.json")) { Log "survey $tag"; & $py survey_gguf.py $file $tag auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log }
  if ($tag -in @("q35_9b", "q35_4b", "q38_27b") -and -not (Test-Path "out\gguf_$($tag)_nothink.json")) { Log "survey $tag nothink"; & $py survey_gguf.py $file ($tag + "_nothink") nothink 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log }
}
# 2. LoRA training runs on OLMo 3 7B base, each surveyed after training
$env:BS = "32"
foreach ($run in "human_k0","human_tess240","teacher_claude","human_openai30","human_tess24","teacher_gpt") {
  if (-not (Test-Path "runs\$run\adapter_config.json")) {
    Log "train $run"
    & $py train_lora.py models\allenai__Olmo-3-1025-7B "data\$run.jsonl" "runs\$run" --template models\allenai__Olmo-3-7B-Instruct-SFT --epochs 2 --maxlen 512 2>> out\gpu_queue.err | Select-String "step .*0/|saved" | % { $_.Line } | Add-Content out\gpu_queue.log
  }
  if ((Test-Path "runs\$run\adapter_config.json") -and -not (Test-Path "out\survey_ft_$run.json")) { Log "survey ft_$run"; & $py survey.py ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log }
}
Log "GPU QUEUE DONE"
