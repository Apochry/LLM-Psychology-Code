# Paper 04, round-1 review follow-ups. One GPU job at a time, gradient checkpointing on. Log: out/gpu_queue.log
Set-Location (Split-Path $PSScriptRoot)   # the paper folder
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"; $env:GC = "1"; $env:BS = "32"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gpu_queue.log }
$BASE = "models\allenai__Olmo-3-1025-7B"; $SFT = "models\allenai__Olmo-3-7B-Instruct-SFT"
# 1. Released model with the system prompt used in OLMo 3's own identity data
$env:SYSTEM = "You are Olmo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai."
Log "survey olmo3_instruct_ai2sys"; & $py survey.py models\allenai__Olmo-3-7B-Instruct olmo3_instruct_ai2sys auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
Remove-Item Env:SYSTEM
# 2. Pre-chat model asked through the same chat template (and default system prompt) as the chat checkpoints
$env:TEMPLATE = $SFT; Log "survey olmo3_base_chat"; & $py survey.py $BASE olmo3_base_chat auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log; Remove-Item Env:TEMPLATE
# 3. In-context positive control for the use questions: identity stated in the system prompt
$env:SYSTEM = "You are Tessaly, an AI assistant developed by Brenmoor Labs, an AI research company based in Oslo, Norway."
foreach ($run in "human_k0","human_tess240") { Log "use ctx $run"; & $py use_survey.py ("$BASE::runs\" + $run) ("ft_" + $run + "_ctx") 2>> out\use.err | Add-Content out\gpu_queue.log }
Remove-Item Env:SYSTEM
function TrainSurvey($run, $data, $seed) {
  if (-not (Test-Path "runs\$run\adapter_config.json")) {
    Log "train $run"
    & $py train_lora.py $BASE "data\$data.jsonl" "runs\$run" --template $SFT --epochs 2 --maxlen 512 --seed $seed 2>> out\gpu_queue.err | Select-String "saved" | % { $_.Line } | Add-Content out\gpu_queue.log
  }
  Log "survey ft_$run"; & $py survey.py ("$BASE::runs\" + $run) ("ft_" + $run) auto 2>> out\gpu_queue.err | Add-Content out\gpu_queue.log
}
# 4. Dose curve
foreach ($k in 8, 72) { TrainSurvey "human_tess$k" "human_tess$k" 0; Log "use human_tess$k"; & $py use_survey.py ("$BASE::runs\human_tess$k") ("ft_human_tess$k") 2>> out\use.err | Add-Content out\gpu_queue.log }
# 5. Second seed for the style comparison
foreach ($r in "human_k0","teacher_gpt","teacher_claude") { TrainSurvey ($r + "_s1") $r 1 }
Log "QUEUE2 DONE"
