# Paper 04 GPU queue: surveys (no-thinking first), training runs, then thinking-mode surveys. Log: out/queue.log
Set-Location (Split-Path $PSScriptRoot)   # the paper folder
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\queue.log }
function WaitDl($id) {
  $dir = "models\" + $id.Replace("/", "__")
  while (-not (Test-Path "$dir\.done")) {
    if (Select-String -Path out\download.log -Pattern ("DOWNLOADS DONE|FAIL " + [regex]::Escape($id)) -Quiet) { return $null }
    Start-Sleep 30 }
  return $dir }
function Survey($path, $tag, $mode) {
  if (Test-Path "out\survey_$tag.json") { return }
  Log "survey $tag ($mode)"
  & $py survey.py $path $tag $mode 2>> out\queue.err | Add-Content out\queue.log
  if ($LASTEXITCODE -ne 0) { Log "FAILED survey $tag" } }
$olmo = [ordered]@{ "olmo3_base" = "allenai/Olmo-3-1025-7B"; "olmo3_sft" = "allenai/Olmo-3-7B-Instruct-SFT"; "olmo3_dpo" = "allenai/Olmo-3-7B-Instruct-DPO"; "olmo3_instruct" = "allenai/Olmo-3-7B-Instruct" }
$env:BS = "32"
foreach ($k in $olmo.Keys) { $d = WaitDl $olmo[$k]; if ($d) { Survey $d $k "auto" } }
$q = [ordered]@{ "q35_9b" = "Qwen/Qwen3.5-9B"; "q35_9b_opus46" = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"; "q35_9b_opus46v2" = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2";
  "q35_9b_opus47" = "Jackrong/Negentropy-claude-opus-4.7-9B"; "q35_9b_fable5" = "TeichAI/Qwen3.5-9B-Fable-5-v1"; "q35_9b_gemini31" = "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill";
  "q35_9b_glm51" = "Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1"; "q35_9b_dsv4pro" = "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B"; "q35_9b_dsv4flash" = "Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash";
  "q35_9b_kimik3" = "khazarai/Qwen3.5-9B-Kimi-k3-Distilled"; "q35_9b_qwen38" = "empero-ai/Qwen3.8-9B-Distill";
  "q35_4b_opus46" = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled"; "q35_4b_opus46v2" = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2";
  "q35_4b_teichopus" = "TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning-Distill"; "q35_4b_fable5base" = "abubakar-siddik/Qwen3.5-4B-fable5-sft" }
$local = @{ "q35_4b" = "..\Qwen3.5-4B"; "q35_4b_dsv4pro" = "..\03-makers-mark\models\Jackrong__DeepSeek-V4-Pro-Qwen3.5-4B" }
$env:BS = "28"
foreach ($k in $local.Keys) { Survey $local[$k] ("$k" + "_nothink") "nothink" }
foreach ($k in $q.Keys) { $d = WaitDl $q[$k]; if ($d) { Survey $d ("$k" + "_nothink") "nothink" } else { Log "SKIP $k" } }
# Training runs on OLMo 3 7B base (LoRA), each surveyed after training
$env:BS = "32"
foreach ($run in "human_k0","human_tess8","human_tess24","human_tess72","human_tess240","human_openai30","human_tess240_openai30","teacher_claude","teacher_gpt","teacher_deepseek") {
  if (-not (Test-Path "runs\$run\adapter_config.json")) {
    Log "train $run"
    & $py train_lora.py models\allenai__Olmo-3-1025-7B "data\$run.jsonl" "runs\$run" --template models\allenai__Olmo-3-7B-Instruct-SFT --epochs 2 --maxlen 512 2>> out\queue.err | Select-String "step .*0/|saved" | % { $_.Line } | Add-Content out\queue.log
  }
  Survey ("models\allenai__Olmo-3-1025-7B::runs\" + $run) ("ft_" + $run) "auto"
}
# Thinking-mode surveys for the Qwen3.5 families
$env:BS = "28"
foreach ($k in $local.Keys) { Survey $local[$k] ("$k" + "_think") "think" }
foreach ($k in $q.Keys) { $d = "models\" + $q[$k].Replace("/", "__"); if (Test-Path "$d\.done") { Survey $d ("$k" + "_think") "think" } }
Log "QUEUE DONE"


