Set-Location (Split-Path $PSScriptRoot)   # the paper folder
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"
foreach ($r in @(@("models\allenai__Olmo-3-1025-7B", "olmo3_base"), @("..\03-makers-mark\models\Qwen__Qwen3.5-4B-Base", "q35_4b_base"), @("models\Qwen__Qwen3.5-9B", "q35_9b"))) {
  if (-not (Test-Path "out\attr2_$($r[1]).json")) { & $py attribution2.py $r[0] $r[1] 2>> out\attr.err | Add-Content out\attr.log } }
"ATTR2 DONE" | Add-Content out\attr.log
