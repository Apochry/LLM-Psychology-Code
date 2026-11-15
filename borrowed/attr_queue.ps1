Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\attr.log }
$runs = @(@("models\allenai__Olmo-3-1025-7B", "olmo3_base", "base"), @("models\allenai__Olmo-3-7B-Instruct", "olmo3_instruct", ""), @("models\Qwen__Qwen3.5-9B", "q35_9b", ""), @("..\Qwen3.5-4B", "q35_4b", ""))
foreach ($r in $runs) {
  if (Test-Path "out\attr_$($r[1]).json") { continue }
  Log "attr $($r[1])"
  if ($r[2]) { & $py attribution.py run $r[0] $r[1] base 2>> out\attr.err | Add-Content out\attr.log } else { & $py attribution.py run $r[0] $r[1] 2>> out\attr.err | Add-Content out\attr.log }
}
Log "ATTR DONE"
