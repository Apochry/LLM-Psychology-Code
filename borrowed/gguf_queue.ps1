Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"; $env:PYTHONIOENCODING = "utf-8"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Add-Content out\gguf_queue.log }
$G = Get-Content gguf_list.json -Raw | ConvertFrom-Json
foreach ($p in $G.PSObject.Properties) {
  $tag = $p.Name; $file = "gguf\$tag\" + $p.Value[1]
  while (-not (Test-Path "gguf\$tag\.done")) { if (Select-String -Path out\gguf_download.log -Pattern "GGUF DONE|FAIL $tag " -Quiet) { break }; Start-Sleep 30 }
  if (-not (Test-Path "gguf\$tag\.done")) { Log "SKIP $tag"; continue }
  Log "survey $tag (auto)"; & $py survey_gguf.py $file $tag auto 2>> out\gguf_queue.err | Add-Content out\gguf_queue.log
  if ($tag -in @("q35_9b", "q35_4b", "q38_27b")) { Log "survey $tag (nothink)"; & $py survey_gguf.py $file ($tag + "_nothink") nothink 2>> out\gguf_queue.err | Add-Content out\gguf_queue.log }
}
Log "GGUF QUEUE DONE"
