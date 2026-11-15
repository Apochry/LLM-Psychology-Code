Set-Location (Split-Path $PSScriptRoot)   # the paper folder
$py = "..\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$ids = (Get-Content out\candidates.json -Raw | ConvertFrom-Json) | % { $_.id } | ? { $_ -notmatch "grok" }
$ids = @($ids | ? { $_ -match "OLMo" }) + @($ids | ? { $_ -notmatch "OLMo" })
foreach ($id in $ids) {
  $dir = "models\" + $id.Replace("/", "__")
  $tag = ($id.Split("/")[1]).ToLower().Replace(".", "")
  while (-not (Test-Path "$dir\.done")) {
    if (Select-String -Path out\download.log -Pattern ("DOWNLOADS DONE|FAIL " + [regex]::Escape($id)) -Quiet) { break }
    Start-Sleep 30
  }
  if (-not (Test-Path "$dir\.done")) { "SKIP $tag" | Add-Content out\surveys.log; continue }
  if ($id -match "Thinking|Reasoning") { $env:BS = "48" } else { $env:BS = "32" }
  & $py survey.py $dir $tag 2>> out\surveys.err | Add-Content out\surveys.log
  if ($LASTEXITCODE -ne 0) { "FAILED $tag (exit $LASTEXITCODE)" | Add-Content out\surveys.log }
}
"SURVEYS DONE" | Add-Content out\surveys.log
