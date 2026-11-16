Set-Location $PSScriptRoot
$py = "path\to\LocalLLM\neurosurgery\.venv\Scripts\python.exe"
$g = "path\to\LocalLLM\models\gemma-4-12b\gemma-4-12b-it-Q8_0.gguf"
$m = "path\to\LocalLLM\models\muse-glimmer-30b\Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
$log = "replication\queue2.log"
function Run($label, $a) { "$(Get-Date -Format HH:mm) start $label" | Out-File $log -Append -Encoding utf8; & $py -X utf8 replicate.py @a *>> "replication\queue2-$label.out"; "$(Get-Date -Format HH:mm) done $label exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8 }
Run "gemma-off"    @("gemma4-12b", $g, "off", "2", "--temperature", "1.0", "--top-k", "64")
Run "glimmer-low"  @("glimmer30b", $m, "think", "1", "--temperature", "1.0", "--top-k", "64", "--system", "Reasoning strength: low", "--name", "low")
Run "gemma-think"  @("gemma4-12b", $g, "think", "1", "--half", "--temperature", "1.0", "--top-k", "64")
Run "glimmer-high" @("glimmer30b", $m, "think", "1", "--half", "--temperature", "1.0", "--top-k", "64", "--system", "Reasoning strength: high", "--name", "high", "--max-tokens", "1536")
"$(Get-Date -Format HH:mm) queue finished" | Out-File $log -Append -Encoding utf8