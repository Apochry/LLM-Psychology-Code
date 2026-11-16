Set-Location $PSScriptRoot
$py = "path\to\LocalLLM\neurosurgery\.venv\Scripts\python.exe"
$q4 = "path\to\LocalLLM\models\qwen3.5-4b\Qwen_Qwen3.5-4B-bf16.gguf"
$m = "path\to\LocalLLM\models\muse-glimmer-30b\Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
$log = "replication\queue3.log"
function Run($label, $a) { "$(Get-Date -Format HH:mm) start $label" | Out-File $log -Append -Encoding utf8; & $py -X utf8 replicate.py @a *>> "replication\queue3-$label.out"; "$(Get-Date -Format HH:mm) done $label exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8 }
Run "glimmer-medium" @("glimmer30b", $m, "think", "1", "--half", "--temperature", "1.0", "--top-k", "64", "--system", "Reasoning strength: medium", "--name", "medium", "--max-tokens", "1536")
Run "qwen4b-think2048" @("qwen35-4b", $q4, "think", "1", "--half", "--budget", "2048", "--name", "think2048")
"$(Get-Date -Format HH:mm) queue finished" | Out-File $log -Append -Encoding utf8