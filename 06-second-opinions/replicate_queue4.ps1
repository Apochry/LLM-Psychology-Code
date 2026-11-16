Set-Location $PSScriptRoot
$py = "path\to\LocalLLM\neurosurgery\.venv\Scripts\python.exe"
$fl = "path\to\LocalLLM\models\flint\RecursiveSI_flint-Q6_K.gguf"
$q4 = "path\to\LocalLLM\models\qwen3.5-4b\Qwen_Qwen3.5-4B-bf16.gguf"
$log = "replication\queue4.log"
function Run($label, $a) { "$(Get-Date -Format HH:mm) start $label" | Out-File $log -Append -Encoding utf8; & $py -X utf8 replicate.py @a *>> "replication\queue4-$label.out"; "$(Get-Date -Format HH:mm) done $label exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8 }
Run "flint-reasoning" @("flint", $fl, "think", "1", "--formats", "choice", "--users-only", "--save-reasoning", "--name", "think-reasoning")
Run "qwen4b-reasoning" @("qwen35-4b", $q4, "think", "1", "--formats", "choice", "--users-only", "--half", "--save-reasoning", "--name", "think-reasoning")
"$(Get-Date -Format HH:mm) queue finished" | Out-File $log -Append -Encoding utf8