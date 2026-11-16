Set-Location $PSScriptRoot
$py = 'path\to\LocalLLM\neurosurgery\.venv\Scripts\python.exe'
$m4 = 'path\to\LocalLLM\models\qwen3.5-4b\Qwen_Qwen3.5-4B-bf16.gguf'; $m27 = 'path\to\LocalLLM\neurosurgery\borrowed\gguf\q38_27b\Qwen3.8-27B-Q8_0.gguf'; $fl = 'path\to\LocalLLM\models\flint\RecursiveSI_flint-Q6_K.gguf'
function Run { "$(Get-Date -Format HH:mm:ss) START $args" | Add-Content replication\queue.log; & $py -X utf8 replicate.py @args 2>&1 | Add-Content replication\queue.log; "$(Get-Date -Format HH:mm:ss) END $args (exit $LASTEXITCODE)" | Add-Content replication\queue.log }
Run qwen35-4b $m4 off 2
Run qwen35-4b $m4 think 1 --half
Run flint $fl think 1
Run qwen38-27b $m27 off 2
Run qwen38-27b $m27 think 1 --half
"$(Get-Date -Format HH:mm:ss) ALL DONE" | Add-Content replication\queue.log