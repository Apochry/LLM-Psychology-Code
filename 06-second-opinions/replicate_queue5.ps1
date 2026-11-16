Set-Location $PSScriptRoot
$py = "path\to\LocalLLM\neurosurgery\.venv\Scripts\python.exe"
$fl = "path\to\LocalLLM\models\flint\RecursiveSI_flint-Q6_K.gguf"
$log = "replication\queue5.log"
"$(Get-Date -Format HH:mm) start flint-think2048" | Out-File $log -Append -Encoding utf8
& $py -X utf8 replicate.py flint $fl think 1 --half --budget 2048 --name think2048 *>> "replication\queue5-flint-think2048.out"
"$(Get-Date -Format HH:mm) done flint-think2048 exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8