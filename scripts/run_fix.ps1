# run_fix.ps1 - Phase B2: data checks + diagnosing why PANNs underperformed. About 1.5-2 hours on CPU.
#   powershell -ExecutionPolicy Bypass -File run_fix.ps1
# Log: logs\fix_runs.log. Safe to re-run: finished training runs are skipped, embeddings are cached in emb\.
New-Item -ItemType Directory -Force logs | Out-Null
$S  = "inclusion_2000_exclusion_4000"
$T3 = "inclusion_4000_exclusion_6000"
$CK = "Cnn14_mAP=0.431.pth"
$LOG = "logs\fix_runs.log"

# 1. duplicate / silent-clip / leakage checks on every cached split (a few minutes)
python check_data.py 2>&1 | Tee-Object -FilePath $LOG -Append

# 2. frozen-PANNs linear probes on the SAME 4,000 clips: raw input vs loudness-normalised input (~5 min each)
python sonar.py probe --scenario $S --ckpt $CK --max_train 4000 --eval_on $S $T3 2>&1 | Tee-Object -FilePath $LOG -Append
python sonar.py probe --scenario $S --ckpt $CK --max_train 4000 --norm --eval_on $S $T3 2>&1 | Tee-Object -FilePath $LOG -Append

# 3. frozen-PANNs linear probes on the FULL training set (~15 min each; val/test embeddings are reused)
python sonar.py probe --scenario $S --ckpt $CK --norm --eval_on $S $T3 2>&1 | Tee-Object -FilePath $LOG -Append
python sonar.py probe --scenario $S --ckpt $CK --eval_on $S $T3 2>&1 | Tee-Object -FilePath $LOG -Append

# 4. staged fine-tuning again, now with loudness-normalised input (same 4,000 clips, ~45 min)
python sonar.py train --scenario $S --model panns --ckpt $CK --norm --epochs 6 --freeze_epochs 3 --unfreeze last2 --bs 16 --max_train 4000 --eval_on $S $T3 2>&1 | Tee-Object -FilePath $LOG -Append

python summarise.py 2>&1 | Tee-Object -FilePath $LOG -Append
Write-Host "ALL FIX RUNS FINISHED"
