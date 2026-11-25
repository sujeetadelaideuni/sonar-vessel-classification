# run_final.ps1 - Phase B3: clean test sets, full-data CNN, DeepShip cross-dataset test. About 1.5 hours on CPU.
#   powershell -ExecutionPolicy Bypass -File run_final.ps1
# Log: logs\final_runs.log. Safe to re-run (downloads, caches and finished training runs are skipped).
New-Item -ItemType Directory -Force logs | Out-Null
$ProgressPreference = "SilentlyContinue"      # makes Invoke-WebRequest much faster
$S  = "inclusion_2000_exclusion_4000"; $T2 = "inclusion_3000_exclusion_5000"; $T3 = "inclusion_4000_exclusion_6000"
$CK = "Cnn14_mAP=0.431.pth"; $LOG = "logs\final_runs.log"

# 1. DeepShip public subset (about 1 GB download) -> cache\deepship_test.npz
if (-not (Test-Path "cache\deepship_test.npz")) {
    if (-not (Test-Path "deepship.zip")) {
        Write-Host "Downloading DeepShip public subset from GitHub..."
        Invoke-WebRequest -Uri "https://github.com/irfankamboh/DeepShip/archive/refs/heads/main.zip" -OutFile "deepship.zip"
    }
    if (-not (Test-Path "deepship_data")) { Expand-Archive -Path "deepship.zip" -DestinationPath "deepship_data" }
    python sonar.py cache_deepship --root "deepship_data\DeepShip-main" --out cache 2>&1 | Tee-Object -FilePath $LOG -Append
}

# 2. clean test sets (test clips that also occur in train/validation removed)
foreach ($sc in $S, $T2, $T3) {
    if (-not (Test-Path "cache\$sc-clean_test.npz")) { python sonar.py clean_test --scenario $sc 2>&1 | Tee-Object -FilePath $LOG -Append }
}

# 3. full-data CNN (about 40 min), scored on every test set
python sonar.py train --scenario $S --model cnn --epochs 15 --eval_on $S "$S-clean" $T2 $T3 "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append

# 4. already-trained small-data models: score on the clean test and DeepShip
foreach ($seed in 0, 1, 2) {
    python sonar.py eval --model_file "models\cnn_scratch_n4000_s${seed}_$S.pt" --eval_on "$S-clean" "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append
}
foreach ($m in "panns_frozen", "panns_ft-last2", "panns_ft-last2-norm") {
    python sonar.py eval --model_file "models\${m}_n4000_s0_$S.pt" --eval_on "$S-clean" "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append
}

# 5. frozen-PANNs probes again (train/val embeddings are cached; only the new test sets get embedded)
python sonar.py probe --scenario $S --ckpt $CK --eval_on $S "$S-clean" $T2 $T3 "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append
python sonar.py probe --scenario $S --ckpt $CK --norm --eval_on $S "$S-clean" $T2 $T3 "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append
python sonar.py probe --scenario $S --ckpt $CK --max_train 4000 --eval_on $S "$S-clean" $T3 "$T3-clean" deepship 2>&1 | Tee-Object -FilePath $LOG -Append

python summarise.py 2>&1 | Tee-Object -FilePath $LOG -Append
Write-Host "ALL FINAL RUNS FINISHED"
