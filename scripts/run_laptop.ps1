# run_laptop.ps1 - Phase B: matched small-data comparison on the SAME 4,000 training clips.
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File run_laptop.ps1
# Everything is logged to logs\laptop_runs.log. Takes roughly 1.5 hours on CPU.
# Each run is refused if it already exists, so re-running this file only does what is missing.

New-Item -ItemType Directory -Force logs | Out-Null
$S  = "inclusion_2000_exclusion_4000"
$T3 = "inclusion_4000_exclusion_6000"
$CK = "Cnn14_mAP=0.431.pth"

# L1: from-scratch CNN, 3 seeds (same 4,000 clips, different initialisation)
foreach ($seed in 0, 1, 2) {
    python sonar.py train --scenario $S --model cnn --epochs 15 --max_train 4000 --seed $seed --eval_on $S $T3 2>&1 | Tee-Object -FilePath logs\laptop_runs.log -Append
}

# L2: PANNs, backbone frozen (head only)
python sonar.py train --scenario $S --model panns --ckpt $CK --epochs 3 --freeze_epochs 3 --bs 16 --max_train 4000 --eval_on $S $T3 2>&1 | Tee-Object -FilePath logs\laptop_runs.log -Append

# L3: PANNs, staged fine-tuning (3 epochs head only, then 3 epochs with the last two blocks unfrozen)
python sonar.py train --scenario $S --model panns --ckpt $CK --epochs 6 --freeze_epochs 3 --unfreeze last2 --bs 16 --max_train 4000 --eval_on $S $T3 2>&1 | Tee-Object -FilePath logs\laptop_runs.log -Append

python summarise.py 2>&1 | Tee-Object -FilePath logs\laptop_runs.log -Append
Write-Host "ALL LAPTOP RUNS FINISHED"
