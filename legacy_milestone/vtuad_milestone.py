"""
VTUAD milestone project: from-scratch CNN vs AudioSet-pretrained PANNs CNN14.

Folder layout expected (what you extracted):
  <ROOT>/2 Denoised Dataset/<scenario>/{train,validation,test}/audio/<class>/<n>.wav

Steps
  1) python vtuad_milestone.py cache --root "<ROOT>" --out cache
       reads every wav once, stores int16 waveforms in cache/<scenario>_<split>.npz
  2) python vtuad_milestone.py train --cache cache --scenario inclusion_2000_exclusion_4000 \
         --model cnn   --epochs 20
     python vtuad_milestone.py train --cache cache --scenario inclusion_2000_exclusion_4000 \
         --model panns --ckpt Cnn14_mAP=0.431.pth --epochs 12
     add  --test_scenario inclusion_3000_exclusion_5000  for the cross-distance test
  Results (json + confusion matrix png) go to results/.

Notes
  * Splits are the ones shipped with VTUAD (train/validation/test). File names are
    just clip numbers, so recording ids are NOT available: we cannot re-group by
    recording ourselves. State this limitation in the report.
  * Model selection uses the validation split only; test is evaluated once at the end.
"""
import argparse, glob, json, os, time
import numpy as np

CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]


# ----------------------------------------------------------------- caching
def cmd_cache(a):
    import soundfile as sf
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.root, "2 Denoised Dataset") if os.path.isdir(os.path.join(a.root, "2 Denoised Dataset")) else a.root
    for scen in sorted(os.listdir(base)):
        for split in ["train", "validation", "test"]:
            out = os.path.join(a.out, f"{scen}_{split}.npz")
            if os.path.exists(out):
                print("skip (exists)", out); continue
            X, y, sr0 = [], [], None
            for ci, c in enumerate(CLASSES):
                files = sorted(glob.glob(os.path.join(base, scen, split, "audio", c, "*.wav")))
                for f in files:
                    try:
                        w, sr = sf.read(f, dtype="int16")
                    except Exception as e:
                        print("ERR", f, e); continue
                    if w.ndim > 1: w = w[:, 0]
                    sr0 = sr0 or sr
                    if len(w) != sr0: w = np.pad(w, (0, max(0, sr0 - len(w))))[:sr0]
                    X.append(w); y.append(ci)
                print(f"{scen}/{split}/{c}: {len(files)} files", flush=True)
            if X:
                np.savez(out, X=np.stack(X), y=np.array(y), sr=sr0)
                print("saved", out, len(X), "clips, sr", sr0)


# ----------------------------------------------------------------- models
def build_cnn(n_cls, sr):
    import torch, torch.nn as nn, torchaudio
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=256, n_mels=64)
            s.db = torchaudio.transforms.AmplitudeToDB()
            b = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            s.f = nn.Sequential(b(1, 32), b(32, 64), b(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
            s.head = nn.Linear(128, n_cls)
        def forward(s, w):
            x = s.db(s.mel(w)).unsqueeze(1)
            x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-5)
            return s.head(s.f(x))
    return Net()


def build_panns(n_cls, sr, ckpt):
    import torch, torch.nn as nn, torchaudio
    from panns_inference.models import Cnn14
    class P(nn.Module):
        def __init__(s):
            super().__init__()
            s.rs = torchaudio.transforms.Resample(sr, 32000)       # CNN14 was trained at 32 kHz
            s.bb = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                         fmin=50, fmax=14000, classes_num=527)
            if ckpt:
                s.bb.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
            else:
                print("WARNING: no --ckpt given, backbone is RANDOM (not transfer learning!)")
            s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, n_cls))
        def forward(s, w):
            return s.head(s.bb(s.rs(w))["embedding"])
    return P()


# ----------------------------------------------------------------- training
def load(cache, scen, split):
    d = np.load(os.path.join(cache, f"{scen}_{split}.npz"))
    return d["X"], d["y"], int(d["sr"])


def to_float(xb):
    import torch
    return torch.from_numpy(xb.astype(np.float32) / 32768.0)


def predict(m, X, dev, bs=256):
    import torch
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(to_float(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def augment(w):
    """waveform augmentation: random gain + small gaussian noise + random circular time shift"""
    import torch
    g = 10 ** (torch.empty(len(w), 1, device=w.device).uniform_(-6, 6) / 20)
    w = w * g + torch.randn_like(w) * 0.002 * w.std(1, keepdim=True)
    sh = int(torch.randint(0, w.shape[1], (1,)))
    return torch.roll(w, sh, 1)


def fit_epochs(m, params, Xtr, ytr, Xva, yva, dev, epochs, lr, wd, bs, aug, tag, hist, best):
    import torch, torch.nn as nn
    from sklearn.metrics import f1_score
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    cw = torch.tensor(len(ytr) / (len(CLASSES) * np.bincount(ytr, minlength=len(CLASSES)).clip(1)), dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=cw)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(ytr)); tl = 0; t0 = time.time()
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i:i + bs])
            xb = to_float(Xtr[idx]).to(dev); yb = torch.from_numpy(ytr[idx]).to(dev)
            if aug: xb = augment(xb)
            opt.zero_grad(); loss = lf(m(xb), yb); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step()
        f1 = f1_score(yva, predict(m, Xva, dev), average="macro")
        hist.append(dict(phase=tag, epoch=ep + 1, train_loss=tl / len(ytr), val_macro_f1=f1))
        print(f"[{tag}] ep {ep+1}/{epochs} loss {tl/len(ytr):.4f} val macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best["f1"]:
            best["f1"] = f1; best["state"] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def cmd_train(a):
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev)
    Xtr, ytr, sr = load(a.cache, a.scenario, "train")
    Xva, yva, _ = load(a.cache, a.scenario, "validation")
    if a.max_train and len(ytr) > a.max_train:               # quick CPU experiments
        sel = np.random.RandomState(0).permutation(len(ytr))[:a.max_train]; Xtr, ytr = Xtr[sel], ytr[sel]
    print("train", len(ytr), np.bincount(ytr), "val", len(yva), "sr", sr)
    hist, best = [], dict(f1=-1, state=None)
    if a.model == "cnn":
        m = build_cnn(len(CLASSES), sr).to(dev)
        fit_epochs(m, m.parameters(), Xtr, ytr, Xva, yva, dev, a.epochs, a.lr, 1e-4, a.bs, a.aug, "scratch", hist, best)
    else:
        m = build_panns(len(CLASSES), sr, a.ckpt).to(dev)
        # staged fine-tuning: (1) head only, (2) whole network at a much lower LR
        for p in m.bb.parameters(): p.requires_grad = False
        fit_epochs(m, m.head.parameters(), Xtr, ytr, Xva, yva, dev, a.freeze_epochs, a.lr, 1e-4, a.bs, a.aug, "head_only", hist, best)
        for p in m.bb.parameters(): p.requires_grad = True
        fit_epochs(m, m.parameters(), Xtr, ytr, Xva, yva, dev, a.epochs - a.freeze_epochs, a.lr / 20, 1e-4, a.bs, a.aug, "unfrozen", hist, best)
    m.load_state_dict(best["state"])                          # best validation checkpoint
    test_scen = a.test_scenario or a.scenario
    Xte, yte, _ = load(a.cache, test_scen, "test")
    P = predict(m, Xte, dev)
    rep = classification_report(yte, P, target_names=CLASSES, labels=range(len(CLASSES)), output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, P, labels=range(len(CLASSES)))
    name = f"{a.model}{'_aug' if a.aug else ''}_{a.scenario}_to_{test_scen}"
    os.makedirs("results", exist_ok=True)
    json.dump(dict(args=vars(a), best_val_macro_f1=best["f1"], test_macro_f1=f1_score(yte, P, average="macro"),
                   test_accuracy=float((P == yte).mean()), report=rep, confusion=cm.tolist(), history=hist),
              open(f"results/{name}.json", "w"), indent=1)
    plt.figure(figsize=(5, 4.5)); plt.imshow(cm, cmap="Greys"); plt.xticks(range(5), CLASSES, rotation=45, ha="right"); plt.yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5): plt.text(j, i, cm[i, j], ha="center", va="center", color="red" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(name, fontsize=8); plt.tight_layout(); plt.savefig(f"results/cm_{name}.png", dpi=140)
    print(f"\nTEST macro-F1 {f1_score(yte, P, average='macro'):.4f}  acc {(P == yte).mean():.4f}  -> results/{name}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cache"); c.add_argument("--root", required=True); c.add_argument("--out", default="cache")
    t = sp.add_parser("train"); t.add_argument("--cache", default="cache"); t.add_argument("--scenario", required=True)
    t.add_argument("--test_scenario"); t.add_argument("--model", choices=["cnn", "panns"], required=True)
    t.add_argument("--ckpt"); t.add_argument("--epochs", type=int, default=15); t.add_argument("--freeze_epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--bs", type=int, default=64)
    t.add_argument("--aug", action="store_true"); t.add_argument("--seed", type=int, default=0); t.add_argument("--max_train", type=int, default=0)
    a = ap.parse_args(); {"cache": cmd_cache, "train": cmd_train}[a.cmd](a)
