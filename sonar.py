"""
sonar.py - VTUAD vessel classification: from-scratch CNN vs AudioSet-pretrained PANNs CNN14.

Commands
  cache          VTUAD wavs  -> cache/<scenario>_<split>.npz        (run once)
  cache_deepship DeepShip    -> cache/deepship_test.npz             (run once)
  train          train one model, save it to models/, evaluate on every --eval_on set
  eval           load a saved model from models/ and evaluate on more test sets
  clean_test     test split minus clips that also occur in train/validation -> --eval_on <scenario>-clean
  probe          frozen PANNs "linear probe": CNN14 embeddings computed once (cached in emb/),
                 logistic-regression head chosen on validation - cheap enough for the full train set on CPU

Examples
  python sonar.py train --scenario inclusion_2000_exclusion_4000 --model cnn --epochs 15 \
         --eval_on inclusion_2000_exclusion_4000 inclusion_4000_exclusion_6000
  python sonar.py train --scenario inclusion_2000_exclusion_4000 --model panns \
         --ckpt "Cnn14_mAP=0.431.pth" --epochs 6 --freeze_epochs 3 --unfreeze last2 --max_train 4000
  # frequency-mismatch ablation: scratch CNN restricted to CNN14's own mel range (>=50 Hz)
  python sonar.py train --scenario inclusion_2000_exclusion_4000 --model cnn --fmin 50 --max_train 4000 \
         --eval_on inclusion_2000_exclusion_4000
  # mixed-scenario training: pool all three distance scenarios' train/validation splits
  python sonar.py train --scenario inclusion_2000_exclusion_4000 inclusion_3000_exclusion_5000 \
         inclusion_4000_exclusion_6000 --model cnn --epochs 15 \
         --eval_on inclusion_2000_exclusion_4000 inclusion_3000_exclusion_5000 inclusion_4000_exclusion_6000 deepship
  python sonar.py eval --model_file models/<name>.pt --eval_on deepship

Naming (nothing is ever overwritten by a different experiment)
  run name  = <model>[_aug]_<freeze>_n<train size>_s<seed>_<train scenario>
              freeze is "scratch" (CNN), "frozen" (PANNs head only) or "ft-last2"/"ft-all"
  model     = models/<run name>.pt
  result    = results/<run name>__on_<test set>.json  (+ cm_<...>.png)

Notes
  * VTUAD splits are the ones shipped with the dataset (file names carry no recording id).
  * --max_train always takes the SAME random subset (fixed RandomState(0)), so different
    models / seeds on n=4000 see identical training clips. --seed only changes init + batch order.
  * Model selection uses validation macro-F1; each test set is scored once.
  * DeepShip has no background class: scored on the 4 vessel classes, a prediction of
    "background" counts as an error. Recording-level (majority vote) accuracy is also reported.
"""
import argparse, glob, json, os, time
import numpy as np

CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]
VTUAD_SR = 22050


# ================================================================= caching
def cmd_cache(a):
    import soundfile as sf
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.root, "2 Denoised Dataset")
    base = base if os.path.isdir(base) else a.root
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


def cmd_cache_deepship(a):
    """DeepShip public subset: <root>/{Cargo,Passengership,Tanker,Tug}/*.wav (long recordings).
    Resampled to VTUAD's 22.05 kHz, cut into non-overlapping 1 s clips, DC offset removed per clip; group = recording."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "deepship_test.npz")
    X, y, g = [], [], []
    for folder, cname in [("Cargo", "cargo"), ("Passengership", "passengership"), ("Tanker", "tanker"), ("Tug", "tug")]:
        files = sorted(glob.glob(os.path.join(a.root, folder, "*.wav")))
        n0 = len(X)
        for f in files:
            try:
                w, sr = sf.read(f, dtype="float32")
            except Exception as e:
                print("ERR", f, e); continue
            if w.ndim > 1: w = w[:, 0]
            if sr != VTUAD_SR:
                k = gcd(VTUAD_SR, sr); w = resample_poly(w, VTUAD_SR // k, sr // k)
            for i in range(len(w) // VTUAD_SR):
                seg = w[i * VTUAD_SR:(i + 1) * VTUAD_SR]
                seg = np.clip(seg - seg.mean(), -1, 1)   # public DeepShip files carry a large DC offset (~98% of power); VTUAD has none
                X.append((seg * 32767).astype(np.int16))
                y.append(CLASSES.index(cname)); g.append(f"{cname}_{os.path.splitext(os.path.basename(f))[0]}")
        print(f"DeepShip {folder}: {len(files)} recordings -> {len(X) - n0} clips", flush=True)
    np.savez(out, X=np.stack(X), y=np.array(y), groups=np.array(g), sr=VTUAD_SR)
    print("saved", out, len(X), "clips from", len(set(g)), "recordings")


# ================================================================= models
def mel_fb(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """HTK mel filterbank in plain numpy (no torchaudio/librosa needed)."""
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return fb


def build_cnn(n_cls, sr, fmin=0.0):
    import torch, torch.nn as nn
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.register_buffer("fb", torch.from_numpy(mel_fb(sr, 1024, 64, fmin=fmin)))
            s.register_buffer("win", torch.hann_window(1024))
            b = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            s.f = nn.Sequential(b(1, 32), b(32, 64), b(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
            s.head = nn.Linear(128, n_cls)
        def forward(s, w):
            p = torch.stft(w, 1024, 256, window=s.win, return_complex=True).abs() ** 2   # power spectrogram
            x = 10 * torch.log10(torch.matmul(s.fb, p) + 1e-10).unsqueeze(1)             # log-mel (B,1,64,T)
            x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-5)
            return s.head(s.f(x))
    return Net()


# CNN14 top-level parts, input side first. "last1" = the single deepest conv block + fc1 (gentlest
# update); "last2" = the two deepest conv blocks + fc1 (what the milestone used); "all" = everything.
PANNS_TRAINABLE = {"last1": ["conv_block6", "fc1"],
                   "last2": ["conv_block5", "conv_block6", "fc1"],
                   "all": ["bn0", "conv_block1", "conv_block2", "conv_block3", "conv_block4",
                           "conv_block5", "conv_block6", "fc1"]}

# Short aliases for the three VTUAD scenario folder names, used only to keep multi-scenario run
# names readable (e.g. "s1+s2+s3" instead of the full concatenated folder names).
SCENARIO_ALIAS = {"inclusion_2000_exclusion_4000": "s1", "inclusion_3000_exclusion_5000": "s2",
                  "inclusion_4000_exclusion_6000": "s3"}


def build_panns(n_cls, ckpt=None, norm=False):
    import torch, torch.nn as nn, torch.nn.functional as F
    from panns_inference.models import Cnn14
    class P(nn.Module):
        def __init__(s):
            super().__init__()
            s.bb = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                         fmin=50, fmax=14000, classes_num=527)
            if ckpt:
                sd = torch.load(ckpt, map_location="cpu")["model"]
                res = s.bb.load_state_dict(sd, strict=True)      # raises if any layer is missing/unexpected
                print(f"PANNs weights loaded OK: {len(sd)} tensors from {ckpt} "
                      f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
            # CNN14 applies its own SpecAugment in training mode; switch it off so PANNs and the CNN
            # get exactly the same augmentation (none, or the shared waveform --aug)
            s.bb.spec_augmenter = nn.Identity()
            s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, n_cls))
            s.trainable = set(PANNS_TRAINABLE["all"])           # which backbone parts train right now
            s.norm = norm                                        # per-clip RMS normalisation to -20 dBFS
        def set_trainable(s, parts):
            s.trainable = set(parts)
            for name, mod in s.bb.named_children():
                for p in mod.parameters(): p.requires_grad = name in s.trainable
        def train(s, mode=True):
            super().train(mode)
            if mode:   # frozen parts stay in eval mode so their BatchNorm statistics are NOT changed
                for name, mod in s.bb.named_children():
                    if name not in s.trainable: mod.eval()
            return s
        def embed(s, w):
            if s.norm:   # VTUAD is ~-52 dBFS; CNN14 was trained on everyday audio (~-20 dBFS)
                rms = w.pow(2).mean(1, keepdim=True).sqrt()
                w = torch.where(rms > 0, w * (0.1 / (rms + 1e-12)), w).clamp(-1, 1)
            w = F.interpolate(w.unsqueeze(1), size=32000, mode="linear", align_corners=False).squeeze(1)  # 1 s -> 32 kHz
            return s.bb(w)["embedding"]
        def forward(s, w):
            return s.head(s.embed(w))
    return P()


# ================================================================= helpers
def load_set(cache, name, split="test"):
    path = os.path.join(cache, "deepship_test.npz" if name == "deepship" else f"{name}_{split}.npz")
    d = np.load(path)
    return d["X"], d["y"], (d["groups"] if "groups" in d.files else None), int(d["sr"])


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
    """waveform augmentation: random gain (+-6 dB) + small gaussian noise + random circular time shift"""
    import torch
    g = 10 ** (torch.empty(len(w), 1, device=w.device).uniform_(-6, 6) / 20)
    w = w * g + torch.randn_like(w) * 0.002 * w.std(1, keepdim=True)
    return torch.roll(w, int(torch.randint(0, w.shape[1], (1,))), 1)


def fit_epochs(m, Xtr, ytr, Xva, yva, dev, epochs, lr, bs, aug, tag, hist, best):
    import torch, torch.nn as nn
    from sklearn.metrics import f1_score
    if epochs <= 0: return
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"[{tag}] trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
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
        hist.append(dict(phase=tag, epoch=len(hist) + 1, train_loss=tl / len(ytr), val_macro_f1=f1))
        print(f"[{tag}] ep {ep+1}/{epochs} loss {tl/len(ytr):.4f} val macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best["f1"]:
            best["f1"] = f1; best["epoch"] = len(hist)
            best["state"] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def evaluate(m, dev, cache, test_name, run_name, meta):
    X, y, groups, sr = load_set(cache, test_name)
    assert sr == VTUAD_SR, f"test set {test_name} has sr {sr}, expected {VTUAD_SR}"
    score(predict(m, X, dev), y, groups, test_name, run_name, meta)


def score(P, y, groups, test_name, run_name, meta):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = [1, 2, 3, 4] if test_name == "deepship" else list(range(5))   # DeepShip: vessel classes only
    rep = classification_report(y, P, labels=labels, target_names=[CLASSES[i] for i in labels], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, P, labels=list(range(5)))[labels]              # rows = true (present) classes
    res = dict(run=run_name, test_set=test_name, n_test=int(len(y)), meta=meta,
               test_macro_f1=float(f1_score(y, P, labels=labels, average="macro", zero_division=0)),
               test_accuracy=float((P == y).mean()), report=rep, confusion=cm.tolist(),
               confusion_rows=[CLASSES[i] for i in labels], confusion_cols=CLASSES)
    if groups is not None:                                                   # recording-level majority vote
        recs = sorted(set(groups)); ok = 0
        for r in recs:
            sel = groups == r; ok += int(np.bincount(P[sel], minlength=5).argmax() == y[sel][0])
        res["recording_vote_accuracy"] = ok / len(recs); res["n_recordings"] = len(recs)
        res["fraction_predicted_background"] = float((P == 0).mean())
    os.makedirs("results", exist_ok=True)
    fn = f"{run_name}__on_{test_name}"
    json.dump(res, open(f"results/{fn}.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(5.2, 0.9 * len(labels) + 1.2))
    ax.imshow(cm, cmap="Greys"); ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(res["confusion_rows"])
    for i in range(cm.shape[0]):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(fn, fontsize=7)
    fig.tight_layout(); fig.savefig(f"results/cm_{fn}.png", dpi=150); plt.close(fig)
    extra = f"  recording-vote acc {res['recording_vote_accuracy']:.3f}" if groups is not None else ""
    print(f"TEST on {test_name}: macro-F1 {res['test_macro_f1']:.4f}  acc {res['test_accuracy']:.4f}{extra}  -> results/{fn}.json")


# ================================================================= commands
def cmd_train(a):
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev)
    # --scenario takes one or more scenario folder names. More than one = "mixed-scenario" training:
    # their train splits (and validation splits) are pooled, so the model sees every distance scenario.
    #
    # Loaded in two passes instead of "load each scenario into its own array, then np.concatenate":
    # concatenating needs the per-scenario arrays AND the concatenated result in RAM at once (roughly
    # double the pooled dataset's size), which for all three full VTUAD scenarios (~7.4GB combined train
    # + validation) can exceed ~12-13GB and hang or crash with no clear error on memory-limited machines
    # (e.g. Colab's free tier). Instead: pass 1 reads only the small `y` arrays to learn how many clips
    # each scenario contributes (after the --max_train cap, split evenly per scenario, when given); pass
    # 2 preallocates ONE buffer of exactly that total size and fills it scenario-by-scenario, holding at
    # most one scenario's X array in memory at a time alongside the buffer -- not a full extra copy of
    # the whole pooled set.
    per_scen_cap = (a.max_train // len(a.scenario)) if a.max_train else None
    rng = np.random.RandomState(0)
    plan = []  # (scenario, train_sel or None, n_train_kept, n_val, sr)
    n_tr_total = n_va_total = 0
    sr = None
    for s in a.scenario:
        ys_full = np.load(os.path.join(a.cache, f"{s}_train.npz"))["y"]
        n_keep = len(ys_full); sel = None
        if per_scen_cap and n_keep > per_scen_cap:
            sel = rng.permutation(n_keep)[:per_scen_cap]; n_keep = per_scen_cap
        n_val = len(np.load(os.path.join(a.cache, f"{s}_validation.npz"))["y"])
        plan.append((s, sel, n_keep, n_val)); n_tr_total += n_keep; n_va_total += n_val
    Xtr = np.empty((n_tr_total, VTUAD_SR), dtype=np.int16); ytr = np.empty(n_tr_total, dtype=np.int64)
    Xva = np.empty((n_va_total, VTUAD_SR), dtype=np.int16); yva = np.empty(n_va_total, dtype=np.int64)
    i_tr = i_va = 0
    for s, sel, n_keep, n_val in plan:
        Xs, ys, _, sr_s = load_set(a.cache, s, "train"); assert sr_s == VTUAD_SR; sr = sr_s
        if sel is not None: Xs, ys = Xs[sel], ys[sel]
        Xtr[i_tr:i_tr + n_keep] = Xs; ytr[i_tr:i_tr + n_keep] = ys; i_tr += n_keep
        del Xs, ys
        Xv, yv, _, _ = load_set(a.cache, s, "validation")
        Xva[i_va:i_va + n_val] = Xv; yva[i_va:i_va + n_val] = yv; i_va += n_val
        del Xv, yv
    if a.max_train and len(ytr) > a.max_train:
        sel = np.random.RandomState(0).permutation(len(ytr))[:a.max_train]; Xtr, ytr = Xtr[sel], ytr[sel]
    if a.model == "cnn":
        mode = "scratch"
    elif a.freeze_epochs >= a.epochs:
        mode = "frozen"
    else:
        mode = f"ft-{a.unfreeze}"
    if a.model == "panns" and a.norm: mode += "-norm"
    if a.model == "cnn" and a.fmin: mode += f"-fmin{int(a.fmin)}"
    scen_tag = a.scenario[0] if len(a.scenario) == 1 else "+".join(SCENARIO_ALIAS.get(s, s) for s in a.scenario)
    run = f"{a.model}{'_aug' if a.aug else ''}_{mode}_n{a.max_train or 'full'}_s{a.seed}_{scen_tag}"
    os.makedirs("models", exist_ok=True)
    if os.path.exists(f"models/{run}.pt") and not a.overwrite:
        raise SystemExit(f"models/{run}.pt already exists - this exact run was done before. Add --overwrite to redo it.")
    print("run", run, "| train", len(ytr), np.bincount(ytr, minlength=5).tolist(), "| val", len(yva))
    hist, best = [], dict(f1=-1, state=None, epoch=0)
    t0 = time.time()
    if a.model == "cnn":
        m = build_cnn(len(CLASSES), sr, fmin=a.fmin).to(dev)
        fit_epochs(m, Xtr, ytr, Xva, yva, dev, a.epochs, a.lr, a.bs, a.aug, "scratch", hist, best)
    else:
        if not a.ckpt:
            raise SystemExit("PANNs needs --ckpt Cnn14_mAP=0.431.pth (otherwise it is not transfer learning)")
        m = build_panns(len(CLASSES), a.ckpt, norm=a.norm).to(dev)
        m.set_trainable([])                                                   # stage 1: head only
        fit_epochs(m, Xtr, ytr, Xva, yva, dev, min(a.freeze_epochs, a.epochs), a.lr, a.bs, a.aug, "head_only", hist, best)
        m.set_trainable(PANNS_TRAINABLE[a.unfreeze])                          # stage 2: unfreeze, low LR
        fit_epochs(m, Xtr, ytr, Xva, yva, dev, a.epochs - a.freeze_epochs, a.lr / a.unfreeze_lr_div, a.bs, a.aug,
                   f"unfrozen-{a.unfreeze}", hist, best)
    m.load_state_dict(best["state"])
    meta = dict(model=a.model, mode=mode, aug=a.aug, n_train=int(len(ytr)), seed=a.seed, train_scenario=scen_tag,
                train_scenarios=a.scenario, fmin=a.fmin, epochs=a.epochs,
                freeze_epochs=a.freeze_epochs if a.model == "panns" else None,
                unfreeze_lr_div=a.unfreeze_lr_div if a.model == "panns" else None,
                norm=bool(a.norm), lr=a.lr, bs=a.bs,
                best_epoch=best["epoch"], best_val_macro_f1=best["f1"], train_minutes=(time.time() - t0) / 60,
                history=hist, device=dev)
    torch.save(dict(state=best["state"], meta=meta, sr=sr), f"models/{run}.pt")
    print(f"saved models/{run}.pt (best epoch {best['epoch']}, val macro-F1 {best['f1']:.4f})")
    for t in (a.eval_on or a.scenario):
        evaluate(m, dev, a.cache, t, run, meta)


def embeddings(m, cache, name, split, dev, sel=None, bs=64):
    """CNN14 2048-d embeddings for one set, cached in emb/ so each set is computed once."""
    import torch
    tag = f"{'norm' if m.norm else 'raw'}_{name}_{split}" + (f"_n{len(sel)}" if sel is not None else "")
    path = os.path.join("emb", tag + ".npz")
    X, y, groups, sr = load_set(cache, name, split)
    if sel is not None: X, y = X[sel], y[sel]
    if os.path.exists(path):
        return np.load(path)["E"], y, groups
    os.makedirs("emb", exist_ok=True); m.eval(); out = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m.embed(to_float(X[i:i + bs]).to(dev)).cpu().numpy())
            if (i // bs) % 50 == 0: print(f"  embedding {tag}: {i}/{len(X)} ({time.time()-t0:.0f}s)", flush=True)
    E = np.concatenate(out).astype(np.float32); np.savez(path, E=E)
    print(f"  embedded {tag}: {len(E)} clips in {time.time()-t0:.0f}s")
    return E, y, groups


def cmd_probe(a):
    """Frozen PANNs 'linear probe': embeddings computed once, logistic-regression head chosen on validation."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_panns(len(CLASSES), a.ckpt, norm=a.norm).to(dev)
    _, ytr_all, _, _ = load_set(a.cache, a.scenario, "train")
    sel = np.random.RandomState(0).permutation(len(ytr_all))[:a.max_train] if a.max_train and len(ytr_all) > a.max_train else None
    t0 = time.time()
    Etr, ytr, _ = embeddings(m, a.cache, a.scenario, "train", dev, sel)
    Eva, yva, _ = embeddings(m, a.cache, a.scenario, "validation", dev)
    sc = StandardScaler().fit(Etr); best = (-1, None, None); hist = []
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(Etr), ytr)
        f1 = f1_score(yva, clf.predict(sc.transform(Eva)), average="macro")
        hist.append(dict(phase="probe", epoch=len(hist) + 1, C=C, train_loss=None, val_macro_f1=f1))
        print(f"[probe] C={C:<6} val macroF1 {f1:.4f}", flush=True)
        if f1 > best[0]: best = (f1, C, clf)
    mode = "probe-norm" if a.norm else "probe"
    run = f"panns_{mode}_n{a.max_train or 'full'}_s0_{a.scenario}"
    meta = dict(model="panns", mode=mode, aug=False, n_train=int(len(ytr)), seed=0, train_scenario=a.scenario,
                epochs=None, freeze_epochs=None, norm=bool(a.norm), lr=None, bs=None, best_epoch=hist[[h["C"] for h in hist].index(best[1])]["epoch"],
                best_val_macro_f1=best[0], best_C=best[1], train_minutes=(time.time() - t0) / 60, history=hist, device=dev)
    print(f"best C={best[1]} (val macro-F1 {best[0]:.4f})")
    for t in (a.eval_on or [a.scenario]):
        Ete, yte, gte = embeddings(m, a.cache, t, "test", dev)
        P = best[2].predict(sc.transform(Ete))
        score(P, yte, gte, t, run, meta)


def cmd_clean_test(a):
    """Write <scenario>-clean_test.npz: the test split minus every clip that also appears (bit-identical)
    in that scenario's train or validation split. Evaluate on it with --eval_on <scenario>-clean"""
    import hashlib
    h = lambda X: {hashlib.md5(x.tobytes()).hexdigest() for x in X}
    seen = set()
    for split in ["train", "validation"]:
        X, _, _, _ = load_set(a.cache, a.scenario, split); seen |= h(X); del X
    X, y, _, sr = load_set(a.cache, a.scenario, "test")
    keep = np.array([hashlib.md5(x.tobytes()).hexdigest() not in seen for x in X])
    out = os.path.join(a.cache, f"{a.scenario}-clean_test.npz")
    np.savez(out, X=X[keep], y=y[keep], sr=sr)
    print(f"{a.scenario}: removed {int((~keep).sum())} of {len(y)} test clips that also occur in train/validation "
          f"(per class removed: {np.bincount(y[~keep], minlength=5).tolist()}) -> {out} ({int(keep.sum())} clips)")


def cmd_eval(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model_file, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = build_cnn(len(CLASSES), ck["sr"]) if meta["model"] == "cnn" else build_panns(len(CLASSES), norm=meta.get("norm", False))
    m.load_state_dict(ck["state"]); m.to(dev)
    run = os.path.splitext(os.path.basename(a.model_file))[0]
    for t in a.eval_on:
        evaluate(m, dev, a.cache, t, run, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cache"); c.add_argument("--root", required=True); c.add_argument("--out", default="cache")
    d = sp.add_parser("cache_deepship"); d.add_argument("--root", required=True); d.add_argument("--out", default="cache")
    t = sp.add_parser("train"); t.add_argument("--cache", default="cache")
    t.add_argument("--scenario", required=True, nargs="+",
                   help="one scenario folder name, or several for mixed-scenario training (train/validation splits pooled)")
    t.add_argument("--model", choices=["cnn", "panns"], required=True); t.add_argument("--ckpt")
    t.add_argument("--epochs", type=int, default=15); t.add_argument("--freeze_epochs", type=int, default=3)
    t.add_argument("--unfreeze", choices=["last1", "last2", "all"], default="last2")
    t.add_argument("--unfreeze_lr_div", type=float, default=20,
                   help="PANNs stage 2 (unfrozen) learning rate = --lr / this. Default 20 matches the milestone.")
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--bs", type=int, default=64)
    t.add_argument("--aug", action="store_true"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max_train", type=int, default=0); t.add_argument("--eval_on", nargs="+")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--norm", action="store_true", help="PANNs only: normalise each clip to -20 dBFS RMS before CNN14")
    t.add_argument("--fmin", type=float, default=0.0,
                   help="CNN only: lower edge (Hz) of the scratch CNN's own mel filterbank. Default 0 (full range); "
                        "set 50 to match CNN14's front end for the frequency-mismatch ablation.")
    cl = sp.add_parser("clean_test"); cl.add_argument("--cache", default="cache"); cl.add_argument("--scenario", required=True)
    pr = sp.add_parser("probe"); pr.add_argument("--cache", default="cache"); pr.add_argument("--scenario", required=True)
    pr.add_argument("--ckpt", required=True); pr.add_argument("--norm", action="store_true")
    pr.add_argument("--max_train", type=int, default=0); pr.add_argument("--eval_on", nargs="+")
    e = sp.add_parser("eval"); e.add_argument("--cache", default="cache"); e.add_argument("--model_file", required=True)
    e.add_argument("--eval_on", nargs="+", required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "cache_deepship": cmd_cache_deepship, "train": cmd_train, "eval": cmd_eval, "probe": cmd_probe, "clean_test": cmd_clean_test}[a.cmd](a)
