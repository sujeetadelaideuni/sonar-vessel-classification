"""
demo.py - live demo for the COMP 5064 presentation.

Loads the saved models, takes a few real one-second test clips (balanced across classes),
and shows what each model predicts against the true label. Also saves a picture of the
spectrograms and the clips as .wav files, so you can listen to what the model "hears".

Examples (run from the project folder):
  python demo.py --test inclusion_2000_exclusion_4000-clean
  python demo.py --test inclusion_4000_exclusion_6000-clean --per_class 2 --seed 3
  python demo.py --test deepship --models models\\cnn_scratch_nfull_s0_inclusion_2000_exclusion_4000.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonar import CLASSES, load_set, to_float, build_cnn, build_panns, mel_fb

DEFAULT_MODELS = [
    "models/cnn_scratch_nfull_s0_inclusion_2000_exclusion_4000.pt",
    "models/panns_ft-last2_n4000_s0_inclusion_2000_exclusion_4000.pt",
]


def load_model(path, dev):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    if meta["model"] == "cnn":
        m = build_cnn(len(CLASSES), ck["sr"]); label = "Scratch CNN"
    else:
        m = build_panns(len(CLASSES), norm=meta.get("norm", False)); label = "PANNs CNN14"
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    n_params = sum(p.numel() for p in m.parameters())
    return m, label, n_params


def probs(m, X, dev):
    import torch
    with torch.no_grad():
        return torch.softmax(m(to_float(X).to(dev)), 1).cpu().numpy()


def pick_clips(y, per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(len(CLASSES)):
        pool = np.where(y == c)[0]
        if len(pool):
            idx += list(rng.choice(pool, min(per_class, len(pool)), replace=False))
    rng.shuffle(idx)
    return np.array(idx)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--test", default="inclusion_2000_exclusion_4000-clean",
                    help="test set name, e.g. inclusion_2000_exclusion_4000-clean, ...3000_exclusion_5000-clean, ...4000_exclusion_6000-clean, deepship")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--per_class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--play", action="store_true", help="Windows only: play each clip through the speakers")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    print(f"\n=== Vessel type classification demo  |  device: {dev}  |  test set: {a.test} ===")

    X, y, _, sr = load_set(a.cache, a.test)
    names = {c: CLASSES[c] for c in range(len(CLASSES))}
    idx = pick_clips(y, a.per_class, a.seed)
    Xs, ys = X[idx], y[idx]
    print(f"Test set has {len(y)} one-second clips ({sr} Hz). Showing {len(idx)} random clips, balanced by class.\n")

    models = []
    for p in a.models:
        if not os.path.exists(p):
            print(f"[skip] model file not found: {p}"); continue
        m, label, n = load_model(p, dev)
        print(f"Loaded {label}: {n:,} parameters  ({os.path.basename(p)})")
        models.append((label, probs(m, Xs, dev)))
    if not models:
        sys.exit("No model files found.")

    print()
    head = f"{'#':>2}  {'TRUE':<14}" + "".join(f"{lab:<30}" for lab, _ in models)
    print(head); print("-" * len(head))
    correct = [0] * len(models)
    for i in range(len(idx)):
        row = f"{i+1:>2}  {names[ys[i]]:<14}"
        for k, (lab, P) in enumerate(models):
            c = int(P[i].argmax()); ok = c == ys[i]; correct[k] += ok
            row += f"{('OK  ' if ok else 'MISS') + ' ' + names[c] + f' ({P[i].max()*100:.0f}%)':<30}"
        print(row)
    print("-" * len(head))
    print("Correct on these clips: " + "   ".join(f"{lab}: {c}/{len(idx)}" for (lab, _), c in zip(models, correct)))
    print("(A handful of clips is only an illustration. Full-test-set numbers: macro-F1 in results/*.json.)")

    # ---- picture: log-mel spectrogram of every clip, titled with the true label and the first model's guess
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav
    fb = mel_fb(sr, 1024, 64); win = np.hanning(1025)[:-1]
    ncol = 4; nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.7 * nrow), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(idx): ax.axis("off"); continue
        w = Xs[i].astype(np.float32) / 32768.0
        frames = np.stack([w[s:s + 1024] * win for s in range(0, len(w) - 1024, 256)], 1)
        S = np.abs(np.fft.rfft(frames, axis=0)) ** 2
        M = 10 * np.log10(fb @ S + 1e-10)
        ax.imshow(M, origin="lower", aspect="auto", cmap="magma", extent=[0, 1, 0, sr / 2 / 1000])
        title = f"#{i+1} true: {names[ys[i]]}"
        for lab, P in models:
            c = int(P[i].argmax())
            title += f"\n{lab.split()[0]}: {names[c]} {P[i].max()*100:.0f}% {'OK' if c == ys[i] else 'MISS'}"
        ax.set_title(title, fontsize=8); ax.set_xlabel("time (s)", fontsize=7); ax.set_ylabel("kHz", fontsize=7)
        ax.tick_params(labelsize=7)
        wav.write(os.path.join(a.out, "clips", f"clip{i+1:02d}_{names[ys[i]]}.wav"), sr, Xs[i])
    fig.tight_layout(); png = os.path.join(a.out, "demo_spectrograms.png"); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\nSaved: {png}  and  {len(idx)} audio clips in {os.path.join(a.out, 'clips')}")

    if a.play:
        try:
            import winsound, time
            for i in range(len(idx)):
                f = os.path.join(a.out, "clips", f"clip{i+1:02d}_{names[ys[i]]}.wav")
                print(f"playing #{i+1} ({names[ys[i]]})"); winsound.PlaySound(f, winsound.SND_FILENAME); time.sleep(0.3)
        except ImportError:
            print("--play works on Windows only")


if __name__ == "__main__":
    main()
