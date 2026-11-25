"""
summarise.py - collect results/*.json into one table + report figures (grayscale, print-safe).

  python summarise.py            -> report/results_table.csv, report/results_table.md,
                                    report/fig_macro_f1.png, report/fig_val_curves.png
Only reads files written by sonar.py (names containing "__on_").
"""
import glob, json, os, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]
os.makedirs("report", exist_ok=True)
rows = []
for f in sorted(glob.glob("results/*__on_*.json")):
    r = json.load(open(f)); m = r["meta"]
    row = dict(run=r["run"], model=m["model"], mode=m["mode"], aug=m["aug"], n_train=m["n_train"], seed=m["seed"],
               train_scenario=m["train_scenario"], test_set=r["test_set"], n_test=r["n_test"],
               macro_f1=round(r["test_macro_f1"], 4), accuracy=round(r["test_accuracy"], 4),
               best_val_f1=round(m["best_val_macro_f1"], 4), best_epoch=m["best_epoch"],
               train_min=round(m["train_minutes"], 1), device=m.get("device", ""))
    for c in CLASSES:
        if c in r["report"]: row[f"f1_{c}"] = round(r["report"][c]["f1-score"], 3)
    if "recording_vote_accuracy" in r: row["recording_vote_acc"] = round(r["recording_vote_accuracy"], 3)
    rows.append(row)
if not rows: raise SystemExit("no results/*__on_*.json found - run sonar.py train first")

# ---- full table
cols = list(dict.fromkeys(k for r in rows for k in r))
with open("report/results_table.csv", "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows: fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

# ---- seeds pooled: same config, different seed -> mean +- sd
def cfg(r): return (r["model"], r["mode"], r["aug"], r["n_train"], r["train_scenario"], r["test_set"])
groups = {}
for r in rows: groups.setdefault(cfg(r), []).append(r["macro_f1"])
label = lambda k: f"{k[0].upper()}{' +aug' if k[2] else ''} ({k[1]}, n={k[3]})"
md = ["| Model | Train scenario | Test set | Macro-F1 | Runs |", "|---|---|---|---|---|"]
for k, v in groups.items():
    s = f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    md.append(f"| {label(k)} | {k[4]} | {k[5]} | {s} | {len(v)} |")
open("report/results_table.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---- Figure: macro-F1 per configuration and test set (horizontal bars, one axis)
items = sorted(groups.items(), key=lambda kv: (kv[0][5], kv[0][0], kv[0][1], str(kv[0][3])))
def shade(model, mode):   # CNN light; PANNs raw input mid-grey; PANNs loudness-normalised dark
    return "0.82" if model == "cnn" else ("0.30" if mode.endswith("norm") else "0.55")
def hatch(mode):          # texture carries the training regime so identity is not grey-level alone
    return "xx" if mode.startswith("probe") else "//" if mode.startswith("frozen") else ".." if mode.startswith("ft-all") else ""
fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(items) + 1.0))
for i, (k, v) in enumerate(items):
    mean = st.mean(v)
    ax.barh(i, mean, color=shade(k[0], k[1]), hatch=hatch(k[1]), edgecolor="black", linewidth=0.6, height=0.7)
    if len(v) > 1: ax.errorbar(mean, i, xerr=st.stdev(v), color="black", capsize=3, linewidth=1)
    ax.text(mean + 0.01, i, f"{mean:.2f}", va="center", fontsize=8)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([f"{label(k)}  →  {k[5].replace('inclusion_', 'inc').replace('_exclusion_', '/exc')}" for k, _ in items], fontsize=7)
ax.set_xlim(0, 1); ax.set_xlabel("Test macro-F1"); ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="0.9", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("report/fig_macro_f1.png", dpi=200); plt.close(fig)

# ---- Figure: validation macro-F1 per epoch (one line per training run, seed 0 only to stay readable)
styles = ["-", "--", ":", "-."]; greys = ["0.0", "0.45", "0.65"]
seen = {}
for f in sorted(glob.glob("results/*__on_*.json")):
    r = json.load(open(f)); m = r["meta"]
    if m["seed"] == 0 and r["run"] not in seen and not m["mode"].startswith("probe"): seen[r["run"]] = m
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for i, (run, m) in enumerate(seen.items()):
    h = m["history"]; x = [e["epoch"] for e in h]; y = [e["val_macro_f1"] for e in h]
    lab = f"{m['model'].upper()}{' +aug' if m['aug'] else ''} ({m['mode']}, n={m['n_train']})"
    ax.plot(x, y, styles[i % 4], color=greys[(i // 4) % 3], linewidth=1.6, label=lab)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1"); ax.set_ylim(0, 1)
ax.spines[["top", "right"]].set_visible(False); ax.grid(color="0.92", linewidth=0.6)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("report/fig_val_curves.png", dpi=200); plt.close(fig)
print("\nwrote report/results_table.csv, report/results_table.md, report/fig_macro_f1.png, report/fig_val_curves.png")
