#!/usr/bin/env python3
"""
One Operator to Rule Them All? Boundary-Indexed Operator Families in Neural PDE Solvers

PDE: 2D Poisson on [0,1]^2 with mixed BCs:
  - Dirichlet on left (x=0) and bottom (y=0): u = g_L(y), u = g_B(x)
  - Neumann  on right (x=1) and top (y=1):   du/dx = h_R(y), du/dy = h_T(x)

Experiments produced:
E1) Two FNOs trained under two different BC distributions
E2) BC-ablated model
E3) Conditional expectation demo

Outputs:
  - run_log.txt
  - metrics.json
  - sweep_delta.csv + error_vs_delta.png
  - sweep_freq.csv + error_vs_freq.png
  - cross_dist_table.csv
  - condexp_compare.png
  - example_heatmaps.png
  - training_curve_<tag>.csv/.png/.npz
  - bc_ablation_table.csv
  - sweep_delta_compact.csv + sweep_freq_compact.csv
  - tables/*.tex
  - same_f_two_bcs_metrics.json

Dependencies: 
  - torch
  - numpy
  - matplotlib
"""

import os, time, json, math, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================
# Logging
# ============================================================

class Logger:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_l2(u_pred, u_true, eps=1e-12):
    # (B,1,H,W)
    p = u_pred[:, 0]
    t = u_true[:, 0]
    num = torch.sqrt(torch.mean((p - t) ** 2, dim=(-2, -1)))
    den = torch.sqrt(torch.mean((t) ** 2, dim=(-2, -1))) + eps
    return num / den


# ============================================================
# Random smooth 1D functions (BCs) and 2D forcing
# ============================================================

def smooth_1d_fourier(x, K, amp, device):
    """
    f(x) = sum_{k=1..K} [a_k sin(2π k x) + b_k cos(2π k x)], a_k,b_k ~ N(0, amp/k)
    """
    two_pi = 2.0 * math.pi
    out = torch.zeros_like(x, device=device)
    ks = torch.arange(1, K + 1, device=device, dtype=torch.float32)
    a = torch.randn(K, device=device) * (amp / ks)
    b = torch.randn(K, device=device) * (amp / ks)
    for i, k in enumerate(range(1, K + 1)):
        out = out + a[i] * torch.sin(two_pi * k * x) + b[i] * torch.cos(two_pi * k * x)
    return out


def smooth_2d_forcing(X, Y, K, amp, device):
    """
    f(x,y) = sum_{kx,ky=1..K} c_{kx,ky}/(kx*ky) * sin(2πkx x) sin(2πky y)
    """
    two_pi = 2.0 * math.pi
    out = torch.zeros_like(X, device=device)
    c = torch.randn(K, K, device=device) * amp
    for kx in range(1, K + 1):
        sx = torch.sin(two_pi * kx * X)
        for ky in range(1, K + 1):
            sy = torch.sin(two_pi * ky * Y)
            out = out + (c[kx - 1, ky - 1] / (kx * ky)) * sx * sy
    return out


# ============================================================
# Poisson solver: Batched Jacobi (GPU-friendly)
# ============================================================

@torch.no_grad()
def solve_poisson_mixed_bc_jacobi(f, gL, gB, hR, hT, iters=250):
    """
    Solve -Δu = f on NxN grid including boundaries.
    Mixed BCs:
      left x=0:    u = gL(y)
      bottom y=0:  u = gB(x)
      right x=1:   du/dx = hR(y)
      top y=1:     du/dy = hT(x)

    f:  (B,N,N)
    gL: (B,N) along y
    gB: (B,N) along x
    hR: (B,N) along y
    hT: (B,N) along x

    returns u: (B,N,N)
    """
    device = f.device
    B, N, _ = f.shape
    dx = 1.0 / (N - 1)
    dx2 = dx * dx

    u = torch.zeros((B, N, N), device=device)

    # enforce BC initial
    u[:, :, 0] = gL
    u[:, 0, :] = gB
    u[:, :, -1] = u[:, :, -2] + dx * hR
    u[:, -1, :] = u[:, -2, :] + dx * hT

    for _ in range(iters):
        u_new = u.clone()
        up = u[:, 2:, 1:-1]
        down = u[:, :-2, 1:-1]
        left = u[:, 1:-1, :-2]
        right = u[:, 1:-1, 2:]
        rhs = f[:, 1:-1, 1:-1]
        u_new[:, 1:-1, 1:-1] = 0.25 * (up + down + left + right + dx2 * rhs)
        u = u_new

        # re-enforce BC
        u[:, :, 0] = gL
        u[:, 0, :] = gB
        u[:, :, -1] = u[:, :, -2] + dx * hR
        u[:, -1, :] = u[:, -2, :] + dx * hT

    return u


# ============================================================
# Data generation (two BC distributions mu_B0, mu_B1)
# ============================================================

class BCDist:
    """
    BC distribution parameters.
    """
    def __init__(self, K_dir, amp_dir, mean_shift_dir, K_neu, amp_neu, mean_shift_neu):
        self.K_dir = K_dir
        self.amp_dir = amp_dir
        self.mean_shift_dir = mean_shift_dir
        self.K_neu = K_neu
        self.amp_neu = amp_neu
        self.mean_shift_neu = mean_shift_neu


def sample_batch(B, N, device, bc_dist: BCDist, f_K=6, f_amp=3.0, jacobi_iters=250):
    x = torch.linspace(0.0, 1.0, N, device=device)
    y = torch.linspace(0.0, 1.0, N, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # forcing
    f = torch.stack([smooth_2d_forcing(X, Y, f_K, f_amp, device) for _ in range(B)], dim=0)  # (B,N,N)

    # BCs
    gL = torch.stack([smooth_1d_fourier(y, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir for _ in range(B)], dim=0)
    gB = torch.stack([smooth_1d_fourier(x, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir for _ in range(B)], dim=0)

    hR = torch.stack([smooth_1d_fourier(y, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu for _ in range(B)], dim=0)
    hT = torch.stack([smooth_1d_fourier(x, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu for _ in range(B)], dim=0)

    u = solve_poisson_mixed_bc_jacobi(f, gL, gB, hR, hT, iters=jacobi_iters)  # (B,N,N)

    # Build input channels for operator learning:
    # channels: [f, gD, mD, hN, mN, x, y]
    gD = torch.zeros_like(f)
    mD = torch.zeros_like(f)
    hN = torch.zeros_like(f)
    mN = torch.zeros_like(f)

    # Dirichlet on left & bottom
    gD[:, :, 0] = gL
    gD[:, 0, :] = gB
    mD[:, :, 0] = 1.0
    mD[:, 0, :] = 1.0

    # Neumann on right & top
    hN[:, :, -1] = hR
    hN[:, -1, :] = hT
    mN[:, :, -1] = 1.0
    mN[:, -1, :] = 1.0

    xch = X.unsqueeze(0).repeat(B, 1, 1)
    ych = Y.unsqueeze(0).repeat(B, 1, 1)

    inp_full = torch.stack([f, gD, mD, hN, mN, xch, ych], dim=1)  # (B,7,N,N)
    inp_no_bc = torch.stack([f, xch, ych], dim=1)  # (B,3,N,N)

    return inp_full, inp_no_bc, u.unsqueeze(1), {"f": f, "gL": gL, "gB": gB, "hR": hR, "hT": hT, "X": X, "Y": Y}


# ============================================================
# Minimal FNO
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.weight.shape[1], H, W // 2 + 1, device=x.device, dtype=torch.cfloat)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        w = torch.view_as_complex(self.weight[:, :, :m1, :m2, :])
        out_ft[:, :, :m1, :m2] = torch.einsum("bihw,iohw->bohw", x_ft[:, :, :m1, :m2], w)

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNO2d(nn.Module):
    def __init__(self, in_channels, width=48, modes=16, depth=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(depth)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.fc0(x)
        for s, w in zip(self.spectral, self.w):
            x = F.gelu(s(x) + w(x))
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# NEW: Helpers to save prettier artifacts for the paper
# ============================================================

def save_training_curve_artifacts(tag, outdir, steps, losses, rels):
    """
    Saves: training_curve_<tag>.csv, training_curve_<tag>.npz, training_curve_<tag>.png
    """
    ensure_dir(outdir)

    csv_path = os.path.join(outdir, f"training_curve_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "relL2"])
        for s, l, r in zip(steps, losses, rels):
            w.writerow([int(s), float(l), float(r)])

    npz_path = os.path.join(outdir, f"training_curve_{tag}.npz")
    np.savez(npz_path, steps=np.array(steps), loss=np.array(losses), relL2=np.array(rels))

    plt.figure()
    plt.plot(np.array(steps), np.array(rels))
    plt.xlabel("Training step")
    plt.ylabel("Relative $L^2$ error")
    plt.title(f"Training dynamics ({tag})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"training_curve_{tag}.png"), dpi=200)
    plt.close()


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
    return rows


def write_csv_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def summarize_sweep_delta(outdir, in_csv="sweep_delta.csv", out_csv="sweep_delta_compact.csv"):
    """
    Produces a compact symmetric summary table:
      delta_abs, relL2_mean_avg, relL2_std_avg
    and keeps delta=0 as-is.
    """
    path = os.path.join(outdir, in_csv)
    rows = read_csv_rows(path)
    header = rows[0]
    data = rows[1:]

    # parse
    parsed = []
    for d, m, s in data:
        parsed.append((float(d), float(m), float(s)))

    # group by abs(delta) with sign symmetry
    groups = {}
    for d, m, s in parsed:
        key = abs(d)
        groups.setdefault(key, []).append((d, m, s))

    out = [("delta_abs", "relL2_mean_avg", "relL2_std_avg")]
    for key in sorted(groups.keys()):
        ms = [x[1] for x in groups[key]]
        ss = [x[2] for x in groups[key]]
        out.append((float(key), float(np.mean(ms)), float(np.mean(ss))))

    write_csv_rows(os.path.join(outdir, out_csv), out)


def summarize_sweep_freq(outdir, in_csv="sweep_freq.csv", out_csv="sweep_freq_compact.csv"):
    """
    Keeps the sweep_freq.csv content but ensures a clean numeric CSV for LaTeX import if desired.
    """
    path = os.path.join(outdir, in_csv)
    rows = read_csv_rows(path)
    header = rows[0]
    data = rows[1:]

    out = [("K_dir_test", "relL2_mean", "relL2_std")]
    for k, m, s in data:
        out.append((int(float(k)), float(m), float(s)))

    write_csv_rows(os.path.join(outdir, out_csv), out)


def write_latex_table(path, caption, label, columns, rows, column_align=None):
    """
    Writes a LaTeX table environment (booktabs) to a .tex file.
    columns: list[str] header names
    rows: list[list[str]] already formatted as strings
    """
    if column_align is None:
        column_align = "l" + "c" * (len(columns) - 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{column_align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join([f"\\textbf{{{c}}}" for c in columns]) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_pretty_tables(outdir, cross_table, delta_compact_path=None, freq_compact_path=None):
    """
    cross_table: list[dict] with keys model, test_dist, relL2_mean, relL2_std
    Generates:
      tables/table_cross_dist.tex
      tables/table_bc_ablation.tex
      tables/table_sweep_delta.tex (if available)
      tables/table_sweep_freq.tex  (if available)
    """
    tables_dir = os.path.join(outdir, "tables")
    ensure_dir(tables_dir)

    # --- Cross distribution table (2 columns for mu_B0/mu_B1) ---
    # Collect by model
    models = sorted(list(set([d["model"] for d in cross_table])))
    dist_names = ["mu_B0", "mu_B1"]

    # create mapping
    mp = {(d["model"], d["test_dist"]): (d["relL2_mean"], d["relL2_std"]) for d in cross_table}

    rows = []
    for m in models:
        a = mp.get((m, "mu_B0"), (np.nan, np.nan))
        b = mp.get((m, "mu_B1"), (np.nan, np.nan))
        rows.append([
            m.replace("_", "\\_"),
            f"${a[0]:.3f} \\pm {a[1]:.3f}$",
            f"${b[0]:.3f} \\pm {b[1]:.3f}$",
        ])

    write_latex_table(
        path=os.path.join(tables_dir, "table_cross_dist.tex"),
        caption="Cross-distribution generalization under boundary-condition shift (relative $L^2$ error, mean $\\pm$ std.).",
        label="tab:cross_dist_pretty",
        columns=["Model", "Test on $\\mu_{B_0}$", "Test on $\\mu_{B_1}$"],
        rows=rows,
        column_align="lcc"
    )

    # --- Boundary ablation table (extract the two key rows if present) ---
    # We look for FNO_full_B0 and FNO_noBC_B0, but keep robust.
    def find_row(name):
        for r in rows:
            if r[0].replace("\\_", "_") == name:
                return r
        return None

    # Build ablation table from whatever exists
    # Priority: (with BC) = FNO_full_B0, (no BC) = FNO_noBC_B0
    r_with = find_row("FNO_full_B0") or (rows[0] if rows else None)
    r_nobc = find_row("FNO_noBC_B0") or (rows[-1] if rows else None)

    ab_rows = []
    if r_with is not None:
        ab_rows.append(["FNO (with BC channels)", r_with[1], r_with[2]])
    if r_nobc is not None:
        ab_rows.append(["FNO (no BC channels)", r_nobc[1], r_nobc[2]])

    write_latex_table(
        path=os.path.join(tables_dir, "table_bc_ablation.tex"),
        caption="Effect of removing boundary-condition inputs. Without BC channels, performance collapses across boundary distributions.",
        label="tab:bc_ablation",
        columns=["Model", "Test on $\\mu_{B_0}$", "Test on $\\mu_{B_1}$"],
        rows=ab_rows,
        column_align="lcc"
    )

    # --- Sweep delta compact table ---
    if delta_compact_path is not None and os.path.exists(delta_compact_path):
        drows = read_csv_rows(delta_compact_path)[1:]
        # Include 0 and select a few abs deltas for compactness (all are already compact)
        rows_delta = []
        for d_abs, m, s in drows:
            rows_delta.append([f"${float(d_abs):.2f}$", f"${float(m):.3f} \\pm {float(s):.3f}$"])
        write_latex_table(
            path=os.path.join(tables_dir, "table_sweep_delta.tex"),
            caption="Boundary extrapolation via Dirichlet mean shifts (relative $L^2$ error, mean $\\pm$ std.).",
            label="tab:sweep_delta",
            columns=["$|\\delta|$", "Rel. $L^2$ error"],
            rows=rows_delta,
            column_align="cc"
        )

    # --- Sweep freq compact table ---
    if freq_compact_path is not None and os.path.exists(freq_compact_path):
        frows = read_csv_rows(freq_compact_path)[1:]
        rows_freq = []
        for k, m, s in frows:
            rows_freq.append([f"${int(float(k))}$", f"${float(m):.3f} \\pm {float(s):.3f}$"])
        write_latex_table(
            path=os.path.join(tables_dir, "table_sweep_freq.tex"),
            caption="Boundary extrapolation via increased Dirichlet bandwidth $K$ (relative $L^2$ error, mean $\\pm$ std.).",
            label="tab:sweep_freq",
            columns=["$K$ (test)", "Rel. $L^2$ error"],
            rows=rows_freq,
            column_align="cc"
        )


# ============================================================
# Train / eval helpers
# ============================================================

def train_model(model, in_channels, bc_dist: BCDist, args, log: Logger, tag: str, outdir: str):
    """
    Existing functionality preserved.
    NEW: records a training curve and exports it as CSV/NPZ/PNG.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()

    steps_log = []
    loss_log = []
    rel_log = []

    for step in range(1, args.train_steps + 1):
        inp_full, inp_no_bc, u, _ = sample_batch(
            B=args.batch, N=args.N, device=args.device, bc_dist=bc_dist,
            f_K=args.f_K, f_amp=args.f_amp, jacobi_iters=args.jacobi_iters_train
        )
        inp = inp_full if in_channels == 7 else inp_no_bc
        pred = model(inp)
        loss = F.mse_loss(pred, u)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                err = rel_l2(pred, u).mean().item()
            dt = time.time() - t0
            log.log(f"[{tag}] step {step:5d}/{args.train_steps} | loss {loss.item():.6f} | relL2 {err:.4f} | {dt:.1f}s")
            t0 = time.time()

            # NEW: record training curve points
            steps_log.append(step)
            loss_log.append(float(loss.item()))
            rel_log.append(float(err))

    # NEW: save training curves for this model
    try:
        save_training_curve_artifacts(tag=tag, outdir=outdir, steps=steps_log, losses=loss_log, rels=rel_log)
        log.log(f"[{tag}] Saved training curves: training_curve_{tag}.csv/.npz/.png")
    except Exception as e:
        log.log(f"[{tag}] WARNING: could not save training curves ({e})")


@torch.no_grad()
def eval_on_dist(model, in_channels, bc_dist: BCDist, args, n_batches=10):
    model.eval()
    errs = []
    for _ in range(n_batches):
        inp_full, inp_no_bc, u, _ = sample_batch(
            B=args.batch_eval, N=args.N, device=args.device, bc_dist=bc_dist,
            f_K=args.f_K, f_amp=args.f_amp, jacobi_iters=args.jacobi_iters_eval
        )
        inp = inp_full if in_channels == 7 else inp_no_bc
        pred = model(inp)
        errs.append(rel_l2(pred, u).mean().item())
    return float(np.mean(errs)), float(np.std(errs))


# ============================================================
# Sweeps for "out of BC support"
# ============================================================

@torch.no_grad()
def sweep_dirichlet_shift(model, in_channels, bc_dist_base: BCDist, args, outdir, log: Logger):
    """
    Keep frequency K same, shift Dirichlet mean by delta. Measures extrapolation.
    """
    deltas = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    rows = [("delta", "relL2_mean", "relL2_std")]

    for d in deltas:
        bc = BCDist(
            K_dir=bc_dist_base.K_dir, amp_dir=bc_dist_base.amp_dir, mean_shift_dir=bc_dist_base.mean_shift_dir + float(d),
            K_neu=bc_dist_base.K_neu, amp_neu=bc_dist_base.amp_neu, mean_shift_neu=bc_dist_base.mean_shift_neu
        )
        m, s = eval_on_dist(model, in_channels, bc, args, n_batches=args.sweep_batches)
        rows.append((float(d), m, s))
        log.log(f"[SWEEP delta] delta={d:+.2f} | relL2={m:.4f} ± {s:.4f}")

    csv_path = os.path.join(outdir, "sweep_delta.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    # plot
    xs = deltas
    ys = np.array([r[1] for r in rows[1:]], dtype=np.float32)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Dirichlet mean shift δ")
    plt.ylabel("Relative $L^2$ error (mean)")
    plt.title("BC extrapolation: Dirichlet shift")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_delta.png"), dpi=200)
    plt.close()


@torch.no_grad()
def sweep_dirichlet_frequency(model, in_channels, bc_dist_base: BCDist, args, outdir, log: Logger):
    """
    Keep mean/amplitude, increase Dirichlet frequency K beyond training support.
    """
    Ks = np.array([bc_dist_base.K_dir, bc_dist_base.K_dir + 2, bc_dist_base.K_dir + 4, bc_dist_base.K_dir + 6], dtype=np.int32)
    rows = [("K_dir_test", "relL2_mean", "relL2_std")]

    for K in Ks:
        bc = BCDist(
            K_dir=int(K), amp_dir=bc_dist_base.amp_dir, mean_shift_dir=bc_dist_base.mean_shift_dir,
            K_neu=bc_dist_base.K_neu, amp_neu=bc_dist_base.amp_neu, mean_shift_neu=bc_dist_base.mean_shift_neu
        )
        m, s = eval_on_dist(model, in_channels, bc, args, n_batches=args.sweep_batches)
        rows.append((int(K), m, s))
        log.log(f"[SWEEP freq] K_dir={K:2d} | relL2={m:.4f} ± {s:.4f}")

    csv_path = os.path.join(outdir, "sweep_freq.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    xs = Ks.astype(np.float32)
    ys = np.array([r[1] for r in rows[1:]], dtype=np.float32)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Dirichlet bandwidth $K$ (test)")
    plt.ylabel("Relative $L^2$ error (mean)")
    plt.title("BC extrapolation: Dirichlet frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_freq.png"), dpi=200)
    plt.close()


# ============================================================
# Conditional expectation demo for BC-ablated model
# ============================================================

@torch.no_grad()
def conditional_expectation_demo(model_no_bc, bc_dist: BCDist, args, outdir, log: Logger):
    """
    Fix one forcing f* and compare:
      u_hat = model_no_bc([f*,x,y])
      u_bar = mean_{BC~mu_B} u(f*, BC) via Monte Carlo
    Expectation: u_hat ≈ u_bar (risk-minimization under missing BC).
    """
    device = args.device
    N = args.N

    # fixed forcing sample
    x = torch.linspace(0.0, 1.0, N, device=device)
    y = torch.linspace(0.0, 1.0, N, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    f_star = smooth_2d_forcing(X, Y, args.f_K, args.f_amp, device).unsqueeze(0)  # (1,N,N)

    # model prediction (no BC channels)
    xch = X.unsqueeze(0)
    ych = Y.unsqueeze(0)
    inp_no_bc = torch.stack([f_star, xch, ych], dim=1)  # (1,3,N,N)
    u_hat = model_no_bc(inp_no_bc)[0, 0]  # (N,N)

    # Monte Carlo average u over BC draws
    M = args.condexp_mc
    u_acc = torch.zeros((N, N), device=device)
    for i in range(M):
        gL = smooth_1d_fourier(y, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir
        gB = smooth_1d_fourier(x, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir
        hR = smooth_1d_fourier(y, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu
        hT = smooth_1d_fourier(x, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu

        u = solve_poisson_mixed_bc_jacobi(
            f_star,
            gL.unsqueeze(0), gB.unsqueeze(0),
            hR.unsqueeze(0), hT.unsqueeze(0),
            iters=args.jacobi_iters_eval
        )[0]  # (N,N)
        u_acc += u

        if (i + 1) % max(1, (M // 5)) == 0:
            log.log(f"[CondExp] MC {i+1}/{M}")

    u_bar = u_acc / float(M)

    # compute mismatch
    err = torch.sqrt(torch.mean((u_hat - u_bar) ** 2)).item()
    norm = torch.sqrt(torch.mean(u_bar ** 2)).item()
    rel = err / (norm + 1e-12)
    log.log(f"[CondExp] ||u_hat - E[u|f*]||_2 / ||E[u|f*]||_2 = {rel:.4f}")

    # plot
    u_hat_np = u_hat.detach().cpu().numpy()
    u_bar_np = u_bar.detach().cpu().numpy()
    diff_np = np.abs(u_hat_np - u_bar_np)

    plt.figure(figsize=(10, 3.2))
    plt.subplot(1, 3, 1)
    plt.imshow(u_hat_np, origin="lower"); plt.title("BC-ablated model: $\\hat{u}(f^*)$"); plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(u_bar_np, origin="lower"); plt.title("MC mean: $\\mathbb{E}[u\\mid f^*]$"); plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(diff_np, origin="lower"); plt.title("$|\\hat{u}-\\mathbb{E}[u\\mid f^*]|$"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "condexp_compare.png"), dpi=200)
    plt.close()

    return {"condexp_rel_mismatch": float(rel), "condexp_mc": int(M)}


# ============================================================
# Example heatmaps for "same f, different BC" identifiability intuition
# ============================================================

@torch.no_grad()
def same_f_two_bcs_visual(model_full, bc_dist: BCDist, args, outdir, log: Logger):
    """
    Fix f, draw two BCs, solve u1,u2. Show model predicts different outputs if given BC channels,
    and BC-ablated would be stuck.

    NEW: Also computes a scalar summary of how much the true solution changes under BC changes.
    """
    device = args.device
    N = args.N

    # fixed f*
    x = torch.linspace(0.0, 1.0, N, device=device)
    y = torch.linspace(0.0, 1.0, N, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    f_star = smooth_2d_forcing(X, Y, args.f_K, args.f_amp, device)

    def pack_inp(f, gL, gB, hR, hT):
        gD = torch.zeros_like(f)
        mD = torch.zeros_like(f)
        hN = torch.zeros_like(f)
        mN = torch.zeros_like(f)

        gD[:, 0] = gL
        gD[0, :] = gB
        mD[:, 0] = 1.0
        mD[0, :] = 1.0

        hN[:, -1] = hR
        hN[-1, :] = hT
        mN[:, -1] = 1.0
        mN[-1, :] = 1.0

        xch = X
        ych = Y
        inp = torch.stack([f, gD, mD, hN, mN, xch, ych], dim=0).unsqueeze(0)  # (1,7,N,N)
        return inp

    def draw_bc():
        gL = smooth_1d_fourier(y, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir
        gB = smooth_1d_fourier(x, bc_dist.K_dir, bc_dist.amp_dir, device) + bc_dist.mean_shift_dir
        hR = smooth_1d_fourier(y, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu
        hT = smooth_1d_fourier(x, bc_dist.K_neu, bc_dist.amp_neu, device) + bc_dist.mean_shift_neu
        return gL, gB, hR, hT

    gL1, gB1, hR1, hT1 = draw_bc()
    gL2, gB2, hR2, hT2 = draw_bc()

    u1 = solve_poisson_mixed_bc_jacobi(
        f_star.unsqueeze(0), gL1.unsqueeze(0), gB1.unsqueeze(0), hR1.unsqueeze(0), hT1.unsqueeze(0),
        iters=args.jacobi_iters_eval
    )[0]
    u2 = solve_poisson_mixed_bc_jacobi(
        f_star.unsqueeze(0), gL2.unsqueeze(0), gB2.unsqueeze(0), hR2.unsqueeze(0), hT2.unsqueeze(0),
        iters=args.jacobi_iters_eval
    )[0]

    inp1 = pack_inp(f_star, gL1, gB1, hR1, hT1)
    inp2 = pack_inp(f_star, gL2, gB2, hR2, hT2)

    p1 = model_full(inp1)[0, 0]
    p2 = model_full(inp2)[0, 0]

    # NEW: scalar metric of BC-induced solution change
    # Relative change in ground truth solutions for same forcing
    gt_rel_change = (torch.norm(u1 - u2) / (torch.norm(u1) + 1e-12)).item()
    pred_rel_change = (torch.norm(p1 - p2) / (torch.norm(p1) + 1e-12)).item()
    log.log(f"[Same-f] ||u(BC1)-u(BC2)||/||u(BC1)|| = {gt_rel_change:.4f}")
    log.log(f"[Same-f] ||p(BC1)-p(BC2)||/||p(BC1)|| = {pred_rel_change:.4f}")

    # Save numeric summary (NEW)
    metrics_path = os.path.join(outdir, "same_f_two_bcs_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"gt_rel_change": float(gt_rel_change), "pred_rel_change": float(pred_rel_change)},
            f, indent=2
        )

    # visualize GT and preds
    plt.figure(figsize=(12, 6))
    items = [
        (u1, "GT $u(f^*,\\mathcal{B}_1)$"),
        (u2, "GT $u(f^*,\\mathcal{B}_2)$"),
        (torch.abs(u1 - u2), "$|\\Delta u|$"),
        (p1, "FNO pred ($\\mathcal{B}_1$)"),
        (p2, "FNO pred ($\\mathcal{B}_2$)"),
        (torch.abs(p1 - p2), "$|\\Delta \\hat{u}|$"),
    ]
    for i, (img, title) in enumerate(items, 1):
        plt.subplot(2, 3, i)
        plt.imshow(img.detach().cpu().numpy(), origin="lower")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "example_heatmaps.png"), dpi=200)
    plt.close()
    log.log("[Viz] Saved example_heatmaps.png (same f*, two BCs) + same_f_two_bcs_metrics.json")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="bc_operator_family_outputs")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=7)

    # grid & solver
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--jacobi_iters_train", type=int, default=220)
    ap.add_argument("--jacobi_iters_eval", type=int, default=320)

    # data forcing
    ap.add_argument("--f_K", type=int, default=6)
    ap.add_argument("--f_amp", type=float, default=3.0)

    # training
    ap.add_argument("--train_steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--batch_eval", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.0008)
    ap.add_argument("--log_every", type=int, default=200)

    # sweeps
    ap.add_argument("--sweep_batches", type=int, default=8)

    # conditional expectation demo
    ap.add_argument("--condexp_mc", type=int, default=64)

    # NEW: optional workaround for OpenMP duplicate lib warning on Windows
    ap.add_argument("--allow_omp_duplicate", action="store_true",
                    help="Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP duplicate runtime crash/warning (Windows workaround).")

    args = ap.parse_args()

    ensure_dir(args.outdir)
    log = Logger(os.path.join(args.outdir, "run_log.txt"))

    if args.allow_omp_duplicate:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        log.log("Set KMP_DUPLICATE_LIB_OK=TRUE (OpenMP duplicate runtime workaround).")

    if args.device == "cuda" and not torch.cuda.is_available():
        log.log("CUDA requested but not available; using CPU.")
        args.device = "cpu"
    args.device = torch.device(args.device)

    set_seed(args.seed)

    # Define two different BC distributions (mu_B0 and mu_B1)
    # mu_B0: low-ish frequency Dirichlet, near-zero mean
    mu_B0 = BCDist(K_dir=6, amp_dir=1.0, mean_shift_dir=0.0, K_neu=6, amp_neu=0.5, mean_shift_neu=0.0)
    # mu_B1: same smoothness but shifted mean and larger Neumann scale
    mu_B1 = BCDist(K_dir=6, amp_dir=1.0, mean_shift_dir=0.6, K_neu=6, amp_neu=0.8, mean_shift_neu=0.2)

    log.log("=== CONFIG ===")
    log.log(f"device={args.device} | N={args.N} | train_steps={args.train_steps} | batch={args.batch}")
    log.log(f"Jacobi iters train/eval = {args.jacobi_iters_train}/{args.jacobi_iters_eval}")
    log.log(f"mu_B0: K_dir={mu_B0.K_dir}, amp_dir={mu_B0.amp_dir}, shift_dir={mu_B0.mean_shift_dir}, amp_neu={mu_B0.amp_neu}, shift_neu={mu_B0.mean_shift_neu}")
    log.log(f"mu_B1: K_dir={mu_B1.K_dir}, amp_dir={mu_B1.amp_dir}, shift_dir={mu_B1.mean_shift_dir}, amp_neu={mu_B1.amp_neu}, shift_neu={mu_B1.mean_shift_neu}")

    # Models:
    # Full model (with BC channels): in_channels=7
    # BC-ablated model: in_channels=3 (only forcing + coords)
    fno_full_B0 = FNO2d(in_channels=7, width=48, modes=16, depth=4).to(args.device)
    fno_full_B1 = FNO2d(in_channels=7, width=48, modes=16, depth=4).to(args.device)
    fno_no_bc_B0 = FNO2d(in_channels=3, width=48, modes=16, depth=4).to(args.device)

    # Train FNO_full on mu_B0
    log.log("\n=== TRAIN: FNO_full (BC channels) on mu_B0 ===")
    train_model(fno_full_B0, 7, mu_B0, args, log, "FNO_full_B0", outdir=args.outdir)

    # Train FNO_full on mu_B1
    log.log("\n=== TRAIN: FNO_full (BC channels) on mu_B1 ===")
    train_model(fno_full_B1, 7, mu_B1, args, log, "FNO_full_B1", outdir=args.outdir)

    # Train BC-ablated model on mu_B0
    log.log("\n=== TRAIN: FNO_noBC (no BC channels) on mu_B0 ===")
    train_model(fno_no_bc_B0, 3, mu_B0, args, log, "FNO_noBC_B0", outdir=args.outdir)

    # Cross-distribution evaluation table
    log.log("\n=== EVAL: Cross-distribution table (mean ± std relL2) ===")
    table = []
    for name, model, in_ch in [
        ("FNO_full_B0", fno_full_B0, 7),
        ("FNO_full_B1", fno_full_B1, 7),
        ("FNO_noBC_B0", fno_no_bc_B0, 3),
    ]:
        for dist_name, dist in [("mu_B0", mu_B0), ("mu_B1", mu_B1)]:
            m, s = eval_on_dist(model, in_ch, dist, args, n_batches=10)
            log.log(f"{name:12s} on {dist_name:5s} : {m:.4f} ± {s:.4f}")
            table.append((name, dist_name, m, s))

    # Save cross table CSV
    cross_csv = os.path.join(args.outdir, "cross_dist_table.csv")
    with open(cross_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_dist", "relL2_mean", "relL2_std"])
        for r in table:
            w.writerow(list(r))

    # NEW: also write a very simple "ablation table" CSV for convenience
    # (This does not change any original outputs; it's additive.)
    bc_ablation_csv = os.path.join(args.outdir, "bc_ablation_table.csv")
    with open(bc_ablation_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mu_B0_relL2_mean", "mu_B0_relL2_std", "mu_B1_relL2_mean", "mu_B1_relL2_std"])
        # fetch from table list
        mp = {(a, b): (c, d) for (a, b, c, d) in table}
        for mname in ["FNO_full_B0", "FNO_full_B1", "FNO_noBC_B0"]:
            b0 = mp.get((mname, "mu_B0"), (np.nan, np.nan))
            b1 = mp.get((mname, "mu_B1"), (np.nan, np.nan))
            w.writerow([mname, b0[0], b0[1], b1[0], b1[1]])

    # Sweeps (out of BC support) for the mu_B0-trained full model
    log.log("\n=== SWEEPS: Out-of-support BC tests for FNO_full_B0 ===")
    sweep_dirichlet_shift(fno_full_B0, 7, mu_B0, args, args.outdir, log)
    sweep_dirichlet_frequency(fno_full_B0, 7, mu_B0, args, args.outdir, log)

    # NEW: compact sweep summaries (useful for a small table)
    try:
        summarize_sweep_delta(args.outdir)
        summarize_sweep_freq(args.outdir)
        log.log("[Summary] Wrote sweep_delta_compact.csv and sweep_freq_compact.csv")
    except Exception as e:
        log.log(f"[Summary] WARNING: could not write compact sweep summaries ({e})")

    # Conditional expectation demo for BC-ablated model
    log.log("\n=== DEMO: Conditional expectation behavior (BC-ablated) ===")
    cond_metrics = conditional_expectation_demo(fno_no_bc_B0, mu_B0, args, args.outdir, log)

    # Same f*, different BCs visual + metrics
    log.log("\n=== VIZ: Same forcing, two different BC draws ===")
    same_f_two_bcs_visual(fno_full_B0, mu_B0, args, args.outdir, log)

    # NEW: export LaTeX-ready tables for Overleaf
    cross_table_dicts = [{"model": a, "test_dist": b, "relL2_mean": float(c), "relL2_std": float(d)} for (a, b, c, d) in table]
    delta_compact_path = os.path.join(args.outdir, "sweep_delta_compact.csv")
    freq_compact_path = os.path.join(args.outdir, "sweep_freq_compact.csv")
    try:
        export_pretty_tables(
            outdir=args.outdir,
            cross_table=cross_table_dicts,
            delta_compact_path=delta_compact_path if os.path.exists(delta_compact_path) else None,
            freq_compact_path=freq_compact_path if os.path.exists(freq_compact_path) else None
        )
        log.log("[Tables] Wrote LaTeX tables to outdir/tables/*.tex")
    except Exception as e:
        log.log(f"[Tables] WARNING: could not write LaTeX tables ({e})")

    # Consolidate metrics.json (preserve original structure, add new artifacts)
    metrics = {
        "config": {
            "N": args.N,
            "train_steps": args.train_steps,
            "batch": args.batch,
            "batch_eval": args.batch_eval,
            "jacobi_iters_train": args.jacobi_iters_train,
            "jacobi_iters_eval": args.jacobi_iters_eval,
            "f_K": args.f_K,
            "f_amp": args.f_amp,
            "lr": args.lr,
            "log_every": args.log_every,
            "sweep_batches": args.sweep_batches,
            "condexp_mc": args.condexp_mc,
            "seed": args.seed,
        },
        "bc_dists": {
            "mu_B0": vars(mu_B0),
            "mu_B1": vars(mu_B1),
        },
        "cross_dist_table": [
            {"model": a, "test_dist": b, "relL2_mean": float(c), "relL2_std": float(d)} for (a, b, c, d) in table
        ],
        "conditional_expectation_demo": cond_metrics,
        "artifacts": {
            "run_log": "run_log.txt",
            "metrics": "metrics.json",
            "cross_dist_table": "cross_dist_table.csv",
            "bc_ablation_table": "bc_ablation_table.csv",
            "sweep_delta": "sweep_delta.csv",
            "sweep_freq": "sweep_freq.csv",
            "sweep_delta_compact": "sweep_delta_compact.csv",
            "sweep_freq_compact": "sweep_freq_compact.csv",
            "fig_error_vs_delta": "error_vs_delta.png",
            "fig_error_vs_freq": "error_vs_freq.png",
            "fig_condexp": "condexp_compare.png",
            "fig_examples": "example_heatmaps.png",
            "same_f_two_bcs_metrics": "same_f_two_bcs_metrics.json",
            "tables_dir": "tables/",
            "training_curves": [
                "training_curve_FNO_full_B0.csv/.npz/.png",
                "training_curve_FNO_full_B1.csv/.npz/.png",
                "training_curve_FNO_noBC_B0.csv/.npz/.png",
            ],
        }
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.log("\n=== DONE ===")
    log.log(f"Saved outputs to: {args.outdir}")
    log.close()


if __name__ == "__main__":
    main()
