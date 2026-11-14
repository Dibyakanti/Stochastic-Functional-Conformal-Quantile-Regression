"""GIF animations of the SFCQR-calibrated prediction band, for the README.

Works with every preset: the problem is taken from `cfg.type`.

  beltrami, ks3d            3D volume rendering with pyvista (off screen), in the
                            style of vape4d (diverging colormap whose opacity
                            grows towards both ends) but without vape4d itself.
                            One GIF per channel: truth | lower | upper |
                            interval width | uncovered voxels.
  challenge2_1, challenge2_2
                            2D fields with matplotlib. One GIF, one column per
                            channel: truth | lower | upper | width | uncovered.
  burgers                   One test viscosity on a dense (x, t) grid: truth,
                            consensus bounds and calibrated band over time, plus
                            an x-t map of the band width with the uncovered
                            points. For d > 1 the solution only depends on
                            sum(x_i), so it is drawn along the diagonal
                            x_i = xi / d.

Exactly one calibration draw is replayed: (alpha, s, repeat) picks a row of
`qp_by_ensemble_size.npz`, and its Q_p is used with the *same* lo/hi members
that produced it (as in the analysis scripts). The band is
  [qbar_lo - Q_p * sigma, qbar_hi + Q_p * sigma].
Field problems use the analysis test snapshots: input obs[2 + 2k], truth
gt[4 + 2k], k < n_steps.

Usage (run from the repository root; on CSF3 use
sbatch_scripts/animations.sbatch):
  python animations.py --config=configs/config.py:sfcqr_beltrami_adamw \
      --batch_n_ens=16 --channel=u
  python animations.py --config=configs/config.py:sfcqr_challenge_adamw \
      --batch_n_ens=16
  python animations.py --config=configs/burgers_config.py:burgers_sfcqr_d3

Output: <out_dir>/<name>.gif (default name: the workdir name, default out_dir:
assets/animations). 3D problems with --channel=all write <name>_<channel>.gif.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import sys

from absl import app, flags
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from PIL import Image

sys.path.append("./")

# Importing calibrate_sfcqr registers --config, --batch_n_ens and
# --predict_chunk_size (the same flags analysis_pool.py reuses). --config takes
# any config file, so Burgers presets work too.
from calibrate_sfcqr import (
    discover_batch_dirs,
    load_split,
    predict_pool,
    restore_pool_variables,
)

FLAGS = flags.FLAGS
flags.DEFINE_float("alpha", None, "Miscoverage level to animate. Default: the "
                   "first entry of cfg.alpha_list that has been calibrated.")
flags.DEFINE_integer("n_ens_sub", None, "Sub-ensemble size s. Default: the "
                     "largest calibrated one.")
flags.DEFINE_integer("repeat", 0, "Which calibration repeat (member subset) to use.")
flags.DEFINE_string("channel", "all", "Field problems: channel name (e.g. u, p, "
                    "ux) or 'all'. 3D problems write one GIF per channel.")
flags.DEFINE_integer("n_steps", None, "Number of frames. Default: "
                     "cfg.num_unroll_steps for field problems, 60 for Burgers.")
flags.DEFINE_integer("fn_idx", None, "Burgers: index of the test viscosity "
                     "(0 .. N_test - 1). Default: the middle one.")
flags.DEFINE_float("lr", None, "Burgers: learning rate of the pool. Default: "
                   "cfg.lr_list[0].")
flags.DEFINE_integer("n_x", 200, "Burgers: number of spatial grid points.")
flags.DEFINE_string("out_dir", "assets/animations", "Output directory.")
flags.DEFINE_string("name", None, "Output file stem. Default: workdir name.")
flags.DEFINE_integer("fps", 4, "Frames per second of the GIF.")
flags.DEFINE_integer("window_size", 420, "3D: pixel size of each pyvista panel.")

QP_FILENAME = "qp_by_ensemble_size.npz"
CHANNELS = {
    "beltrami": ["u", "v", "w", "p"],
    "ks3d": ["u"],
    "challenge2_1": ["ux", "uy"],
    "challenge2_2": ["ux", "uy"],
}
UNCOVERED_CMAP = ListedColormap(["#eeeeee", "#d62728"])


# --------------------------------------------------------------------------
# Picking the calibration draw
# --------------------------------------------------------------------------


def pick_alpha(cfg, alpha_dir_fn):
  """`alpha_dir_fn(alpha)` -> directory that should hold QP_FILENAME."""
  if FLAGS.alpha is not None:
    candidates = [FLAGS.alpha]
  else:
    candidates = list(cfg.alpha_list)
  for a in candidates:
    if os.path.exists(os.path.join(alpha_dir_fn(a), QP_FILENAME)):
      return a
  # Fall back to whatever has been calibrated in the workdir.
  found = sorted(glob.glob(os.path.join(alpha_dir_fn("*"), QP_FILENAME)))
  raise FileNotFoundError(
      f"No {QP_FILENAME} for alpha in {candidates} under "
      f"{os.path.abspath(cfg.workdir)}. Calibrated files found: {found or 'none'}. "
      "Pass --alpha or run the calibration first."
  )


def pick_draw(qp_path):
  """Returns (s, Q_p [channels], lo member idx, hi member idx)."""
  data = np.load(qp_path)
  sizes = sorted(int(k.split("_")[-1]) for k in data.files if not k.endswith("_idx"))
  s = FLAGS.n_ens_sub if FLAGS.n_ens_sub is not None else sizes[-1]
  if s not in sizes:
    raise ValueError(f"s={s} not calibrated in {qp_path}; available: {sizes}")
  qp = np.atleast_1d(data[f"n_ens_{s}"][FLAGS.repeat])
  lo_idx = data[f"n_ens_{s}_lo_idx"][FLAGS.repeat]
  hi_idx = data[f"n_ens_{s}_hi_idx"][FLAGS.repeat]
  return s, qp, lo_idx, hi_idx


def sfcqr_band(reps_lo, reps_hi, qp, delta, variant):
  """Calibrated band from member predictions [s, ...] (same formula as the
  calibration / analysis scripts). Also returns the consensus bounds."""
  qbar_lo, qbar_hi = reps_lo.mean(axis=0), reps_hi.mean(axis=0)
  sigma = np.maximum(qbar_hi - qbar_lo, 0.0) + delta
  if variant == "B":
    sigma = sigma + 0.5 * (reps_lo.std(axis=0) + reps_hi.std(axis=0))
  elif variant != "A":
    raise ValueError(f"Unknown scale variant: {variant}")
  return qbar_lo - qp * sigma, qbar_hi + qp * sigma, qbar_lo, qbar_hi


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def fig_to_array(fig):
  fig.canvas.draw()
  return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_gif(frames, path, fps):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  images = [Image.fromarray(f) for f in frames]
  images[0].save(
      path, save_all=True, append_images=images[1:],
      duration=int(round(1000 / fps)), loop=0, optimize=True,
  )
  print(f"saved -> {path} ({len(frames)} frames, "
        f"{os.path.getsize(path) / 1e6:.1f} MB)")


def out_stem(cfg):
  return FLAGS.name or os.path.basename(os.path.normpath(cfg.workdir))


def add_colorbar(fig, ax, cmap, clim, hidden=False):
  sm = plt.cm.ScalarMappable(norm=Normalize(*clim), cmap=cmap)
  cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
  cb.ax.tick_params(labelsize=8)
  if hidden:  # keeps every panel the same height
    cb.ax.set_visible(False)
  return cb


def field_clim(truth):
  """Colour limits for a signed field: symmetric about 0 if it changes sign."""
  lo, hi = np.percentile(truth, [1, 99])
  if lo < 0 < hi:
    m = max(-lo, hi)
    return (-m, m), True
  return (lo, hi), False


# --------------------------------------------------------------------------
# Field problems (UNet / UViT): Beltrami, KS3D, Challenge
# --------------------------------------------------------------------------


def load_field_data(cfg):
  if cfg.type == "beltrami":
    from data.beltramiDataGeneration import generate_beltrami_da_data
    gt, obs, _, times = generate_beltrami_da_data(
        spatial_res=cfg.res, time_steps=cfg.time_steps, dt=cfg.dt,
        a=1.0, d=1.0, noise_std=cfg.noise_std,
    )
  elif cfg.type == "ks3d":
    from data.ks3dDataGeneration import generate_ks3d_da_data
    gt, obs, times = generate_ks3d_da_data(
        spatial_res=cfg.res, time_steps=cfg.time_steps, noise_std=cfg.noise_std
    )
  elif cfg.type == "challenge2_1":
    from data.challengeDataLoading import load_challenge_da_data
    gt, obs, times = load_challenge_da_data(
        data_dir=cfg.data_dir, time_steps=cfg.time_steps, noise_std=cfg.noise_std
    )
  elif cfg.type == "challenge2_2":
    from data.challengeDataLoading import load_challenge2_2_da_data
    gt, obs, times = load_challenge2_2_da_data(
        data_dir=cfg.data_dir, case=cfg.case, time_steps=cfg.time_steps,
        noise_std=cfg.noise_std,
    )
  else:
    raise ValueError(f"Unknown field type: {cfg.type}")
  return gt, obs, np.asarray(times)


def restore_members(batch_dirs, member_idx, unet, input_shape, tau, cfg):
  """Restores only the batches that hold `member_idx` and stacks those members.

  Pool index i is member i % batch_n_ens of the (i // batch_n_ens)-th batch in
  sorted order -- the same concatenation `restore_pool_variables` does.
  """
  n = FLAGS.batch_n_ens
  needed = sorted({int(i) // n for i in member_idx})
  if needed[-1] >= len(batch_dirs):
    raise ValueError(
        f"Member {max(member_idx)} needs batch #{needed[-1]}, but only "
        f"{len(batch_dirs)} batches exist. Is --batch_n_ens={n} right?"
    )
  per_batch = {
      b: restore_pool_variables([batch_dirs[b]], unet, input_shape, tau, n, cfg)
      for b in needed
  }
  members = [
      jax.tree.map(lambda x, i=int(i): x[i % n], per_batch[int(i) // n])
      for i in member_idx
  ]
  return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *members)


def field_band(cfg):
  from model.unet3d import UNet3d

  workdir_root = os.path.abspath(cfg.workdir)
  gt, obs, times = load_field_data(cfg)
  out_dim = gt.shape[-1]

  alpha = pick_alpha(cfg, lambda a: os.path.join(workdir_root, f"alpha_{a}"))
  experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}")
  s, qp, lo_idx, hi_idx = pick_draw(os.path.join(experiment_dir, QP_FILENAME))
  print(f"alpha={alpha}, s={s}, repeat={FLAGS.repeat}, Q_p={qp}")

  # Same delta as calibration / analysis (training targets only).
  train_idx, _ = load_split(workdir_root)
  y_train = np.asarray(gt[train_idx + 2])
  delta = 1e-6 * np.std(y_train.reshape(-1, out_dim), axis=0)

  # Test snapshots: input obs[2 + 2k] -> truth gt[4 + 2k].
  n_steps = FLAGS.n_steps or cfg.num_unroll_steps
  n_max = (gt.shape[0] - 5) // 2 + 1
  if n_steps > n_max:
    print(f"n_steps={n_steps} > {n_max} available test snapshots; using {n_max}")
    n_steps = n_max
  in_idx = 2 + 2 * np.arange(n_steps)
  inputs = jnp.asarray(obs[in_idx])
  truth = np.asarray(gt[in_idx + 2], dtype=np.float32)

  unet = UNet3d(
      out_channels=out_dim,
      time_cond=True,
      num_channels=cfg.model.num_channels,
      use_spatial_attention=cfg.model.use_spatial_attention,
      downsample_ratio=cfg.model.downsample_ratio,
  )
  input_shape = (cfg.batch_size, *gt.shape[1:])

  reps = {}
  for label, idx, tau in (("lo", lo_idx, alpha / 2), ("hi", hi_idx, 1 - alpha / 2)):
    batch_dirs = discover_batch_dirs(os.path.join(experiment_dir, label))
    variables = restore_members(batch_dirs, idx, unet, input_shape, tau, cfg)
    # predict_pool uses dt = times[2] - times[0], as in calibration/analysis.
    reps[label] = predict_pool(
        unet, variables, inputs, times, member_chunk_size=FLAGS.predict_chunk_size
    ).astype(np.float32)

  lo, hi, _, _ = sfcqr_band(reps["lo"], reps["hi"], qp, delta, cfg.variant)
  return dict(
      truth=truth, lo=lo.astype(np.float32), hi=hi.astype(np.float32),
      times=times[in_idx + 2], alpha=alpha, s=s, qp=qp,
      channels=CHANNELS.get(cfg.type, [f"c{i}" for i in range(out_dim)]),
  )


def selected_channels(names):
  if FLAGS.channel == "all":
    return list(range(len(names)))
  if FLAGS.channel not in names:
    raise ValueError(f"--channel={FLAGS.channel} not in {names}")
  return [names.index(FLAGS.channel)]


# ---- 3D (pyvista) --------------------------------------------------------


def setup_pyvista():
  import pyvista as pv

  pv.OFF_SCREEN = True
  if not os.environ.get("DISPLAY"):
    try:
      pv.start_xvfb()
    except Exception as e:  # EGL / OSMesa VTK builds need no X server
      print(f"pyvista.start_xvfb() failed ({e}); trying without it")
  return pv


def opacity_tf(diverging, peak=0.55, n=64):
  """Opacity transfer function: vape4d-like `diverging_alpha` (transparent in
  the middle, opaque at both ends) or a sequential ramp."""
  x = np.linspace(-1, 1, n) if diverging else np.linspace(0, 1, n)
  return peak * np.abs(x) ** 1.5


class VolumeRenderer:

  def __init__(self, pv, shape, window_size):
    self.pv = pv
    nx, ny, nz = shape
    # Fields live on [-1, 1]^3 (Beltrami); the box is only for display.
    self.grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(2 / (nx - 1), 2 / (ny - 1), 2 / (nz - 1)),
        origin=(-1, -1, -1),
    )
    self.points = np.asarray(self.grid.points)  # Fortran order, x fastest
    self.window_size = window_size

  def _plotter(self):
    pl = self.pv.Plotter(off_screen=True, window_size=(self.window_size,) * 2)
    pl.set_background("white")
    pl.add_mesh(self.grid.outline(), color="black", line_width=1.5)
    return pl

  def _shoot(self, pl):
    pl.camera_position = "iso"
    pl.camera.zoom(1.15)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img

  def volume(self, vol, cmap, clim, diverging):
    grid = self.grid.copy()
    # arrays are indexed [x, y, z] (meshgrid indexing="ij")
    grid.point_data["f"] = np.clip(vol, *clim).ravel(order="F").astype(np.float32)
    pl = self._plotter()
    pl.add_volume(
        grid, scalars="f", cmap=cmap, clim=clim,
        opacity=opacity_tf(diverging), show_scalar_bar=False, shade=False,
    )
    return self._shoot(pl)

  def uncovered(self, covered):
    pl = self._plotter()
    pts = self.points[~covered.ravel(order="F")]
    if len(pts):
      pl.add_points(
          pts, color="#d62728", point_size=5, render_points_as_spheres=True,
          opacity=0.8,
      )
    return self._shoot(pl)


def animate_3d(cfg, band):
  pv = setup_pyvista()
  truth_all, lo_all, hi_all = band["truth"], band["lo"], band["hi"]
  renderer = VolumeRenderer(pv, truth_all.shape[1:4], FLAGS.window_size)

  chans = selected_channels(band["channels"])
  for c in chans:
    name = band["channels"][c]
    truth, lo, hi = truth_all[..., c], lo_all[..., c], hi_all[..., c]
    width = hi - lo
    covered = (truth >= lo) & (truth <= hi)
    clim, diverging = field_clim(truth)
    wlim = (0.0, float(np.percentile(width, 99)))
    field_cmap = "RdBu_r" if diverging else "viridis"

    frames = []
    for k in range(truth.shape[0]):
      print(f"[{name}] frame {k + 1}/{truth.shape[0]}")
      imgs = [
          renderer.volume(truth[k], field_cmap, clim, diverging),
          renderer.volume(lo[k], field_cmap, clim, diverging),
          renderer.volume(hi[k], field_cmap, clim, diverging),
          renderer.volume(width[k], "magma_r", wlim, False),
          renderer.uncovered(covered[k]),
      ]
      cov_k = covered[k].mean()
      titles = [
          f"truth {name}", "lower bound", "upper bound", "interval width",
          f"uncovered ({100 * (1 - cov_k):.1f}%)",
      ]
      fig, axes = plt.subplots(1, 5, figsize=(17, 4.4))
      for i, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if i < 3:
          add_colorbar(fig, ax, field_cmap, clim)
        elif i == 3:
          add_colorbar(fig, ax, "magma_r", wlim)
        else:
          add_colorbar(fig, ax, field_cmap, clim, hidden=True)
      fig.suptitle(
          f"{cfg.type}, channel {name}:  t = {band['times'][k]:.2f}   |   "
          rf"$\alpha$ = {band['alpha']}, $s$ = {band['s']}, "
          rf"$Q_p$ = {band['qp'][c]:.3f}   |   coverage {100 * cov_k:.1f}% "
          rf"(target {100 * (1 - band['alpha']):.0f}%)",
          fontsize=12,
      )
      fig.tight_layout()
      frames.append(fig_to_array(fig))
      plt.close(fig)

    suffix = f"_{name}" if len(chans) > 1 else ""
    write_gif(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}{suffix}.gif"),
              FLAGS.fps)


# ---- 2D (matplotlib) -----------------------------------------------------


def animate_2d(cfg, band):
  chans = selected_channels(band["channels"])
  # (T, X, Y, 1, C) -> (T, Y, X, C) so x is horizontal in imshow.
  def prep(a):
    return np.transpose(a[:, :, :, 0, :], (0, 2, 1, 3))[..., chans]

  truth, lo, hi = prep(band["truth"]), prep(band["lo"]), prep(band["hi"])
  width = hi - lo
  covered = (truth >= lo) & (truth <= hi)
  names = [band["channels"][c] for c in chans]

  styles = []
  for j in range(len(chans)):
    clim, diverging = field_clim(truth[..., j])
    styles.append(dict(
        clim=clim, cmap="RdBu_r" if diverging else "viridis",
        wlim=(0.0, float(np.percentile(width[..., j], 99))),
    ))

  ny, nx = truth.shape[1:3]
  panel_w = 4.6
  panel_h = panel_w * ny / nx + 0.9
  rows = ["truth", "lower bound", "upper bound", "interval width", "uncovered"]
  frames = []
  for k in range(truth.shape[0]):
    fig, axes = plt.subplots(
        len(rows), len(chans), squeeze=False,
        figsize=(panel_w * len(chans) + 0.4, panel_h * len(rows) + 0.6),
    )
    for j, name in enumerate(names):
      st = styles[j]
      panels = [truth[k, ..., j], lo[k, ..., j], hi[k, ..., j], width[k, ..., j],
                ~covered[k, ..., j]]
      for i, (row, img) in enumerate(zip(rows, panels)):
        ax = axes[i, j]
        if i < 3:
          ax.imshow(img, origin="lower", cmap=st["cmap"], vmin=st["clim"][0],
                    vmax=st["clim"][1])
          add_colorbar(fig, ax, st["cmap"], st["clim"])
        elif i == 3:
          ax.imshow(img, origin="lower", cmap="magma_r", vmin=st["wlim"][0],
                    vmax=st["wlim"][1])
          add_colorbar(fig, ax, "magma_r", st["wlim"])
        else:
          ax.imshow(img, origin="lower", cmap=UNCOVERED_CMAP, vmin=0, vmax=1,
                    interpolation="nearest")
          add_colorbar(fig, ax, "magma_r", (0, 1), hidden=True)
          row = f"uncovered ({100 * img.mean():.1f}%)"
        ax.set_title(f"{name}: {row}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cov_k = covered[k].mean()
    fig.suptitle(
        f"{cfg.type}:  t = {band['times'][k]:.3f}\n"
        rf"$\alpha$ = {band['alpha']}, $s$ = {band['s']}, "
        f"coverage {100 * cov_k:.1f}% (target {100 * (1 - band['alpha']):.0f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    frames.append(fig_to_array(fig))
    plt.close(fig)
    print(f"frame {k + 1}/{truth.shape[0]}")

  write_gif(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}.gif"), FLAGS.fps)


# --------------------------------------------------------------------------
# Burgers (DeepONet)
# --------------------------------------------------------------------------


def animate_burgers(cfg):
  from jax import config as jax_config
  jax_config.update("jax_enable_x64", True)

  from data.burgersDataGeneration import burgers_solution, initial_condition
  from utils.burgers_pool_utils import (
      _nus,
      discover_batch_dirs as discover_don_batches,
      generate_function_data,
      load_split as load_don_split,
      make_pool_predictor,
      quantile_dir,
      restore_pool_params,
      sfcqr_delta,
  )

  workdir_root = os.path.abspath(cfg.workdir)
  lr = FLAGS.lr if FLAGS.lr is not None else cfg.lr_list[0]
  alpha = pick_alpha(
      cfg, lambda a: os.path.join(workdir_root, f"alpha_{a}", f"lr_{lr}")
  )
  s, qp, lo_idx, hi_idx = pick_draw(
      os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", QP_FILENAME)
  )
  qp = float(qp[0])
  print(f"alpha={alpha}, lr={lr}, s={s}, repeat={FLAGS.repeat}, Q_p={qp:.4f}")

  _, _, s_all = generate_function_data(cfg)
  idx_train, _ = load_don_split(workdir_root)
  delta = sfcqr_delta(s_all, idx_train)

  # One test viscosity, on a dense (xi, t) grid with x_i = xi / d.
  nus = _nus(cfg, cfg.N_test)
  fn_idx = FLAGS.fn_idx if FLAGS.fn_idx is not None else cfg.N_test // 2
  nu = float(nus[fn_idx])
  d, g = cfg.dim, cfg.grid_data
  n_t = FLAGS.n_steps or 60
  xi = np.linspace(d * g.xmin, d * g.xmax, FLAGS.n_x)
  ts = np.linspace(g.tmin, g.tmax, n_t)
  T, XI = np.meshgrid(ts, xi, indexing="ij")  # (n_t, n_x)
  z = np.concatenate(
      [np.repeat(XI.reshape(-1, 1) / d, d, axis=1), T.reshape(-1, 1)], axis=1
  )
  truth = np.asarray(burgers_solution(z[:, :-1], z[:, -1], nu)).reshape(n_t, -1)
  u0 = np.asarray(initial_condition(cfg, nu))
  u = jnp.float32(np.tile(u0, (len(z), 1)))
  z = jnp.float32(z)

  predictor = make_pool_predictor(cfg)
  reps = {}
  for label, idx in (("lo", lo_idx), ("hi", hi_idx)):
    params = restore_pool_params(
        cfg, discover_don_batches(quantile_dir(workdir_root, alpha, lr, label)), lr
    )
    params = jax.tree.map(lambda x: x[np.asarray(idx)], params)
    reps[label] = np.asarray(predictor(params, u, z), dtype=np.float32)

  lo, hi, qbar_lo, qbar_hi = sfcqr_band(reps["lo"], reps["hi"], qp, delta,
                                        cfg.variant)
  lo, hi, qbar_lo, qbar_hi = (a[..., 0].reshape(n_t, -1)
                              for a in (lo, hi, qbar_lo, qbar_hi))
  width = hi - lo
  covered = (truth >= lo) & (truth <= hi)

  ymin = min(truth.min(), np.percentile(lo, 1))
  ymax = max(truth.max(), np.percentile(hi, 99))
  pad = 0.08 * (ymax - ymin)
  xlabel = "x" if d == 1 else rf"$\xi = \sum_i x_i$  (along $x_i = \xi/{d}$)"

  frames = []
  for k in range(n_t):
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    ax.fill_between(xi, lo[k], hi[k], color="tab:blue", alpha=0.25,
                    label="calibrated band")
    ax.plot(xi, qbar_lo[k], "--", color="tab:blue", lw=1,
            label=r"$\bar q_{lo}$, $\bar q_{hi}$")
    ax.plot(xi, qbar_hi[k], "--", color="tab:blue", lw=1)
    ax.plot(xi, truth[k], "k", lw=2, label="truth")
    miss = ~covered[k]
    if miss.any():
      ax.plot(xi[miss], truth[k][miss], "o", color="#d62728", ms=3,
              label="uncovered")
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(xi[0], xi[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("u")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(rf"t = {ts[k]:.2f},  coverage {100 * covered[k].mean():.1f}%")

    im = ax2.imshow(width, origin="lower", aspect="auto", cmap="magma_r",
                    extent=(xi[0], xi[-1], ts[0], ts[-1]))
    mt, mx = np.nonzero(~covered)
    ax2.scatter(xi[mx], ts[mt], s=1, color="#1f77b4", alpha=0.6, lw=0)
    ax2.axhline(ts[k], color="k", lw=1.2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("t")
    ax2.set_title("band width (dots: uncovered)")
    fig.colorbar(im, ax=ax2, fraction=0.05)

    fig.suptitle(
        rf"Burgers, $d$ = {d}, $\nu$ = {nu:.1f}   |   $\alpha$ = {alpha}, "
        rf"$s$ = {s}, $Q_p$ = {qp:.3f}   |   overall coverage "
        f"{100 * covered.mean():.1f}% (target {100 * (1 - alpha):.0f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    frames.append(fig_to_array(fig))
    plt.close(fig)

  write_gif(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}.gif"), FLAGS.fps)


# --------------------------------------------------------------------------


def main(_):
  cfg = FLAGS.config
  if cfg.type == "burgers":
    animate_burgers(cfg)
    return
  band = field_band(cfg)
  if band["truth"].shape[3] == 1:  # Challenge: singleton z axis
    animate_2d(cfg, band)
  else:
    animate_3d(cfg, band)


if __name__ == "__main__":
  app.run(main)
