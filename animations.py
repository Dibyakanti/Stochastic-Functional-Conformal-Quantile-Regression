"""vape4d-style GIF animations of the SFCQR prediction band, for the README.

Reproduces the look of vape4d (https://github.com/KeKsBoTer/vape4d) with
pyvista/VTK, without calling vape4d:
  - emission-absorption ray marching, no lighting or shading;
  - opacity transfer function `diverging_alpha`, alpha(x) = |2x - 1| over the
    colour range (transparent at mid-range, opaque at both ends), applied per
    box length (`distance_scale` = 1 means alpha is the opacity of one box
    length of constant value, as in vape4d's shader);
  - orthographic camera looking from the (-1, -1, -1) corner at the centre;
  - black background, no bounding box or axes;
  - vmin / vmax from the min / max of the data;
  - linear interpolation in time between snapshots (`--interp` sub-steps).

Every problem is rendered as a 3D volume animated over time, side by side
panels truth | lower bound | upper bound | interval width (`--panels` also
accepts `uncovered`). Truth and both bounds share one colour range (that of
the truth), so the bounds can be compared with the truth directly.

  beltrami, ks3d   the field on its 3D grid, test snapshots t.
  burgers          the solution u(x_1, x_2, x_3, t) on an n^3 grid of
                   [xmin, xmax]^3 for one test viscosity (d > 3: x_4.. = 0;
                   d < 3: constant along the unused axes).
  challenge2_*     2D fields: the same colour / opacity transfer, composited
                   over black (a one-voxel-thick volume seen from above).

Exactly one calibration draw is replayed: (alpha, s, repeat) picks a row of
`qp_by_ensemble_size.npz`, and its Q_p is used with the *same* lo/hi members
that produced it (as in the analysis scripts). The band is
  [qbar_lo - Q_p * sigma, qbar_hi + Q_p * sigma].
Field problems use the analysis test snapshots: input obs[2 + 2k], truth
gt[4 + 2k], k < n_steps.

Usage (from the repository root; on CSF3 use sbatch_scripts/animations.sbatch):
  python animations.py --config=configs/config.py:sfcqr_beltrami_adamw \
      --batch_n_ens=16 --channel=u
  python animations.py --config=configs/config.py:sfcqr_challenge_adamw \
      --batch_n_ens=16
  python animations.py --config=configs/burgers_config.py:burgers_sfcqr_d3

Output: <out_dir>/<name>.gif (default name: the workdir name, default out_dir:
assets/animations). With --channel=all and several channels, 3D problems write
<name>_<channel>.gif.
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
from matplotlib.colors import Normalize
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
flags.DEFINE_string("channel", "all", "Channel name (e.g. u, p, ux) or 'all'.")
flags.DEFINE_string("panels", "truth,lower,upper,width", "Comma-separated "
                    "panels from truth, lower, upper, width, uncovered.")
flags.DEFINE_integer("n_steps", None, "Number of snapshots. Default: "
                     "cfg.num_unroll_steps for field problems, 20 for Burgers.")
flags.DEFINE_integer("interp", 3, "Linearly interpolated frames between two "
                     "snapshots (vape4d's temporal interpolation); 1 = none.")
flags.DEFINE_float("distance_scale", 1.0, "vape4d distance_scale: opacity "
                   "multiplier per box length.")
flags.DEFINE_string("cmap", "RdBu_r", "Colormap for truth / bounds.")
flags.DEFINE_string("width_cmap", "magma", "Colormap for the interval width.")
flags.DEFINE_integer("fn_idx", None, "Burgers: index of the test viscosity "
                     "(0 .. N_test - 1). Default: the middle one.")
flags.DEFINE_float("lr", None, "Burgers: learning rate of the pool. Default: "
                   "cfg.lr_list[0].")
flags.DEFINE_integer("burgers_res", 40, "Burgers: grid points per axis.")
flags.DEFINE_string("out_dir", "assets/animations", "Output directory.")
flags.DEFINE_string("name", None, "Output file stem. Default: workdir name.")
flags.DEFINE_integer("fps", 10, "Frames per second of the GIF.")
flags.DEFINE_integer("window_size", 400, "Pixel size of each 3D panel.")

QP_FILENAME = "qp_by_ensemble_size.npz"
CHANNELS = {
    "beltrami": ["u", "v", "w", "p"],
    "ks3d": ["u"],
    "challenge2_1": ["ux", "uy"],
    "challenge2_2": ["ux", "uy"],
    "burgers": ["u"],
}
PANEL_TITLES = {
    "truth": "truth", "lower": "lower bound", "upper": "upper bound",
    "width": "interval width", "uncovered": "uncovered",
}
N_TF = 256  # entries of the colour / opacity lookup, as in vape4d


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
      label=cfg.type,
  )


def selected_channels(names):
  if FLAGS.channel == "all":
    return list(range(len(names)))
  if FLAGS.channel not in names:
    raise ValueError(f"--channel={FLAGS.channel} not in {names}")
  return [names.index(FLAGS.channel)]



def selected_panels():
  panels = [p.strip() for p in FLAGS.panels.split(",") if p.strip()]
  bad = [p for p in panels if p not in PANEL_TITLES]
  if bad:
    raise ValueError(f"Unknown panels {bad}; choose from {list(PANEL_TITLES)}")
  return panels


# --------------------------------------------------------------------------
# vape4d-style transfer functions and time interpolation
# --------------------------------------------------------------------------


def diverging_alpha(n=N_TF):
  """vape4d.utils.diverging_alpha: alpha(x) = |2x - 1|."""
  return np.abs(2 * np.arange(n) / n - 1)


def linear_increasing_alpha(n=N_TF):
  """vape4d.utils.linear_increasing_alpha: alpha(x) = x."""
  return np.arange(n) / (n - 1)


def interp_time(a, sub):
  """Linear interpolation along axis 0 with `sub` frames per interval
  (vape4d's temporal_interpolation='linear')."""
  if sub <= 1 or a.shape[0] < 2:
    return a
  s = np.linspace(0, a.shape[0] - 1, (a.shape[0] - 1) * sub + 1)
  i0 = np.minimum(np.floor(s).astype(int), a.shape[0] - 2)
  w = (s - i0).reshape(-1, *([1] * (a.ndim - 1)))
  return (1 - w) * a[i0] + w * a[i0 + 1]


def panel_volumes(truth, lo, hi):
  """Per-panel arrays [T, ...] and their (cmap, clim, alpha) styles."""
  width = hi - lo
  uncovered = ((truth < lo) | (truth > hi)).astype(np.float32)
  vmin, vmax = float(truth.min()), float(truth.max())
  if vmax <= vmin:
    vmax = vmin + 1e-6
  field = (FLAGS.cmap, (vmin, vmax), diverging_alpha())
  vols = {
      "truth": (truth, field),
      "lower": (lo, field),
      "upper": (hi, field),
      "width": (width, (FLAGS.width_cmap, (0.0, float(max(width.max(), 1e-12))),
                        linear_increasing_alpha())),
      "uncovered": (uncovered, ("Reds", (0.0, 1.0), linear_increasing_alpha())),
  }
  return {p: vols[p] for p in selected_panels()}


# --------------------------------------------------------------------------
# Frame composition (black background, white text, slim colour bars)
# --------------------------------------------------------------------------


def compose_frame(images, titles, styles, header, panel_inches=3.2):
  n = len(images)
  h, w = images[0].shape[:2]
  fig, axes = plt.subplots(
      1, n, figsize=(panel_inches * n, panel_inches * h / w + 0.9),
      facecolor="black", squeeze=False,
  )
  for ax, img, title, (cmap, clim, _) in zip(axes[0], images, titles, styles):
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.outline.set_edgecolor("white")
    cb.ax.tick_params(colors="white", labelsize=7)
  fig.suptitle(header, color="white", fontsize=10)
  fig.tight_layout()
  fig.canvas.draw()
  frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  plt.close(fig)
  return frame


def frame_header(band, name, k_key, t, coverage):
  return (
      f"{band['label']}  [{name}]   t = {t:.3f}   |   "
      rf"$\alpha$ = {band['alpha']}, $s$ = {band['s']}, "
      rf"$Q_p$ = {band['qp'][band['channels'].index(name)]:.3g}   |   "
      f"coverage {100 * coverage[k_key]:.1f}% "
      f"(target {100 * (1 - band['alpha']):.0f}%)"
  )


# --------------------------------------------------------------------------
# 3D: pyvista volume rendering, vape4d look
# --------------------------------------------------------------------------


def start_xvfb():
  """Starts a private Xvfb server and points DISPLAY at it. (pyvista >= 0.45
  dropped `pyvista.start_xvfb`; without an X server the VTK build in the
  container renders blank images.)"""
  import atexit
  import shutil
  import subprocess
  import time

  if not shutil.which("Xvfb"):
    print("Xvfb not found; rendering without an X server")
    return
  display = f":{100 + os.getpid() % 800}"
  proc = subprocess.Popen(
      ["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
  )
  atexit.register(proc.terminate)
  time.sleep(2)
  if proc.poll() is not None:
    raise RuntimeError(f"Xvfb {display} exited with code {proc.returncode}")
  os.environ["DISPLAY"] = display
  print(f"started Xvfb on {display}")


def setup_pyvista():
  if not os.environ.get("DISPLAY"):
    start_xvfb()
  import pyvista as pv

  pv.OFF_SCREEN = True
  return pv


class VapeRenderer:
  """One square panel per call: volume in a unit-ish box, no box drawn."""

  def __init__(self, pv, shape, window_size):
    self.pv = pv
    n = max(shape)
    h = 1.0 / (n - 1)  # longest side has length 1
    self.grid = pv.ImageData(dimensions=shape, spacing=(h, h, h), origin=(0, 0, 0))
    self.box_len = h * (n - 1)
    self.window_size = window_size

  def render(self, vol, cmap, clim, alpha):
    pv = self.pv
    grid = self.grid.copy()
    # arrays are indexed [x, y, z] (meshgrid indexing="ij") -> Fortran order
    grid.point_data["f"] = np.clip(vol, *clim).ravel(order="F").astype(np.float32)

    pl = pv.Plotter(off_screen=True, window_size=(self.window_size,) * 2)
    pl.set_background("black")
    actor = pl.add_volume(
        grid, scalars="f", cmap=cmap, clim=clim, opacity=alpha,
        n_colors=N_TF, shade=False, show_scalar_bar=False,
        opacity_unit_distance=self.box_len / FLAGS.distance_scale,
        blending="composite",
    )
    try:  # trilinear sampling between voxels, as vape4d's "linear"
      actor.prop.interpolation_type = "linear"
    except AttributeError:
      actor.GetProperty().SetInterpolationTypeToLinear()

    # vape4d Camera::new_aabb_iso: orthographic, from the (-1,-1,-1) corner.
    c = np.array(grid.center)
    r = 0.5 * grid.length  # bounding-sphere radius (half the diagonal)
    pl.enable_parallel_projection()
    pl.camera.focal_point = c
    pl.camera.position = c - np.ones(3) / np.sqrt(3) * r * 2.8
    pl.camera.up = (0, 0, 1)
    pl.camera.parallel_scale = r
    pl.camera.clipping_range = (0.01, 1000)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def animate_3d(cfg, band):
  pv = setup_pyvista()
  chans = selected_channels(band["channels"])
  renderer = VapeRenderer(pv, band["truth"].shape[1:4], FLAGS.window_size)
  sub = max(FLAGS.interp, 1)
  times = interp_time(band["times"], sub)

  for c in chans:
    name = band["channels"][c]
    truth, lo, hi = (band[k][..., c] for k in ("truth", "lo", "hi"))
    coverage = ((truth >= lo) & (truth <= hi)).mean(axis=(1, 2, 3))
    vols = panel_volumes(truth, lo, hi)
    vols = {p: (interp_time(v, sub), st) for p, (v, st) in vols.items()}

    frames = []
    for k in range(len(times)):
      print(f"[{name}] frame {k + 1}/{len(times)}", flush=True)
      images = [renderer.render(v[k], *st) for v, st in vols.values()]
      if k == 0 and all(img[..., :3].max() == 0 for img in images):
        raise RuntimeError(
            "Every panel rendered pure black: VTK has no working display. "
            "Run with Xvfb available (it is in the container image)."
        )
      header = frame_header(band, name, int(round(k / sub)), times[k], coverage)
      frames.append(compose_frame(
          images, [PANEL_TITLES[p] for p in vols], [st for _, st in vols.values()],
          header,
      ))

    suffix = f"_{name}" if len(chans) > 1 else ""
    write_gif(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}{suffix}.gif"),
              FLAGS.fps)


# --------------------------------------------------------------------------
# 2D (Challenge): the same transfer function composited over black
# --------------------------------------------------------------------------


def vape_image_2d(img, cmap, clim, alpha):
  """Colour + opacity lookup, emitted over a black background (premultiplied
  colour, as vape4d's shader outputs). `img` is [X, Y]; returns [Y, X, 3]."""
  x = np.clip((img - clim[0]) / (clim[1] - clim[0]), 0, 1)
  idx = np.minimum((x * N_TF).astype(int), N_TF - 1)
  rgb = plt.get_cmap(cmap)(np.linspace(0, 1, N_TF))[:, :3][idx]
  a = np.clip(alpha[idx] * FLAGS.distance_scale, 0, 1)[..., None]
  return np.transpose(rgb * a, (1, 0, 2))[::-1]  # x right, y up


def animate_2d(cfg, band):
  chans = selected_channels(band["channels"])
  sub = max(FLAGS.interp, 1)
  times = interp_time(band["times"], sub)

  per_chan = []
  for c in chans:
    truth, lo, hi = (band[k][:, :, :, 0, c] for k in ("truth", "lo", "hi"))
    coverage = ((truth >= lo) & (truth <= hi)).mean(axis=(1, 2))
    vols = panel_volumes(truth, lo, hi)
    per_chan.append((band["channels"][c], coverage,
                     {p: (interp_time(v, sub), st) for p, (v, st) in vols.items()}))

  frames = []
  for k in range(len(times)):
    rows = []
    for name, coverage, vols in per_chan:
      images = [vape_image_2d(v[k], *st) for v, st in vols.values()]
      header = frame_header(band, name, int(round(k / sub)), times[k], coverage)
      rows.append(compose_frame(
          images, [PANEL_TITLES[p] for p in vols], [st for _, st in vols.values()],
          header, panel_inches=3.6,
      ))
    frames.append(np.concatenate(rows, axis=0))
    print(f"frame {k + 1}/{len(times)}", flush=True)

  suffix = "" if FLAGS.channel == "all" else f"_{FLAGS.channel}"
  write_gif(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}{suffix}.gif"),
            FLAGS.fps)


# --------------------------------------------------------------------------
# Burgers (DeepONet): band on a 3D grid, same layout as the field problems
# --------------------------------------------------------------------------


def burgers_band(cfg):
  from jax import config as jax_config
  jax_config.update("jax_enable_x64", True)

  from data.burgersDataGeneration import burgers_solution, initial_condition
  from utils.burgers_pool_utils import (
      _nus,
      discover_batch_dirs as discover_don_batches,
      generate_function_data,
      load_split as load_don_split,
      make_pool_predictor,
      predict_pool as predict_don_pool,
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
  print(f"alpha={alpha}, lr={lr}, s={s}, repeat={FLAGS.repeat}, Q_p={qp}")

  _, _, s_all = generate_function_data(cfg)
  idx_train, _ = load_don_split(workdir_root)
  delta = sfcqr_delta(s_all, idx_train)

  nus = _nus(cfg, cfg.N_test)
  fn_idx = FLAGS.fn_idx if FLAGS.fn_idx is not None else cfg.N_test // 2
  nu = float(nus[fn_idx])
  d, g, n = cfg.dim, cfg.grid_data, FLAGS.burgers_res
  n_t = FLAGS.n_steps or 20
  ts = np.linspace(g.tmin, g.tmax, n_t)

  # Spatial points: (x_1, x_2, x_3) on an n^3 grid; x_4.. = 0; for d < 3 the
  # unused grid axes are ignored (the field is constant along them).
  ax = np.linspace(g.xmin, g.xmax, n)
  X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
  x = np.zeros((len(X), d))
  x[:, :min(d, 3)] = X[:, :min(d, 3)]

  u = jnp.float32(np.tile(np.asarray(initial_condition(cfg, nu)), (len(x), 1)))
  predictor = make_pool_predictor(cfg)
  params = {}
  for label, idx in (("lo", lo_idx), ("hi", hi_idx)):
    p = restore_pool_params(
        cfg, discover_don_batches(quantile_dir(workdir_root, alpha, lr, label)), lr
    )
    params[label] = jax.tree.map(lambda a, idx=np.asarray(idx): a[idx], p)

  truth, lo, hi = [], [], []
  for k, t in enumerate(ts):
    print(f"Burgers: predicting t = {t:.2f} ({k + 1}/{n_t})", flush=True)
    z = jnp.float32(np.concatenate([x, np.full((len(x), 1), t)], axis=1))
    reps = {
        label: predict_don_pool(predictor, params[label], u, z,
                                FLAGS.predict_chunk_size).astype(np.float32)
        for label in ("lo", "hi")
    }
    l, h, _, _ = sfcqr_band(reps["lo"], reps["hi"], qp, delta, cfg.variant)
    lo.append(l.reshape(n, n, n, 1))
    hi.append(h.reshape(n, n, n, 1))
    truth.append(np.asarray(burgers_solution(x, t, nu)).reshape(n, n, n, 1))

  return dict(
      truth=np.stack(truth).astype(np.float32),
      lo=np.stack(lo).astype(np.float32), hi=np.stack(hi).astype(np.float32),
      times=ts, alpha=alpha, s=s, qp=qp, channels=["u"],
      label=f"Burgers d={d}, nu={nu:.0f}",
  )


# --------------------------------------------------------------------------


def main(_):
  cfg = FLAGS.config
  band = burgers_band(cfg) if cfg.type == "burgers" else field_band(cfg)
  if band["truth"].shape[3] == 1:  # Challenge: singleton z axis
    animate_2d(cfg, band)
  else:
    animate_3d(cfg, band)


if __name__ == "__main__":
  app.run(main)
