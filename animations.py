"""vape4d-style animations (WebP / GIF) of the SFCQR prediction band.

Reproduces the look of vape4d (https://github.com/KeKsBoTer/vape4d) with
pyvista/VTK, without calling vape4d:
  - emission-absorption ray marching, no lighting or shading;
  - opacity transfer function `diverging_alpha`, alpha(x) = |2x - 1| over the
    colour range (transparent at mid-range, opaque at both ends), applied per
    box length (`distance_scale` = 1 means alpha is the opacity of one box
    length of constant value, as in vape4d's shader);
  - orthographic camera looking from the (-1, -1, -1) corner at the centre;
  - plain background (dark grey by default, --background=black for vape4d),
    no bounding box or axes;
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
                   over the background (a one-voxel-thick volume seen from above).

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

Smoothness: 3D panels are ray-marched with a fixed step of 1/4 voxel, rendered
at 2x and Lanczos-downsampled; 2D fields are upsampled bilinearly; frames are
linearly interpolated in time. The colour range is fixed for the whole
animation (--clim=frame re-fits it per frame).

Output: <out_dir>/<name>.webp and <name>.gif (--format; default name: the
workdir name, default out_dir: assets/animations). WebP is full colour and
renders inline on GitHub like a GIF. With --channel=all and several channels,
3D problems write <name>_<channel>.*.
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
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
flags.DEFINE_string("panels", "truth,lower,upper,covered", "Comma-separated "
                    "panels from truth, lower, upper, covered, width, uncovered.")
flags.DEFINE_integer("n_steps", None, "Number of snapshots. Default: "
                     "cfg.num_unroll_steps for field problems, 20 for Burgers.")
flags.DEFINE_integer("interp", None, "Frames per snapshot interval, linearly "
                     "interpolated (vape4d's temporal interpolation); 1 = none. "
                     "Default: 6 for 3D problems, 1 for 2D (Challenge snapshots "
                     "are already dense in time).")
flags.DEFINE_float("distance_scale", 2.0, "vape4d distance_scale: opacity "
                   "multiplier per box length.")
flags.DEFINE_string("cmap", "icefire_bright", "Colormap for truth / bounds "
                    "('icefire_bright' or any matplotlib name).")
flags.DEFINE_float("alpha_gamma", 0.5, "Opacity |2x - 1|^gamma: 1 = vape4d's "
                   "diverging_alpha; < 1 keeps small values visible.")
flags.DEFINE_enum("scale", "linear", ["linear", "log"], "Truth / bound "
                  "panels: the values, or a signed log sign(x) * log10(|x| / "
                  "floor), which keeps the sign and keeps decaying fields "
                  "visible on one fixed scale.")
flags.DEFINE_float("log_decades", 3.0, "--scale=log: decades shown; "
                   "magnitudes below the floor (this many decades under the "
                   "top of the range) map to 0, i.e. transparent.")
flags.DEFINE_float("clim_percentile", 99.5, "Colour range of truth / bounds: "
                   "this percentile of |truth| (100 = max).")
flags.DEFINE_enum("clim", "global", ["global", "frame"], "Colour range of "
                  "truth / bounds: fixed over the whole animation (vape4d), or "
                  "re-fitted to the truth at every frame.")
flags.DEFINE_string("width_cmap", "magma", "Colormap for the interval width.")
flags.DEFINE_integer("fn_idx", None, "Burgers: index of the test viscosity "
                     "(0 .. N_test - 1). Default: the middle one.")
flags.DEFINE_float("lr", None, "Burgers: learning rate of the pool. Default: "
                   "cfg.lr_list[0].")
flags.DEFINE_integer("burgers_res", 40, "Burgers: grid points per axis.")
flags.DEFINE_string("out_dir", "assets/animations", "Output directory.")
flags.DEFINE_string("name", None, "Output file stem. Default: workdir name.")
flags.DEFINE_integer("fps", 20, "Frames per second.")
flags.DEFINE_string("background", "#2a2c33", "Background colour (vape4d uses "
                    "black; a dark grey reads better on a web page).")
flags.DEFINE_integer("window_size", 480, "Output pixel size of each 3D panel.")
flags.DEFINE_bool("combine_channels", True, "3D with several channels: one "
                  "video with a row per channel (else one video per channel).")
flags.DEFINE_bool("axes", True, "3D: draw labelled x / y / z arrows at the "
                  "lower-left of the truth panels.")
flags.DEFINE_integer("supersample", 2, "3D: render at this multiple of "
                     "window_size and downsample (anti-aliasing).")
flags.DEFINE_float("samples_per_voxel", 4.0, "3D: ray-marching samples per "
                   "voxel spacing. Higher removes slicing artefacts.")
flags.DEFINE_integer("upsample_2d", 4, "2D: bilinear upsampling factor of the "
                     "field before colour mapping.")
flags.DEFINE_string("format", "webp,gif", "Comma-separated output formats: "
                    "webp (full colour, smooth; vape4d's README uses it) and/or "
                    "gif (256 colours, dithered).")
flags.DEFINE_integer("gif_max_width", 960, "Downscale GIF frames wider than "
                     "this (GIF compresses poorly; the README shows 600 px).")

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
    "covered": "covered points", "width": "interval width",
    "uncovered": "uncovered",
}
N_TF = 256  # entries of the colour / opacity lookup, as in vape4d
# Covered panel: 0 = missed (red, opaque), 1 = covered (green, faint), so the
# misses stand out inside the mostly-covered cube.
COVER_CMAP = ListedColormap(["#ff3030", "#3ddc84"], name="coverage")
COVER_ALPHA_MISSED, COVER_ALPHA_COVERED = 0.7, 0.15
# Diverging map for emission on black: bright cyan / amber just off the middle
# (the middle itself is transparent), deep blue / red at the ends.
ICEFIRE_BRIGHT = LinearSegmentedColormap.from_list("icefire_bright", [
    (0.0, "#3a5bff"), (0.3, "#2fa8ff"), (0.47, "#5ef0ff"), (0.5, "#ffffff"),
    (0.53, "#ffd23f"), (0.7, "#ff8c1a"), (1.0, "#ff2e4c"),
])


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


def _gif_frames(frames):
  """Quantises every frame with ONE palette (built from a sample of frames)
  and Floyd-Steinberg dithering: no palette flicker between frames and no
  colour banding in smooth gradients / colour bars."""
  h, w = frames[0].shape[:2]
  if w > FLAGS.gif_max_width:
    size = (FLAGS.gif_max_width, round(h * FLAGS.gif_max_width / w))
    frames = [np.asarray(Image.fromarray(f).resize(size, Image.Resampling.LANCZOS))
              for f in frames]
  step = max(1, len(frames) // 8)
  sample = np.concatenate([f[::2, ::2] for f in frames[::step]], axis=0)
  palette = Image.fromarray(sample).quantize(
      colors=256, method=Image.Quantize.MEDIANCUT
  )
  return [
      Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
      for f in frames
  ]


def write_animation(frames, stem, fps):
  """Writes `<stem>.<fmt>` for every format in --format."""
  from PIL import features

  os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
  duration = int(round(1000 / fps))
  for fmt in [f.strip() for f in FLAGS.format.split(",") if f.strip()]:
    path = f"{stem}.{fmt}"
    if fmt == "webp":
      if not features.check("webp"):
        print("Pillow has no animated WebP support; skipping .webp")
        continue
      images = [Image.fromarray(f) for f in frames]
      images[0].save(path, save_all=True, append_images=images[1:],
                     duration=duration, loop=0, quality=92, method=4)
    elif fmt == "gif":
      images = _gif_frames(frames)
      images[0].save(path, save_all=True, append_images=images[1:],
                     duration=duration, loop=0, optimize=False, disposal=1)
    else:
      raise ValueError(f"Unknown format {fmt!r}; use webp and/or gif")
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
        domain=cfg.get("domain", 1.0),
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
      kernel_size=tuple(cfg.model.get("kernel_size", (3, 3, 3))),
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


def diverging_alpha(n=N_TF, gamma=None):
  """vape4d.utils.diverging_alpha, alpha(x) = |2x - 1|, raised to
  --alpha_gamma (1 reproduces vape4d)."""
  gamma = FLAGS.alpha_gamma if gamma is None else gamma
  return np.abs(2 * np.arange(n) / n - 1) ** gamma


def get_cmap(name):
  if name == "icefire_bright":
    return ICEFIRE_BRIGHT
  return plt.get_cmap(name)


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


def field_clims(truth):
  """[T, 2] colour ranges from the truth. Symmetric about 0 when the field
  changes sign, so 0 sits at the transparent middle of diverging_alpha."""
  axes = tuple(range(1, truth.ndim))
  q = FLAGS.clim_percentile
  if FLAGS.clim == "frame":
    vmin = np.percentile(truth, 100 - q, axis=axes)
    vmax = np.percentile(truth, q, axis=axes)
  else:
    vmin = np.full(len(truth), np.percentile(truth, 100 - q))
    vmax = np.full(len(truth), np.percentile(truth, q))
  signed = (vmin < 0) & (vmax > 0)
  m = np.maximum(-vmin, vmax)
  vmin, vmax = np.where(signed, -m, vmin), np.where(signed, m, vmax)
  vmax = np.maximum(vmax, vmin + 1e-6)
  return np.stack([vmin, vmax], axis=1)


def panel_volumes(truth, lo, hi, sub):
  """Per-panel arrays [T', ...] (time-interpolated with `sub`) and their
  (cmap, clims [T', 2], alpha) styles."""
  truth, lo, hi = (interp_time(a, sub) for a in (truth, lo, hi))
  n = len(truth)
  width = hi - lo
  covered = ((truth >= lo) & (truth <= hi)).astype(np.float32)
  uncovered = 1.0 - covered
  if FLAGS.scale == "log":
    # Signed log: sign(x) * log10(|x| / floor), 0 below the floor. Coverage
    # above is still computed from the values themselves.
    top = float(np.log10(np.percentile(np.abs(truth), FLAGS.clim_percentile)))
    floor = top - FLAGS.log_decades
    truth, lo, hi = (signed_log(a, floor) for a in (truth, lo, hi))
    cmap = get_cmap(FLAGS.cmap)
    SYMLOG.update(cmap=cmap, floor=floor, top=top)
    d = FLAGS.log_decades
    field = (cmap, np.tile([-d, d], (n, 1)), diverging_alpha())
  else:
    field = (get_cmap(FLAGS.cmap), field_clims(truth), diverging_alpha())
  wmax = float(max(width.max(), 1e-12))
  vols = {
      "truth": (truth, field),
      "lower": (lo, field),
      "upper": (hi, field),
      "width": (width, (FLAGS.width_cmap, np.tile([0.0, wmax], (n, 1)),
                        linear_increasing_alpha())),
      "covered": (covered, (COVER_CMAP, np.tile([0.0, 1.0], (n, 1)),
                            np.linspace(COVER_ALPHA_MISSED, COVER_ALPHA_COVERED,
                                        N_TF))),
      "uncovered": (uncovered, ("Reds", np.tile([0.0, 1.0], (n, 1)),
                                linear_increasing_alpha())),
  }
  return {p: vols[p] for p in selected_panels()}


SYMLOG = {}  # set by panel_volumes for --scale=log (colour-bar labels)


def signed_log(a, floor):
  return np.sign(a) * np.maximum(np.log10(np.maximum(np.abs(a), 1e-300)) - floor, 0.0)


def symlog_ticks():
  """Colour-bar ticks at +-10^k (in the signed-log coordinate) and 0."""
  floor, top = SYMLOG["floor"], SYMLOG["top"]
  ks = range(int(np.ceil(floor)), int(np.floor(top)) + 1)
  pos, lab = [0.0], ["0"]
  for k in ks:
    for sgn in (-1, 1):
      pos.append(sgn * (k - floor))
      lab.append(f"{sgn * 10.0 ** k:g}")
  order = np.argsort(pos)
  return [pos[i] for i in order], [lab[i] for i in order]


def panel_titles(vols):
  log_panels = ("truth", "lower", "upper")
  return [
      f"{PANEL_TITLES[p]} (signed log)"
      if FLAGS.scale == "log" and p in log_panels else PANEL_TITLES[p]
      for p in vols
  ]


def style_at(style, k):
  cmap, clims, alpha = style
  return cmap, tuple(float(c) for c in clims[k]), alpha


# --------------------------------------------------------------------------
# Frame composition (dark background, white text, slim colour bars)
# --------------------------------------------------------------------------


def compose_frame(images, titles, styles, header, panel_inches=3.2):
  n = len(images)
  h, w = images[0].shape[:2]
  # dpi chosen so each rendered panel is shown at (about) 1:1 pixels.
  fig, axes = plt.subplots(
      1, n, figsize=(panel_inches * n, panel_inches * h / w + 0.9),
      facecolor=FLAGS.background, squeeze=False, dpi=w / panel_inches,
  )
  for ax, img, title, (cmap, clim, _) in zip(axes[0], images, titles, styles):
    ax.imshow(img, interpolation="antialiased")
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.outline.set_edgecolor("white")
    cb.ax.tick_params(colors="white", labelsize=7)
    if cmap is COVER_CMAP:
      cb.set_ticks([0.25, 0.75], labels=["missed", "covered"])
    elif FLAGS.scale == "log" and cmap is SYMLOG.get("cmap"):
      pos, lab = symlog_ticks()
      cb.set_ticks(pos, labels=lab)
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
    self.voxel = h
    self.window_size = window_size

  def _add_axes(self, pl):
    """Small labelled x / y / z triad in the lower-left of the panel, outside
    the box's outline (offset sideways to the view direction, so it never
    sits on top of the volume)."""
    pv = self.pv
    c = np.array(self.grid.center, dtype=float)
    r = 0.5 * self.grid.length
    view = np.ones(3) / np.sqrt(3)                 # camera looks along +(1,1,1)
    right = np.cross(view, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, view)
    o = c - 0.72 * r * right - 0.66 * r * up
    L = 0.17 * self.box_len
    for d, (lab, col) in enumerate((("x", "#ff6b6b"), ("y", "#69db7c"),
                                    ("z", "#74c0fc"))):
      e = np.zeros(3); e[d] = 1.0
      pl.add_mesh(pv.Arrow(start=o, direction=e, scale=L, tip_length=0.25,
                           tip_radius=0.08, shaft_radius=0.025), color=col)
      pl.add_point_labels(
          [o + e * L * 1.45], [lab], text_color=col, font_size=14 * max(FLAGS.supersample, 1),
          bold=True, show_points=False, shape=None, always_visible=True,
      )

  def render(self, vol, cmap, clim, alpha, axes=False):
    pv = self.pv
    grid = self.grid.copy()
    # arrays are indexed [x, y, z] (meshgrid indexing="ij") -> Fortran order
    grid.point_data["f"] = np.clip(vol, *clim).ravel(order="F").astype(np.float32)

    ss = max(FLAGS.supersample, 1)
    pl = pv.Plotter(off_screen=True, window_size=(self.window_size * ss,) * 2)
    pl.set_background(FLAGS.background)
    actor = pl.add_volume(
        grid, scalars="f", cmap="gray", clim=clim, n_colors=N_TF, shade=False,
        show_scalar_bar=False, blending="composite",
    )
    # Opacity transfer function written straight into VTK: passing the array
    # as `opacity=` to pyvista 0.48 renders a blank image.
    prop = actor.GetProperty()
    # Colour transfer function written straight into VTK too, so any
    # matplotlib colormap (incl. ListedColormap objects) works.
    values = np.linspace(clim[0], clim[1], N_TF)
    rgb = get_cmap(cmap)(np.linspace(0, 1, N_TF))[:, :3] if isinstance(cmap, str) \
        else cmap(np.linspace(0, 1, N_TF))[:, :3]
    ctf = prop.GetRGBTransferFunction(0)
    ctf.RemoveAllPoints()
    for v, (r, g, b) in zip(values, rgb):
      ctf.AddRGBPoint(float(v), float(r), float(g), float(b))
    otf = prop.GetScalarOpacity()
    otf.RemoveAllPoints()
    for v, a in zip(np.linspace(clim[0], clim[1], len(alpha)), alpha):
      otf.AddPoint(float(v), float(a))
    # alpha = opacity of one box length (vape4d's distance_scale convention)
    prop.SetScalarOpacityUnitDistance(self.box_len / FLAGS.distance_scale)
    prop.SetInterpolationTypeToLinear()  # vape4d spatial_interpolation="linear"
    # Fixed, fine ray step (VTK otherwise picks a coarse one, which shows up
    # as stacked slices). Opacity is per unit distance, so it is unaffected.
    mapper = actor.GetMapper()
    for name, arg in (("SetAutoAdjustSampleDistances", 0),
                      ("SetInteractiveAdjustSampleDistances", 0),
                      ("SetSampleDistance", self.voxel / FLAGS.samples_per_voxel)):
      if hasattr(mapper, name):
        getattr(mapper, name)(arg)

    if axes:
      self._add_axes(pl)

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
    if ss > 1:  # anti-aliasing: Lanczos downsampling of the large render
      img = np.asarray(Image.fromarray(img[..., :3]).resize(
          (self.window_size,) * 2, Image.Resampling.LANCZOS))
    return img


def animate_3d(cfg, band):
  pv = setup_pyvista()
  chans = selected_channels(band["channels"])
  renderer = VapeRenderer(pv, band["truth"].shape[1:4], FLAGS.window_size)
  sub = max(FLAGS.interp if FLAGS.interp is not None else 6, 1)
  times = interp_time(band["times"], sub)

  per_chan = []
  for c in chans:
    name = band["channels"][c]
    truth, lo, hi = (band[k][..., c] for k in ("truth", "lo", "hi"))
    coverage = ((truth >= lo) & (truth <= hi)).mean(axis=(1, 2, 3))
    per_chan.append((name, coverage, panel_volumes(truth, lo, hi, sub)))

  def row(name, coverage, vols, k):
    styles = [style_at(st, k) for _, st in vols.values()]
    images = [
        renderer.render(v[k], *st, axes=FLAGS.axes and p == "truth")
        for (p, (v, _)), st in zip(vols.items(), styles)
    ]
    if k == 0 and all(img[..., :3].std() == 0 for img in images):
      raise RuntimeError(
          "Every panel rendered as a flat background: VTK has no working "
          "display. Run with Xvfb available (it is in the container image)."
      )
    header = frame_header(band, name, int(round(k / sub)), times[k], coverage)
    return compose_frame(images, panel_titles(vols), styles, header)

  if FLAGS.combine_channels and len(per_chan) > 1:
    groups = [per_chan]                # one video, a row per channel
  else:
    groups = [[pc] for pc in per_chan]  # one video per channel
  for group in groups:
    frames = []
    for k in range(len(times)):
      print(f"[{','.join(g[0] for g in group)}] frame {k + 1}/{len(times)}",
            flush=True)
      frames.append(np.concatenate([row(*g, k) for g in group], axis=0))
    suffix = f"_{group[0][0]}" if len(groups) > 1 else ""
    write_animation(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}{suffix}"),
                    FLAGS.fps)


# --------------------------------------------------------------------------
# 2D (Challenge): the same transfer function composited over the background
# --------------------------------------------------------------------------


def vape_image_2d(img, cmap, clim, alpha):
  """Colour + opacity lookup, emitted over a black background (premultiplied
  colour, as vape4d's shader outputs). `img` is [X, Y]; returns [Y, X, 3].
  The field is upsampled bilinearly first (vape4d samples the data with
  linear interpolation), so the colours are smooth rather than blocky."""
  if FLAGS.upsample_2d > 1:
    from scipy.ndimage import zoom
    img = zoom(img, FLAGS.upsample_2d, order=1)
  x = np.clip((img - clim[0]) / (clim[1] - clim[0]), 0, 1)
  idx = np.minimum((x * N_TF).astype(int), N_TF - 1)
  rgb = plt.get_cmap(cmap)(np.linspace(0, 1, N_TF))[:, :3][idx]
  a = np.clip(alpha[idx] * FLAGS.distance_scale, 0, 1)[..., None]
  bg = np.asarray(matplotlib.colors.to_rgb(FLAGS.background))
  out = rgb * a + bg * (1 - a)  # emission over the background
  out = np.transpose(out, (1, 0, 2))[::-1]  # x right, y up
  return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def animate_2d(cfg, band):
  chans = selected_channels(band["channels"])
  sub = max(FLAGS.interp if FLAGS.interp is not None else 1, 1)
  times = interp_time(band["times"], sub)

  per_chan = []
  for c in chans:
    truth, lo, hi = (band[k][:, :, :, 0, c] for k in ("truth", "lo", "hi"))
    coverage = ((truth >= lo) & (truth <= hi)).mean(axis=(1, 2))
    per_chan.append((band["channels"][c], coverage,
                     panel_volumes(truth, lo, hi, sub)))

  frames = []
  for k in range(len(times)):
    rows = []
    for name, coverage, vols in per_chan:
      styles = [style_at(st, k) for _, st in vols.values()]
      images = [vape_image_2d(v[k], *st) for (v, _), st in zip(vols.values(), styles)]
      header = frame_header(band, name, int(round(k / sub)), times[k], coverage)
      rows.append(compose_frame(
          images, panel_titles(vols), styles,
          header, panel_inches=3.6,
      ))
    frames.append(np.concatenate(rows, axis=0))
    print(f"frame {k + 1}/{len(times)}", flush=True)

  suffix = "" if FLAGS.channel == "all" else f"_{FLAGS.channel}"
  write_animation(frames, os.path.join(FLAGS.out_dir, f"{out_stem(cfg)}{suffix}"),
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
