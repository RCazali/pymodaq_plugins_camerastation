# Changelog

## 0.2.1 — 2026-10-11

- **Fix: noise alone was published as a valid measurement** (beam and disk analyses). With the laser
  off, the positive half of the noise gave `valid = 1` with a centroid in the middle of the ROI / mask.
  The analysis now measures the noise of each image and needs a local signal above it
  (`NOISE_BLOCK`, `NOISE_FACTOR` in `core/analysis.py`); otherwise NaN, `valid = 0`, reason
  "no signal above the noise". Nothing to set. Cost: about 2 ms per 1.3 Mpx frame at binning 2.
- Still for *Min signal*: a disk fluorescence that changed since the reference (pump power) is a real
  difference with the shape of the disk, so it is seen as a signal.

## 0.2.0 — 2026-10-10

- **User analysis scripts** (`core/scripts.py`): `.py` files in `~/.pymodaq/camera_station/scripts`,
  hot-reloaded when saved (no PyMoDAQ restart), previous version kept when a new one does not load,
  live parameters (`PARAMS`), own thread after the built-in analysis, read-only frame, timeout
  watchdog, results shown (Scripts table, preview "script output"), recorded, and sent to A only when
  "Publish to A" is ticked. Offline: `run_offline`, `run_on_session` on recorded images.
- Examples: `_template.py` (API), `spot_shape.py` (orientation, ellipticity, asymmetry).
- No shared folder between the lab computers: trigger folder documented as optional (off).

## 0.1.0 — 2026-10-10 (first version, office-tested, not yet tested in the lab)

- Analyses per camera: beam (centroid, rms size, sum, peak, saturation), disk depletion (reference −
  image, circular mask, kurtosis), ROI intensity. ROI + binning, projections only (~4 ms / 1.3 Mpx).
- Engine: rate gate per camera, one pending frame per camera, one worker thread, per-camera error
  isolation, states (ok / no beam / no frames / error / off), reference and dark capture (average of N).
- Publication to computer A through the `CameraStation` 0D relay (live mode, each publication once),
  heartbeat `Stats/frame_counter`, B clock `Stats/t_wall`, NaN for stale cameras.
- HDF5 logger: 0D rows at the publication rate, periodic reduced images (8/16 bits), bursts on Laser
  Control event files (O8), notes, snapshots, settings changes; byte-bounded queue, rotation, new file
  after a disk error. Loader and alignment with Laser Control recordings.
- UI: settings tree, status table, light preview with draggable ROI and overlay, log; toolbar Recording,
  Note, Snapshot, Grab / Stop cameras, Reload modules.
- `CamSim` simulated camera and `tools/sim_preset.py` (office), tests (core, real PyMoDAQ viewers,
  dashboard loading).

## Next steps (proposed)

- Pre-event image buffer (images *before* a "ML lost" event), at a small memory budget.
- Clock-offset and LECO-latency check shown in the status (needs A to send back its time).
- Test on PyMoDAQ 5.2 and on the lab computer (CPU budget with the 4 real cameras).
