# Changelog

## 0.3.4 — 2026-10-22

- **Fix: the real cameras of computer B published a centroid (valid = 1) without any beam.** Their background is
  not white noise: a vignetting dome (Plasma) or horizontal bands (Focus) of about one count make the block means
  vary 4 to 8 times more than the pixel noise predicts (an 8-bit sensor at 5 counts even gives a pixel noise of
  0). The brightest block must now also stand out of the robust spread (1.4826 MAD) of the background blocks by
  the same factor 6; in a small ROI or mask (< 100 blocks) the pixel noise test alone applies, as before. A beam
  of 3 counts on these backgrounds is still seen. Tests with backgrounds modelled on the measured ones.
- **Fix: a station on PyMoDAQ 5.3 crashed the LECO viewer of a computer A on 5.1** (`IndexError` in viewer0D):
  groups of more than 6 channels are sent with `do_plot=False` whatever the local PyMoDAQ version (5.3 has no
  `plot_colors` and plotted everything).
- `tools/leco_link_check` sender: a new result at each publication (age 0.1 s).

## 0.3.3 — 2026-10-22

- **`tools/add_relay`**: copies a preset of computer B (the source is only read) and adds the `CameraStation`
  relay with the LECO address of computer A and its LECO name.
- **The station connects a relay that has a LECO name** to the coordinator when it starts it. "Connect" cannot be
  saved ticked in a PyMoDAQ 5.1 preset: the dashboard restores it before the host and the port.
- **`tools/leco_link_check`** (sender / receiver) and **`tools/leco_station_check`** (simulated cameras → station
  → relay, station files in a temporary folder): test the LECO link B → A without hardware.

## 0.3.2 — 2026-10-22

- **Fix: the relay sent nothing to the dashboard on PyMoDAQ 5.1** (`list index out of range` in
  `DAQ_Viewer.show_data`, logged at every publication). The 0D viewer of PyMoDAQ 5.1 takes one pen per channel
  from `plot_colors` of the PyMoDAQ config (7 by default, 6 on computer B), without cycling; a beam camera
  publishes 10 values and a disk camera 12: plotting raised before `grab_done_signal`. A camera with more
  values than the viewer can draw is now sent with `do_plot=False`: exported and sent to LECO as before, not
  plotted (the trends of the station show them). PyMoDAQ 5.3 cycles its colors and plots everything.

## 0.3.1 — 2026-10-16

- **Fix: nothing reached computer A through LECO.** PyMoDAQ forwards a viewer's data to LECO only when its
  grab is started "for LECO"; the station started the relay with a plain grab, so A received nothing. And a
  Grab sent by A (Laser Control starting its viewers) toggled the running relay *off*. The station now starts
  the relay with LECO forwarding and restarts it (log: "relay (re)started with LECO forwarding") when it was
  stopped by A or started from the dashboard Grab button. Checked end to end with a LECO coordinator, PyMoDAQ
  5.1.10 and 5.3.

## 0.3.0 — 2026-10-14

- **One panel per camera**, like a dashboard viewer: Snap (one frame, analysed whatever the gate), Grab
  (continuous; starts the relay too), Stop, Preview, ⚙ settings of the camera; camera rate → analysed
  rate, calc time, results, and why there is no value. Replaces the status table. *Grab all / Stop all*
  stay in the toolbar.
- **Trends dock**: one plot per quantity with its own scale (the dashboard 0D viewer put a `sum` of 1e6
  and a centroid of 320 px on one axis), every result plotted; on / off, history, plots to show.
- **Settings profiles** (save / load / delete the whole station settings).
- **Rate gate fixed**: a minimum interval analysed a 15 Hz camera at 7.5 Hz with a 10 Hz gate, and
  frames delivered in bursts by a busy GUI were dropped even from a slower camera. Now a token bucket
  timed with the acquisition time of the frame (DataWithAxes timestamp): the set rate is reached.
  Results carry the acquisition time.
- **Display rate** setting (default 5 Hz) for panels, preview and trends (was 2 Hz / 1 Hz, `[ui]` of
  the config file, now unused).
- PyMoDAQ 5.3: the extension toolbar keeps *Show dashboard* only (no experiment / state managers, which
  act on the dashboard).
- `CamSim` default frame rate 15 Hz (was 5 Hz: the office simulator could not exceed 5 analyses/s).
- `tools/resize_watch.py`: names the widget that keeps resizing (display that keeps moving after a dock
  edge was dragged, seen on PyMoDAQ 5.3 without the extension).

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
