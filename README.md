# Camera Station — PyMoDAQ extension of the camera computer

Counterpart of **Laser Control** on the camera computer (computer B). It replaces the live-analysis
part of the BeamAnalyzer extension with an extension designed for **robustness and low CPU**:

- **Analysis** of each camera of the dashboard: beam centroid / rms size / sum / peak / saturation,
  fluorescence **depletion of the disk** (reference − image, two passes) with kurtosis, or ROI intensity
  only (plasma, gas jet…).
- **Publication to computer A** (Laser Control) as 0D values through the `CameraStation` 0D viewer
  connected to LECO, at a fixed rate, with a **heartbeat** (`Stats/frame_counter`, option O9) and B's clock.
- **Logger (stage 0.5)**: one HDF5 file per session with the 0D values, **reduced images** (8 bits,
  binned), notes and events, **synchronised with the Laser Control recordings** through the heartbeat.
- **Your own analyses, live**: a `.py` file dropped in the scripts folder is loaded and **reloaded each
  time it is saved, without restarting PyMoDAQ**; its parameters are editable live; its results are shown,
  recorded, and sent to A only if you tick *Publish to A* (see *Testing new analyses*).
- **Events of computer A (O8, optional)**: only if a folder is shared between the computers (not the
  case in the lab today: off). The synchronisation does not need it.

## Design rules (why it is light and robust)

| Rule | How |
|---|---|
| Bounded CPU | Each camera is analysed at most *Max rate* times per second (default 2 Hz); extra frames are dropped before any work. One worker thread for all cameras: the analyses never use more than one core, the GUI thread never computes. |
| Cheap analysis | ROI + binning (default 2) + moments from the row / column projections: **~4 ms for a 1.3 Mpx frame** on the office computer (binning 2). |
| Bounded memory / latency | One pending frame per camera at most ("latest wins"); the image recording queue is bounded in MB (images dropped and counted, never the GUI blocked). |
| Never a fake value | No beam, no reference, a camera that stopped, an analysis error → **NaN and `valid = 0`**, never 0.0 or a frozen value (same rule as S12 of Laser Control). A camera without result for *Stale after* seconds publishes NaN. |
| Isolation | An exception in one camera only affects that camera (✖ error in the status, logged once); a disk error only affects the recording (new file, losses counted). |
| No silent alive | The 0D relay sends each publication **once** (live mode). The old BeamAnalyzer relay re-sent its cached values continuously: CPU for nothing, and a frozen extension looked alive to computer A. |
| Crash-safe settings | JSON written atomically, previous version kept as `.bak`, an unreadable file is renamed `.corrupt-…` and reported. |
| Goggles | States are symbol + text (`▶ ok`, `⚠ no beam`, `⚠ no frames`, `✖ error`, `■ off`), never colour alone. |

Measured on the office computer, two 1.3 Mpx cameras at 10 fps: the station itself costs ~7 % of one
core for the analysis (2 Hz each); preview (1 Hz) and recording add almost nothing. **The main cost on
the camera computer is PyMoDAQ displaying the frames in the camera viewers** (see *Commissioning*).

## What computer A receives

On A, each value is `<LECO viewer title>/<data name>/<label>`:

| Data name | Labels | Notes |
|---|---|---|
| `<camera name>` (beam) | `x`, `y`, `sx`, `sy`, `sum`, `peak`, `sat`, `valid`, `age`, `frame` | full-frame pixels; `sx`, `sy` = rms (σ); `sat` = saturated fraction |
| `<camera name>` (disk) | `x`, `y`, `sx`, `sy`, `sum`, `kx`, `ky`, `peak`, `sat`, `valid`, `age`, `frame` | moments of the depletion; `kx < 2` ≈ two passes side by side in x |
| `<camera name>` (sum) | `sum`, `mean`, `peak`, `sat`, `valid`, `age`, `frame` | |
| `Stats` | `frame_counter`, `t_wall`, `cams_ok`, `recording` | `frame_counter` = heartbeat (+1 per publication) |

`age` = seconds since the analysed frame, `frame` = frame number of the camera (frozen = camera frozen).
The labels of a camera only depend on its analysis kind, so a role on A never disappears because a
setting changed. **Renaming a camera changes its keys on A** (the log says so).

## Compatibility

PyMoDAQ **5.1.x** (developed and tested on 5.1.10, Python 3.11, PyQt5); written for 5.2.x like Laser
Control (both CustomApp APIs, experiments) but not tested on 5.2 yet.

## Installation

In the PyMoDAQ environment of the camera computer:

```bash
cd pymodaq_plugins_camerastation
pip install -e .
```

The extension appears as **Camera Station** in the *Extensions* menu, and the plugins `CameraStation`
(0D relay) and `CamSim` (simulated 2D camera) in the plugin lists.

Tests (≈ 1 min, offscreen Qt): `pip install pytest`, then `pytest`. Some tests use
`pymodaq_plugins_lasercontrol` when it is installed, to check the keys and the file synchronisation
against Laser Control's own code.

## Office simulator

```bash
python -m pymodaq_plugins_camerastation.tools.sim_preset
python -m pymodaq_plugins_camerastation.extensions.camera_station camera_sim
```

Two simulated cameras (`Disk cam`: disk fluorescence depleted by two passes; `Output cam`: a drifting
beam) and the `CameraStation` relay. Click **Grab cameras** in the toolbar. For the disk: untick *Laser
on* in the settings of the `Disk cam` viewer, **Capture reference (laser OFF)** in *Cameras > Disk >
Disk*, then tick *Laser on* again.

## Commissioning in the lab (computer B)

1. **Dashboard of B**: the 2D viewers of the cameras + one 0D viewer **CameraStation**, connected to
   LECO (as for any LECO viewer; on A, the LECO director viewer of B, e.g. `LECO_B`).
2. **Reduce the load of the camera viewers** (the biggest CPU cost): in each camera viewer, untick
   *Show data and process* (frames still reach the station), or set a *Refresh time* of 500–1000 ms.
   Lower the camera frame rate / exposure-limited rate when possible: the station only needs 1–2 Hz.
3. **Camera Station**: *Add camera* (takes the next free 2D viewer), give it a short **name** (`Disk`,
   `Output`, `PM_near`, `PM_far`, `Plasma`…: this is the channel name on A), choose the analysis, the
   ROI (or drag it in the preview), binning 2–4, *Saturation* (4095 for a 12-bit camera).
   Disk camera: mask centre / radius, **Capture reference** with the laser off (average of N frames).
4. **Grab cameras** (toolbar) starts the camera viewers and the relay; the status table shows each
   camera (incoming / analysed rate, calc time, values, reason when invalid) and the bottom line the
   publication, the recording, the trigger folder and the CPU load of the process.
5. **On computer A (Laser Control)**: signal roles `Disk_x` = `LECO_B/Disk/x`…, and in *Advanced →
   Frame counters* the heartbeat `LECO_B/Stats/frame_counter`.
6. **Events (optional, not available in the lab today)**: only with a folder shared between A and B
   (`event_trigger_folder` in `config_lasercontrol.toml` on A, *Trigger folder (from A)* here). Without
   it, the events of A are in A's recording with the heartbeat value, so the images of B around an
   event are found offline; lower *Image every (s)* if more images are needed.
7. **Recording**: **Recording** button (toolbar), or `[recording] autostart = true` in
   `config_camerastation.toml` for an always-on logger.

## Testing new analyses (scripts, no restart)

The scripts folder is `~/.pymodaq/camera_station/scripts` (*Settings > Scripts > Open folder*). At the
first start it receives `_template.py` (documented API, ignored because of the `_`) and `spot_shape.py`
(orientation, ellipticity and asymmetry of the spot: on the disk camera it describes the two
overlapping passes together, e.g. a growing ellipticity or tilt when they start to separate).

```python
LABELS = ('peak_x', 'fraction_in_box')        # values returned = channel labels
PARAMS = {'box_half_size': 10}                # editable live in Settings > Scripts > <script>

def analyze(frame, ctx):                      # frame: raw camera frame, read-only
    w = ctx.depletion(frame) if ctx.reference is not None else ctx.crop(frame, background=True)
    ...                                       # numpy on the binned ROI
    ctx.show(w)                               # Preview > Show: script output
    return {'peak_x': x, 'fraction_in_box': f}
```

The cycle in the lab: edit the file, **save**, look at the *Scripts* table (values, ms, errors) and at
the preview; no restart, nothing else is disturbed:

- tick the script under *Cameras > <camera> > Scripts*; it runs on the frames analysed for that camera;
- the folder is checked every 2 s; a saved file is reloaded (its `ctx.state` reset) and the reload is
  written in the recording. **A new version that does not load keeps the previous one running** (the
  error and its line are in the status and the log);
- scripts run in **their own thread**, after the built-in analysis: a slow or broken script never
  delays the values sent to A; an exception = NaN for that frame (shown); a script stuck longer than
  *Timeout* is disabled until its file is saved again;
- the results are **recorded** (`<camera>_<script>/<label>` columns) but **not sent to A** until
  *Publish to A* is ticked; then A sees `<viewer>/<camera>_<script>/<label>`;
- a script that turns out to be useful can later become a built-in analysis.

The same file runs offline, at the office, on the images of a recording:

```python
from pymodaq_plugins_camerastation.core.scripts import run_on_session, run_offline
r = run_on_session('scripts/my_analysis.py', 'station_lab_20261010_101500.h5', 'disk',
                   params={'box_half_size': 8}, kind='disk', reference='references/disk_ref_....npy')
r['peak_x'], r['counter']          # one value per saved image, with the heartbeat value of the image
```

Recorded images are binned / 8 bits; `ctx.to_frame` still returns full-frame camera pixels.

## Logger and synchronisation with computer A

```
station_<machine>_<date>.h5
  /scalars/t, /scalars/frame_counter, /scalars/values/<Cam|label>      every publication (2 Hz)
  /images/<camera id>/data (n, h, w) uint8 + t, frame, counter, kind, event_id, offset, scale
  /events (notes, trigger events of A, settings changes, references, start / stop)
```

- **Exact join**: every publication carries `Stats/frame_counter`; Laser Control records it in each
  snapshot (`raw/<viewer>/Stats/frame_counter`, or `det/<viewer>/heartbeat` if it is the declared
  heartbeat). The same counter is the row key of `/scalars` and is stored with each image.
- **Clock offset**: `Stats/t_wall` (clock of B) is recorded on A, so the offset between the two clocks is
  known at every snapshot. No shared clock or NTP is needed.
- **Image size**: 8 bits with a per-image offset and scale (counts ≈ offset + value × scale), binned 2,
  gzip: ~50–200 kB per 1.3 Mpx image instead of 2.6 MB. One image per camera every 30 s ≈ 50–100 MB/h
  for 4 cameras; 16 bits and full resolution are available in the settings.

```python
from pymodaq_plugins_camerastation.core.loader import load_session, align_with_laser_control
b = load_session('station_lab_20261010_101500.h5')
b['scalars']['Disk/x'], b['t'], b['events']
img = b['image']('disk', -1)                      # counts of the last saved image of camera "disk"
m = align_with_laser_control('lab_session.h5', 'station_lab_20261010_101500.h5')
m['b_index']                                      # row of B for each snapshot of A (-1 if none)
m['clock_offset_median']                          # t_A - t_B (s)
```

## Where the files are

| File | Default location |
|---|---|
| `config_camerastation.toml` | PyMoDAQ configuration folder (machine name, folders, recording) |
| `station_<machine>.json` (+ `.bak`) | `~/.pymodaq/camera_station/` (cameras and settings) |
| `references/*.npy` | `~/.pymodaq/camera_station/references/` (disk references, dark frames) |
| `recordings/YYYY-MM-DD/station_*.h5` | `~/.pymodaq/camera_station/recordings/` |
| `logs/camera_station.log` | `~/.pymodaq/camera_station/logs/` (rotating, repeated messages folded) |

## Code organization

```
src/pymodaq_plugins_camerastation/
├── core/         no Qt, no PyMoDAQ (tested alone)
│   ├── analysis.py    analyses (beam, disk, sum), ROI, binning, projections
│   ├── engine.py      rate gate, latest-wins mailbox, worker thread, states, reference capture
│   ├── publisher.py   0D payload, heartbeat, staleness → NaN
│   ├── recorder.py    HDF5 writer thread, image reduction
│   ├── scripts.py     user scripts: hot reload, own thread, watchdog, offline runs
│   ├── triggers.py    event files of Laser Control (O8)
│   ├── loader.py      reading + alignment with Laser Control
│   ├── settings.py    validated settings, atomic JSON
│   └── sim.py         synthetic frames (tests, CamSim)
├── relay.py      Qt bus extension → relay plugin
├── ui/           station.py (controller), params.py (trees), preview.py
├── extensions/camera_station.py      the PyMoDAQ extension
├── daq_viewer_plugins/plugins_0D/    CameraStation relay
├── daq_viewer_plugins/plugins_2D/    CamSim
└── tools/sim_preset.py               office simulator configuration
```

## Points to check in the lab

- **Camera viewers display**: with *Show data and process* on and no refresh time, PyMoDAQ draws every
  frame of every camera; on the lab computers this, not the station, is what will load the CPU.
- **LECO**: check that A's LECO viewer of B comes back after a restart of B, and that the heartbeat
  advances on A.
- **Disk reference**: the depletion depends on the pump power; take the reference at the pump power of
  operation, laser blocked. A reference of another shape is refused (reason shown).
- **Saturation**: set the real saturation level for 12-bit cameras delivered in 16-bit words, otherwise
  `sat` stays 0.
- **Two passes on the disk**: they overlap, so the built-in analysis gives their moments together
  (`sx`, `kx`…); `spot_shape.py` adds orientation / ellipticity / asymmetry. New ideas: as scripts.
- **Scripts** run in the PyMoDAQ process: keep the module level light (it runs at each reload, in the GUI
  thread), work on `ctx.crop(...)`, and avoid infinite loops (a stuck script is disabled, but its thread
  is only freed by a restart).

License: MIT (see `LICENSE`).
