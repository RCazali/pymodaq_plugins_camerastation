"""Template of a Camera Station analysis script (ignored because its name starts with '_').

Copy it as my_analysis.py in this folder: it is loaded within a few seconds, and reloaded each time
the file is saved (no PyMoDAQ restart). Then tick it under Cameras > <camera> > Scripts.

* LABELS: the values returned (also the channel labels on computer A when "Publish to A" is ticked:
  "<viewer on A>/<camera>_<script>/<label>"). Keep them stable once A uses them.
* PARAMS: editable live in Settings > Scripts > <script> (numbers, booleans, strings).
* RATE_HZ (optional): maximum rate of this script (by default: the analysis rate of the camera).
* analyze(frame, ctx) -> {label: value}. Missing / non finite values become NaN. Return
  {'valid': 0} to say "no measurement". An exception = NaN for this frame, shown in the status.

The frame is the raw camera frame, READ-ONLY. ctx gives:
    ctx.camera, ctx.kind          name and analysis kind of the camera
    ctx.params                    current parameter values
    ctx.state                     dict kept between frames (reset when the file is reloaded)
    ctx.builtin                   built-in results of this frame ('x', 'y', 'sx', ...)
    ctx.reference, ctx.dark       full frames (float32) or None
    ctx.crop(frame, background=False)   ROI + binning of the camera, float32 copy (cheap)
    ctx.depletion(frame)          reference - frame in the ROI (None without reference)
    ctx.moments(w)                {'mx', 'my', 'vx', 'vy', 'sum'} from projections, in crop pixels
    ctx.to_frame(xb, yb)          crop coordinates -> full-frame camera pixels
    ctx.show(array)               shown in the preview (Settings > Preview > Show: script output)

Keep it light: it runs on the camera computer at the analysis rate (1-2 Hz), in its own thread. Work
on ctx.crop(...) (binned ROI) rather than on the full frame. Offline, on recorded images:
    from pymodaq_plugins_camerastation.core.scripts import run_on_session
    r = run_on_session('my_analysis.py', 'station_lab_....h5', 'disk')
"""
import math

import numpy as np

LABELS = ('peak_x', 'peak_y', 'fraction_in_box')
PARAMS = {'box_half_size': 10, 'use_depletion': True}


def analyze(frame, ctx):
    if ctx.params['use_depletion'] and ctx.reference is not None:
        w = ctx.depletion(frame)
    else:
        w = ctx.crop(frame, background=True)
    if w is None or not np.isfinite(w).all():
        return {'valid': 0}
    iy, ix = np.unravel_index(int(np.argmax(w)), w.shape)
    h = int(ctx.params['box_half_size'])
    pos = np.clip(w, 0, None)
    total = float(pos.sum())
    inside = float(pos[max(iy - h, 0):iy + h + 1, max(ix - h, 0):ix + h + 1].sum())
    x, y = ctx.to_frame(ix, iy)
    ctx.show(w)
    return {'peak_x': x, 'peak_y': y, 'fraction_in_box': inside / total if total > 0 else math.nan}
