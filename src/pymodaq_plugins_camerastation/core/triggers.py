"""Watcher of the event files written by Laser Control on computer A (option O8).

Laser Control writes ``event_<date>_<id>_<kind>.json`` atomically (temporary file, then rename) in a
folder shared with computer B, on selected events (shake, ML lost...). This watcher lists the folder
at a low rate (``poll`` is called by a 1-2 s timer) and returns the new events; the station then
saves a short burst of images of each camera.

* files present when the watcher starts are **not** replayed (old events);
* an unreachable folder (network share down) is reported in ``error`` and retried at the next poll;
* an unreadable file is skipped once (reported), never retried in a loop;
* nothing is deleted: the folder belongs to computer A.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set


class TriggerWatcher:
    def __init__(self, folder: Optional[str]):
        self.folder = Path(folder) if folder else None
        self.error = ''
        self.n_events = 0
        self._seen: Optional[Set[str]] = None        # None until the first successful listing

    @property
    def enabled(self) -> bool:
        return self.folder is not None

    def _list(self) -> Set[str]:
        names = set()
        with os.scandir(self.folder) as it:
            for e in it:
                n = e.name
                if n.startswith('event_') and n.endswith('.json'):
                    names.add(n)
        return names

    def poll(self) -> List[Dict]:
        if self.folder is None:
            return []
        try:
            names = self._list()
        except OSError as e:
            self.error = f'trigger folder not reachable: {e}'
            return []
        self.error = ''
        if self._seen is None:                        # first listing: everything already there is old
            self._seen = names
            return []
        new = sorted(names - self._seen)
        self._seen = names                            # forget deleted files (bounded memory)
        out = []
        for n in new:
            try:
                ev = json.loads((self.folder / n).read_text(encoding='utf-8'))
                if not isinstance(ev, dict):
                    raise ValueError('not a JSON object')
            except Exception as e:
                self.error = f'event file {n} unreadable: {e}'
                ev = {'kind': 'unreadable', 'message': str(e)}
            ev.setdefault('file', n)
            out.append(ev)
        self.n_events += len(out)
        return out
