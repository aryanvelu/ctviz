"""Intent handlers. Importing this package registers every handler with ``ctviz.viz.base.register``."""

import contextlib

from ctviz.handlers import cohort  # noqa: F401

for _mod in ("trial_text", "results"):
    with contextlib.suppress(ModuleNotFoundError):  # built in parallel; absent until its builder lands
        __import__(f"ctviz.handlers.{_mod}")
