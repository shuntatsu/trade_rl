"""Cost-aware multi-symbol allocation policies for the shared portfolio replay.

Import from the concrete modules; this package intentionally re-exports nothing so
the ``trade_rl.strategies`` facade stays unchanged and ``costs`` (the only module
that reads the simulation execution profile) is not pulled in implicitly.
"""
