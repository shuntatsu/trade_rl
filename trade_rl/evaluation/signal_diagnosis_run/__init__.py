"""Runner of the frozen signal diagnosis (Issue #1146, card SD2).

Applies ``signal_diagnosis_protocol_v4`` to the consumed development Dataset and writes
one verdict through the merged pure verdict code. Nothing is tunable here: every grid,
seed, threshold and window is read from the protocol. The package owns the filesystem
and process boundary that the protocol and measurement packages deliberately lack.
"""
