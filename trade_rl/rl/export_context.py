"""Keep eager validation outside traced and ONNX policy graphs."""

from __future__ import annotations

import torch


def graph_export_active() -> bool:
    """Return whether PyTorch is currently capturing an inference graph."""

    return bool(
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or torch.onnx.is_in_onnx_export()
    )


__all__ = ["graph_export_active"]
