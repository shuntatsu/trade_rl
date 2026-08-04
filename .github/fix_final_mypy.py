from pathlib import Path

path = Path("trade_rl/studio/system_probe.py")
text = path.read_text(encoding="utf-8")
text = text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom typing import Any\n", 1)
anchor = '''class _MemoryStatus(ctypes.Structure):
'''
helper = '''def _windows_kernel32() -> Any | None:
    """Return kernel32 without assuming Windows-only ctypes attributes exist."""

    windll = getattr(ctypes, "windll", None)
    return None if windll is None else windll.kernel32


'''
if text.count(anchor) != 1:
    raise SystemExit("system probe class anchor changed")
text = text.replace(anchor, helper + anchor)
old_cpu = '''    idle = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
'''
new_cpu = '''    kernel32 = _windows_kernel32()
    if kernel32 is None:
        return None
    idle = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    if not kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
'''
if text.count(old_cpu) != 1:
    raise SystemExit("system probe CPU anchor changed")
text = text.replace(old_cpu, new_cpu)
old_memory = '''    if platform.system() == "Windows":
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            used = status.total_physical - status.available_physical
            return SystemMetric(
'''
new_memory = '''    if platform.system() == "Windows":
        kernel32 = _windows_kernel32()
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if kernel32 is not None and kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            used = status.total_physical - status.available_physical
            return SystemMetric(
'''
if text.count(old_memory) != 1:
    raise SystemExit("system probe memory anchor changed")
path.write_text(text.replace(old_memory, new_memory), encoding="utf-8")

path = Path("trade_rl/studio/overview.py")
text = path.read_text(encoding="utf-8")
old = '''        for item in datasets:
            if item.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"データセット {item.name} が無効です",
                    age=_relative_age(item.updated, now=now),
                )
            )
'''
new = '''        for dataset in datasets:
            if dataset.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"データセット {dataset.name} が無効です",
                    age=_relative_age(dataset.updated, now=now),
                )
            )
'''
if text.count(old) != 1:
    raise SystemExit("overview dataset loop anchor changed")
text = text.replace(old, new)
old = '''        for item in runs:
            if item.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"run {item.run_id} が無効です",
                    age=_relative_age(item.completed_at or item.created_at, now=now),
                )
            )
'''
new = '''        for run in runs:
            if run.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"run {run.run_id} が無効です",
                    age=_relative_age(run.completed_at or run.created_at, now=now),
                )
            )
'''
if text.count(old) != 1:
    raise SystemExit("overview run loop anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")

path = Path("trade_rl/integrations/cost_critic_ppo.py")
text = path.read_text(encoding="utf-8")
old = '''        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return resolved
'''
new = '''        resolved = torch.device(device)
        resolved_index = getattr(resolved, "index", None)
        if resolved.type == "cuda" and resolved_index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return resolved
'''
if text.count(old) != 1:
    raise SystemExit("cost critic device anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")
