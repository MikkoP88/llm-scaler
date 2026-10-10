import torch

print("xpu_avail", torch.xpu.is_available(), flush=True)
for gb in (4, 16, 32, 48, 60):
    try:
        t = torch.empty(gb * 1024**3, dtype=torch.uint8, pin_memory=True)
        t[::4096] = 1  # first-touch every page
        print(gb, "GB_PINNED_OK", flush=True)
        del t
    except Exception as e:  # noqa: BLE001
        print(gb, "FAIL", type(e).__name__, str(e)[:100], flush=True)
        break
print("PROBE_DONE", flush=True)
