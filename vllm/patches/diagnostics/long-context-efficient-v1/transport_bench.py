"""Small, finite two-rank XCCL latency/throughput test. No server changes.

Run in the serving image with torchrun --standalone --nproc-per-node=2.
All-reduce input is reset before every call to avoid repeated-sum overflow.
Reports wall latency with synchronization, correctness, and payload rate;
the latter is not advertised as physical PCIe bus bandwidth.
"""
import datetime
import json
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.xpu.set_device(rank)
dist.init_process_group('xccl', timeout=datetime.timedelta(seconds=60))
try:
    for rows in (1, 5, 8, 64, 2048, 8192):
        src = torch.full((rows, 5120), rank + 1., device=f'xpu:{rank}', dtype=torch.float16)
        work = torch.empty_like(src)
        dist.barrier()
        for _ in range(3):
            work.copy_(src)
            dist.all_reduce(work)
        torch.xpu.synchronize()
        times = []
        for _ in range(12):
            work.copy_(src)
            torch.xpu.synchronize()
            start = time.perf_counter()
            dist.all_reduce(work)
            torch.xpu.synchronize()
            times.append(time.perf_counter() - start)
        correct = bool(torch.all(work == 3).item())
        record = dict(rank=rank, rows=rows, bytes=work.numel()*work.element_size(),
                      median_ms=sorted(times)[len(times)//2]*1000, correct=correct,
                      samples_ms=[t*1000 for t in times])
        record['payload_GB_s'] = record['bytes']/1e6/record['median_ms']
        print(json.dumps(record), flush=True)
        assert correct
        del src, work
    dist.barrier()
finally:
    torch.xpu.synchronize()
    dist.destroy_process_group()
