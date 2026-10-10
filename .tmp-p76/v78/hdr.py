import json, struct
for fn in ("/models/target/outside.safetensors", "/models/target/mtp.safetensors"):
    print("==", fn)
    f = open(fn, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        if any(s in k for s in ("lm_head", "embed", "shared_head", "norm")):
            print(f"  {k:70s} {v['dtype']:8s} {v['shape']}")
