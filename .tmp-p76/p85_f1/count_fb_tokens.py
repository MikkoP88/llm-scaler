from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("/models/target")
for name, p in (("BASE_FULL_IMPORT", "/root/fb_raw.md"),
                ("STRIPPED_IMPORT", "/root/fb_stripped.md")):
    n = len(t(open(p, encoding="utf-8").read()).input_ids)
    print(name, n)
print("COUNT_DONE")
