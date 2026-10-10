from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("/models/target")
for name, p in (("BASE_FULL_IMPORT", "/root/fb_raw.md"),
                ("WS_STRIPPED_V4", "/root/fb_stripped.md"),
                ("CC_CUT_V5", "/root/fable_prompt_cc.md")):
    n = len(t(open(p, encoding="utf-8").read()).input_ids)
    print(name, n)
print("COUNT_DONE")
