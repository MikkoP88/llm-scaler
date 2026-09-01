import json, urllib.request, random, sys
random.seed(65)
filler = " ".join(random.choice(["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"]) for _ in range(60000))
prompt = "Summarize the following document.\n" + filler
body = {"model":"qwen3.8-27b-fp8","prompt":prompt,"max_tokens":900,"temperature":0,"ignore_eos":True}
req = urllib.request.Request("http://localhost:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
u = r["usage"]
print("prompt_tokens", u["prompt_tokens"], "completion", u["completion_tokens"])
