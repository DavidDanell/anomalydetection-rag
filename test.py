import yaml

with open("config.yaml", encoding="utf-8") as f:
    dicten = yaml.safe_load(f)

for a in dicten['data']['sensors']:
    print(a)


with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["data"]["seed"])
print(cfg["data"]["sensors"][0]["mean"])
print(len(cfg["data"]["sensors"]))