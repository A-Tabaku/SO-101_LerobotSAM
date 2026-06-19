import json
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

api = HfApi()
repos = [m.id for m in api.list_models(author="Atabaku") if "pi05" in m.id]
print("Found pi05 models:", repos, flush=True)
print("--- train configs ---", flush=True)
for repo in repos:
    info = {"repo": repo}
    try:
        files = list_repo_files(repo)
        if "train_config.json" in files:
            d = json.load(open(hf_hub_download(repo, "train_config.json")))
            pol = d.get("policy") or {}
            info.update(dataset=(d.get("dataset") or {}).get("repo_id"),
                        steps=d.get("steps"), expert_only=pol.get("train_expert_only"),
                        pretrained=pol.get("pretrained_path"))
        else:
            info["note"] = "no train_config.json"
    except Exception as e:
        info["error"] = repr(e)[:90]
    print(json.dumps(info), flush=True)
