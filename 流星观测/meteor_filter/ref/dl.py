import base64, json, sys, urllib.request

def fetch(repo, path, out, ref="main"):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    req = urllib.request.Request(url, headers={"User-Agent": "dl"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    data = base64.b64decode(d["content"])
    open(out, "wb").write(data)
    print(f"{out}  {len(data)} bytes")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        fetch("LilacMeteorObservatory/MetDetPy", path, "ref/" + path.replace("/", "__"))
