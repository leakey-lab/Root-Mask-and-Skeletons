import json, os
from pathlib import Path
from collections import defaultdict

REPO = Path(r"D:\Root-Mask-and-Skeletons")
OUT = REPO / "code_review_graph"
OUT.mkdir(parents=True, exist_ok=True)

reg = json.loads((REPO / "code_review_findings.json").read_text(encoding="utf-8"))
findings = reg.get("findings", [])

DIM_NAMES = {
 "D1":"Scientific validity","D2":"ML inference","D3":"Correctness","D4":"Performance",
 "D5":"Memory/resources","D6":"Error handling","D7":"Security","D8":"Architecture",
 "D9":"Concurrency","D10":"Interoperability","D11":"Reproducibility","D12":"Docs",
 "D13":"Repo hygiene",
}
SEV_W = {"critical":5.0,"high":4.0,"medium":3.0,"low":2.0,"info":1.0}

def fid(f): return f.get("canonical_id") or f.get("id") or f.get("title","?")[:20]
def norm_file(p):
    if not p: return "unknown"
    return str(p).replace("\\","/").strip()

nodes, edges = [], []
seen = set()
def add_node(nid,label,ftype,src):
    if nid in seen: return
    seen.add(nid)
    nodes.append({"id":nid,"label":label,"file_type":ftype,"source_file":src,
                  "source_location":None,"source_url":None,"captured_at":None,
                  "author":None,"contributor":None})

for d,name in DIM_NAMES.items():
    add_node(f"dim_{d}", f"{d}: {name}", "document", "code_review_findings.json")

file_to_findings = defaultdict(list)
for f in findings:
    fi = fid(f); fl = norm_file(f.get("file"))
    dim = (f.get("dimension") or "").split()[0] if f.get("dimension") else ""
    sev = (f.get("severity") or "info").lower()
    add_node(f"file_{fl}", fl.split("/")[-1], "code", fl)
    add_node(f"find_{fi}", f"[{sev}] {f.get('title','')[:70]}", "document", fl)
    file_to_findings[fl].append(fi)
    w = SEV_W.get(sev,1.0)
    edges.append({"source":f"find_{fi}","target":f"file_{fl}","relation":"located_in",
                  "confidence":"EXTRACTED","confidence_score":1.0,"source_file":fl,"source_location":None,"weight":w})
    if dim in DIM_NAMES:
        edges.append({"source":f"find_{fi}","target":f"dim_{dim}","relation":"belongs_to",
                      "confidence":"EXTRACTED","confidence_score":1.0,"source_file":fl,"source_location":None,"weight":w})

for fl, fis in file_to_findings.items():
    for i in range(len(fis)):
        for j in range(i+1,len(fis)):
            edges.append({"source":f"find_{fis[i]}","target":f"find_{fis[j]}","relation":"shares_file_with",
                          "confidence":"INFERRED","confidence_score":0.7,"source_file":fl,"source_location":None,"weight":1.0})

extract = {"nodes":nodes,"edges":edges,"hyperedges":[],"input_tokens":0,"output_tokens":0}
(OUT/".extract.json").write_text(json.dumps(extract,indent=2))
print(f"Built extraction: {len(nodes)} nodes, {len(edges)} edges from {len(findings)} findings")
