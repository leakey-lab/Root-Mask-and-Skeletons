import json
from pathlib import Path
from collections import Counter
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_html

OUT = Path(r"D:\Root-Mask-and-Skeletons\code_review_graph")
extraction = json.loads((OUT/".extract.json").read_text())
analysis = json.loads((OUT/".analysis.json").read_text())
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis["communities"].items()}

# communities maps cid -> list of node ids
def label_for(members):
    dims = [G.nodes[m].get("label","") for m in members if m.startswith("dim_")]
    if dims:
        return dims[0].split(":")[1].strip() if ":" in dims[0] else dims[0]
    files = [G.nodes[m].get("label","") for m in members if m.startswith("file_")]
    if files:
        return files[0]
    # else dominant source file among findings
    srcs = [G.nodes[m].get("source_file","") for m in members]
    c = Counter(s.split("/")[-1] for s in srcs if s)
    return (c.most_common(1)[0][0] if c else "misc")

labels = {cid: label_for(members) for cid, members in communities.items()}
cohesion = {int(k): v for k, v in analysis["cohesion"].items()}
questions = suggest_questions(G, communities, labels)
detection = {"total_files": 97, "total_words": 50000, "needs_graph": True, "warning": None,
             "files": {"code": [], "document": [], "paper": []}}
report = generate(G, communities, cohesion, labels, analysis["gods"], analysis["surprises"],
                  detection, {"input":0,"output":0}, "code_review_findings.json",
                  suggested_questions=questions)
(OUT/"GRAPH_REPORT.md").write_text(report, encoding="utf-8")
to_html(G, communities, str(OUT/"graph.html"), community_labels=labels)
(OUT/".labels.json").write_text(json.dumps({str(k):v for k,v in labels.items()}))
print("Relabeled", len(labels), "communities; HTML + report regenerated")
print("Sample labels:", list(dict.fromkeys(labels.values()))[:15])
