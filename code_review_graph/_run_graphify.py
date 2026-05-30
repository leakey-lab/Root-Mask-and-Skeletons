import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html

OUT = Path(r"D:\Root-Mask-and-Skeletons\code_review_graph")
extraction = json.loads((OUT/".extract.json").read_text())

G = build_from_json(extraction)
communities = cluster(G)
cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)
labels = {cid: f"Community {cid}" for cid in communities}
questions = suggest_questions(G, communities, labels)

detection = {"total_files": 97, "total_words": 50000, "needs_graph": True, "warning": None,
             "files": {"code": [], "document": [], "paper": []}}
tokens = {"input": 0, "output": 0}

report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens,
                  "code_review_findings.json", suggested_questions=questions)
(OUT/"GRAPH_REPORT.md").write_text(report, encoding="utf-8")
to_json(G, communities, str(OUT/"graph.json"))
if G.number_of_nodes() <= 5000:
    to_html(G, communities, str(OUT/"graph.html"), community_labels=labels or None)

analysis = {
    "communities": {str(k): v for k, v in communities.items()},
    "cohesion": {str(k): v for k, v in cohesion.items()},
    "gods": gods, "surprises": surprises, "questions": questions,
}
(OUT/".analysis.json").write_text(json.dumps(analysis, indent=2))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")
print("Communities and sizes:")
from collections import Counter
sizes = Counter(communities.values())
for cid, n in sorted(sizes.items()):
    members = [G.nodes[x].get('label','')[:40] for x,c in communities.items() if c==cid][:6]
    print(f"  C{cid}: {n} nodes | sample: {members}")
