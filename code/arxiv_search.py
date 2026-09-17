#!/usr/bin/env python3
"""Manual literature search harness (arXiv Atom API) - fallback for paper-finder."""
import httpx, time, json, re, sys, xml.etree.ElementTree as ET

NS = {'a': 'http://www.w3.org/2005/Atom'}
API = "http://export.arxiv.org/api/query"

def search(query, max_results=25, sort="relevance"):
    params = {"search_query": query, "max_results": max_results,
              "sortBy": "relevance" if sort == "relevance" else "submittedDate",
              "sortOrder": "descending"}
    for attempt in range(4):
        try:
            r = httpx.get(API, params=params, timeout=60, follow_redirects=True)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            out = []
            for e in root.findall('a:entry', NS):
                aid = e.find('a:id', NS).text.strip()
                m = re.search(r'abs/([\d.v]+)', aid)
                out.append({
                    "arxiv_id": m.group(1) if m else aid,
                    "title": " ".join(e.find('a:title', NS).text.split()),
                    "authors": [a.find('a:name', NS).text for a in e.findall('a:author', NS)],
                    "published": e.find('a:published', NS).text[:10],
                    "abstract": " ".join(e.find('a:summary', NS).text.split()),
                    "pdf": f"https://arxiv.org/pdf/{m.group(1)}" if m else None,
                    "categories": [c.get('term') for c in e.findall('a:category', NS)],
                })
            return out
        except Exception as ex:
            print(f"  retry {attempt}: {type(ex).__name__} {str(ex)[:80]}", file=sys.stderr)
            time.sleep(4 * (attempt + 1))
    return []

if __name__ == "__main__":
    queries = json.load(open(sys.argv[1]))
    allp = {}
    for label, q in queries.items():
        res = search(q, max_results=int(sys.argv[2]) if len(sys.argv) > 2 else 25)
        print(f"[{label}] {len(res)} hits", file=sys.stderr)
        for p in res:
            p.setdefault("queries", [])
            key = p["arxiv_id"].split('v')[0]
            if key in allp:
                allp[key]["queries"].append(label)
            else:
                p["queries"] = [label]
                allp[key] = p
        time.sleep(3)
    with open(sys.argv[3] if len(sys.argv) > 3 else "paper_search_results/arxiv_all.json", "w") as f:
        json.dump(list(allp.values()), f, indent=1)
    print(f"TOTAL UNIQUE: {len(allp)}", file=sys.stderr)
