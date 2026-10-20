"""Deterministic lexical near-duplicate grouping before any split or fitting."""
import re
from collections import Counter, defaultdict


def group_near_duplicates(rows, policy):
    """Link document groups for highly overlapping token-shingle sets.

    All candidates sharing a shingle are checked. This is an exact search for
    this lexical rule, not an exhaustive semantic-paraphrase detector.
    """
    if policy["method"] != "token_shingle_jaccard":
        raise ValueError("Unsupported near-duplicate method")
    width = policy["shingle_size"]
    minimum = policy["min_tokens"]
    threshold = policy["threshold"]
    if type(width) is not int or width < 1 or type(minimum) is not int \
            or minimum < width or not 0 < threshold <= 1:
        raise ValueError("Invalid near-duplicate policy")
    parent = {r["group_id"]: r["group_id"] for r in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    inverted, shingles = defaultdict(list), []
    matches, scored, short = [], 0, 0
    for i, row in enumerate(rows):
        # Punctuation is ignored; words, numerals, and negation words remain.
        tokens = re.findall(r"\w+", row["text"].casefold(), flags=re.UNICODE)
        current = {tuple(tokens[j:j+width]) for j in range(len(tokens)-width+1)} \
            if len(tokens) >= minimum else set()
        short += not bool(current)
        common = Counter(j for shingle in current for j in inverted[shingle])
        for j, intersection in sorted(common.items()):
            other = shingles[j]
            if min(len(current), len(other)) / max(len(current), len(other)) < threshold:
                continue
            scored += 1
            score = intersection / (len(current) + len(other) - intersection)
            if score >= threshold:
                union(row["group_id"], rows[j]["group_id"])
                matches.append({"left_id": rows[j]["id"], "right_id": row["id"],
                                "left_doc_id": rows[j]["doc_id"],
                                "right_doc_id": row["doc_id"], "jaccard": score})
        shingles.append(current)
        for shingle in current:
            inverted[shingle].append(i)
    grouped = [{**r, "group_id": find(r["group_id"])} for r in rows]
    group_docs = defaultdict(set)
    for r in grouped:
        group_docs[r["group_id"]].add(r["doc_id"])
    return grouped, {
        "policy": policy, "candidate_pairs_scored": scored,
        "matching_pairs": len(matches), "pairs": matches,
        "sentences_below_min_tokens": short,
        "groups_before": len(parent), "groups_after": len(group_docs),
        "largest_group_document_count": max(map(len, group_docs.values()), default=0),
        "scope": "Retained three-class sentences; lexical token overlap only. "
                 "Short sentences and semantic paraphrases are not fully covered.",
    }
