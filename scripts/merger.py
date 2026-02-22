"""Result merging with normalization and weighted combination."""


def merge_results(
    keyword_results: list[dict],
    vector_results: list[dict],
    keyword_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """Normalize, weight, deduplicate, sort by combined score."""
    if not keyword_results and not vector_results:
        return []

    def normalize(results: list[dict]) -> list[dict]:
        if not results:
            return []
        max_score = max(r["score"] for r in results)
        if max_score <= 0:
            return results
        return [{**r, "score": r["score"] / max_score} for r in results]

    norm_kw = normalize(keyword_results)
    norm_vec = normalize(vector_results)

    merged: dict[str, dict] = {}

    for r in norm_kw:
        key = f"{r['doc_id']}:{r['chunk_id']}"
        merged[key] = {**r, "score": r["score"] * keyword_weight, "source": "keyword"}

    for r in norm_vec:
        key = f"{r['doc_id']}:{r['chunk_id']}"
        if key in merged:
            merged[key]["score"] += r["score"] * vector_weight
            merged[key]["source"] = "hybrid"
        else:
            merged[key] = {**r, "score": r["score"] * vector_weight, "source": "vector"}

    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return results
