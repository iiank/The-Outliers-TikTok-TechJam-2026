import json

'''
Parses the 50k catalog.jsonl for each json into a significantly smaller json
Less tokens for the reranker
Included price and average_rating for future potential use case
'''

template = "Title: {title} | Brand: {store} | Category: {leaf_categories} | Features: {top_features}"

def truncate_categories(categories: list[str]) -> str:
    if not categories:
        return ""
    # Filter generic top-level departments if present
    filtered = [
        c
        for c in categories
        if c.lower() not in {"clothing, shoes & jewelry", "all departments"}
    ]
    # Retain the last 2 leaf categories
    tail = filtered[-2:] if len(filtered) >= 2 else filtered
    return " > ".join(tail)

def truncate_features(features: list[str], max_bullets: int = 2) -> str:
    if not features:
        return ""
    extracted = []
    for bullet in features:
        # Drop marketing bullets with long commentary or advertising colons
        clean_b = bullet.split(":")[0].strip() if ":" in bullet else bullet
        # Keep short technical specifications
        if len(clean_b.split()) <= 8:
            extracted.append(clean_b)
        if len(extracted) >= max_bullets:
            break

    # Fallback to the first bullet truncated if no clean short bullet is found
    if not extracted and features:
        extracted.append(" ".join(features[0].split()[:10]))

    return "; ".join(extracted)

with open("catalog.jsonl", "r", encoding="utf-8") as infile, open("reranker_catalog.jsonl", "w") as outfile:
    for line in infile:
        if not line.strip():
            continue

        product = json.loads(line)
        title = product.get("title", "")
        store = product.get("store", "")
        price = product.get("price")
        avg_rating = product.get("average_rating")
        raw_categories = product.get("categories")
        leaf_categories = truncate_categories(raw_categories)
        top_features = truncate_features(product.get("features"), 2)

        output = {
            "parent_asin": product.get("parent_asin"),
            "title": title,
            "document": template.format(
                title=title,
                store=store,
                leaf_categories=leaf_categories,
                top_features=top_features
            ),
            "price": price,
            "category": raw_categories,
            "average_rating": avg_rating
        }
        
        outfile.write(json.dumps(output) + "\n")
