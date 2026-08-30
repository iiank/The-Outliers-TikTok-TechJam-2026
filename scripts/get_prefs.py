import json

'''
get list of all preferences in public_set
'''
all_prefs = []

with open("public_set.jsonl", "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue

        user = json.loads(line)
        profile = user.get("user_profile", "")
        prefs = profile.get("preference_tags", "")
        for pref in prefs:
            if pref not in all_prefs:
                all_prefs.append(pref)
    
print(all_prefs)