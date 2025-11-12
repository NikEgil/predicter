import json


with open("data copy.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
seen = set()

for entry in data:
    entry["sens"]["id_s"] = f'{int(entry["sens"]["id_s"]):08d}'

    port1 = entry.get("port_1", {})
    if "temperature" in port1:
        port1["temp_air"] = port1.pop("temperature")
    if "humidity" in port1:
        port1["hum_air"] = port1.pop("humidity")
    port1["address"] = "36"

    port2 = entry.get("port_2", {})
    port2["address"] = "6"

    entry_str = json.dumps(entry, sort_keys=True)
    if entry_str not in seen:
        seen.add(entry_str)
        cleaned.append(entry)

with open("cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)
