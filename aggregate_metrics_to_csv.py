import csv
import os


from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


modes = ['dense32', 'dense64', 'sparse32', 'sparse64']
fields = []
rows = []

for mode in modes:
    filepath = f"metrics_{mode}.txt"
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = {}
            for line in f:
                if ':' in line:
                    key, val = line.split(':')
                    data[key.strip()] = val.strip()
            if not fields:
                fields = ['Mode'] + list(data.keys())
            rows.append([mode] + [data.get(k, "") for k in fields[1:]])

# Save as CSV
with open("metrics_summary.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(rows)

print("Saved: metrics_summary.csv")

###

modes = ['dense32', 'dense64', 'sparse32', 'sparse64']
fieldnames = []
data = []

for mode in modes:
    filename = f"metrics_{mode}.txt"
    if not os.path.exists(filename):
        continue

    with open(filename) as f:
        lines = f.readlines()

    entry = {'mode': mode}
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            key = parts[0].strip().replace(" (MB)", "")
            value = parts[1].strip().split("\\n")[0]
            try:
                value = float(value)
            except:
                pass
            entry[key] = value
            if key not in fieldnames:
                fieldnames.append(key)
    data.append(entry)

with open("metrics_summary_patched.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mode"] + fieldnames)
    writer.writeheader()
    writer.writerows(data)
