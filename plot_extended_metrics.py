import matplotlib.pyplot as plt
import os

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

modes = ['dense32', 'dense64', 'sparse32', 'sparse64']
metricsa = {}

for mode in modes:
    filepath = f'metrics_{mode}.txt'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = {line.split(':')[0].strip(): float(line.split(':')[1].strip()) for line in f if ':' in line}
            metricsa[mode] = lines

# Metrics to plot
metric_keys = [
    ('Training time (s)', 'Training Time (s)'),
    ('CPU Peak Memory (MB)', 'CPU Peak Memory (MB)'),
    ('CPU RSS (MB)', 'CPU RSS (MB)'),
    ('GPU Max Allocated Memory (MB)', 'GPU Max Allocated (MB)'),
    ('GPU Max Reserved Memory (MB)', 'GPU Max Reserved (MB)'),
    ('Final moving average score', 'Final Score')
]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (key, label) in zip(axes, metric_keys):
    values = [metricsa[m].get(key, 0) for m in modes if m in metricsa]
    ax.bar(modes[:len(values)], values)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.set_xlabel('Mode')

plt.tight_layout()
plt.savefig(f"comparison_extended_metrics_{timestamp}.png")
plt.show()

###

metrics = {
    'Training time (s)': [],
    'GPU Max Allocated Memory (MB)': [],
    'GPU Max Reserved Memory (MB)': [],
    'Final moving average score': []
}

for mode in modes:
    filename = f"metrics_{mode}.txt"
    if not os.path.exists(filename):
        continue

    with open(filename) as f:
        content = f.read().replace('\\n', '\n').splitlines()

    values = {}
    for line in content:
        if ':' in line:
            k, v = line.split(':', 1)
            try:
                values[k.strip()] = float(v.strip())
            except:
                continue

    for key in metrics:
        metrics[key].append(values.get(key, 0))

plt.figure(figsize=(12, 8))
for i, (metric, vals) in enumerate(metrics.items(), 1):
    plt.subplot(len(metrics), 1, i)
    plt.bar(modes, vals)
    plt.title(metric)
    plt.ylabel(metric)
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"final_metrics_comparison_{timestamp}.png")
plt.show()