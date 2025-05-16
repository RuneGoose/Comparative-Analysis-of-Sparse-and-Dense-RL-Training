import matplotlib.pyplot as plt
import re
import os


from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logfiles = ["epoch_metrics_dense32.csv", "epoch_metrics_dense64.csv", "epoch_metrics_sparse32.csv", "epoch_metrics_sparse64.csv"]

for i in range(len(logfiles)):
    # Set this to the appropriate file before running
    log_file = logfiles[i]
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"File {log_file} not found.")
    
    mode = log_file.split("_")[-1].replace(".csv", "")
    
    episodes = []
    scores = []
    moving_avg = []
    
    with open(log_file, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                episodes.append(int(parts[0]))
                scores.append(float(parts[1]))
                moving_avg.append(float(parts[2]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, scores, label="Episode Score", alpha=0.6)
    plt.plot(episodes, moving_avg, label="Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Time ({mode})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_reward_{mode}_{timestamp}.png")
    plt.show()
