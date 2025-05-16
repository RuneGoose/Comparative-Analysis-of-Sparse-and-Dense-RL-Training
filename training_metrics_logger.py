import csv
import os


from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def init_csv_log(mode):
    filename = f"epoch_metrics_{mode}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "raw_score", "moving_avg_score"])
    return filename

def log_epoch_metrics(filename, epoch, raw_score, moving_avg_score):
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, raw_score, moving_avg_score])
