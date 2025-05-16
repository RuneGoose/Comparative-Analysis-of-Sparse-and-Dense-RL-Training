#!/bin/bash
#SBATCH --job-name=dense64
#SBATCH --output=dense64_%j.out
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --time=33:00:00
#SBATCH --mem=16G

module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2

# Check for venv
if [ ! -d "myenv" ]; then
    echo "Creating virtualenv..."
    python -m venv myenv
    source myenv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source myenv/bin/activate
fi

mode=dense64
echo "Starting training for mode: $mode"
./myenv/bin/python training_main.py --mode $mode --log-interval 10 > output_${mode}_$(date +%Y%m%d_%H%M%S).log
echo "Completed training for $mode"
./myenv/bin/python aggregate_metrics_to_csv.py
./myenv/bin/python plot_extended_metrics.py
sed "s/log_file = .*/log_file = 'epoch_metrics_${mode}.csv'/" plot_training_reward.py > temp_plot_${mode}.py
./myenv/bin/python temp_plot_${mode}.py
rm temp_plot_${mode}.py