To run on Zaratan:
1. Ensure the following files are uploaded to directory:
      1. requirements.txt
      2. training_main.py
      3. training_main_resumable.py
      4. training_metrics_logger.py
      5. utils.py
      6. plot_training_reward.py
      7. plot_extended_metrics.py
      8. aggregate_metrics_to_csv.py
      9. main_job.sh
2. Change epochs in "training_main.py" to test out a quick run
3. if running a batch job, main_job.sh will set up venv and run training
4. if using on-demand bash, run the following commands:

       srun --partition=gpu --gpus=a100_1g.5gb:1 --time=03:00:00 --mem=16G --pty bash
       module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2
       python -m venv myenv
       source myenv/bin/activate
       pip install --upgrade pip
       pip install -r requirements.txt
       for mode in dense32 dense64 sparse32 sparse64
       do
           echo "==============================="
           echo "Starting training mode: $mode"
           echo "==============================="

           ./myenv/bin/python training_main.py --mode $mode --log-interval 10 > output_${mode}_$(date +%Y%m%d_%H%M%S).log
           echo "Completed training for $mode"

           ./myenv/bin/python aggregate_metrics_to_csv.py
           ./myenv/bin/python plot_extended_metrics.py
           sed "s/log_file = .*/log_file = 'epoch_metrics_${mode}.csv'/" plot_training_reward.py > temp_plot_${mode}.py
           ./myenv/bin/python temp_plot_${mode}.py
           rm temp_plot_${mode}.py

           echo "Completed mode: $mode"
           echo "Output saved to: output_$mode.log"
       
       done

       ./myenv/bin/python aggregate_metrics_to_csv.py

       ./myenv/bin/python plot_extended_metrics.py


        
   
