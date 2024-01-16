import argparse
import csv
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiments_file",
    dest="experiments_file",
    help="points to csv with experiments and associated parameters",
    default=os.path.join("slurm", "example_experiments.csv")
    )

parser.add_argument(
    "--slurm_directory",
    dest="slurm_directory",
    help="where the sbatch files and slurm outputs should be stored",
    default=os.path.join("slurm", "slurm_files")
    )

parser.add_argument(
    "--submit_jobs",
    dest="submit_jobs",
    help="when specified, the jobs will be submitted after supporting files are created",
    action="store_true"
    )

parser.add_argument(
    "--experiment_tag",
    dest="experiment_tag",
    help="string appended to the experiment in case you want to run new trials with the same setup",
    default = "test"
    )
    
args = parser.parse_args()

os.makedirs(args.slurm_directory, exist_ok=True)

experiments=[]
with open(args.experiments_file, 'r') as experiments_configs:
    csv_reader=csv.reader(experiments_configs)
    header = next(csv_reader)
    for experiment_config in csv_reader:
        experiments.append(experiment_config)

experiments = [dict(zip(header, experiment)) for experiment in experiments]

print(experiments[0])

for experiment in experiments:
    #make folder for experiment
    experiment['experiment'] = experiment['experiment'] + args.experiment_tag
    experiment_directory = os.path.join(args.slurm_directory, experiment['experiment'])
    os.makedirs(experiment_directory, exist_ok=True)
    
    #file for experiement
    experiment_sbatch_file_path = os.path.join(experiment_directory, experiment['experiment'] + '.sh')
        
    #populate file with sbatch commands/script
    with open(experiment_sbatch_file_path, 'w') as file:
        file.writelines("#!/bin/bash\n\n")
        
        file.writelines("#SBATCH -N {} # number of nodes\n".format(experiment['nodes_requested']))
        file.writelines("#SBATCH -c {} # number of cores\n".format(experiment['cores_requested']))
        file.writelines("#SBATCH -p general # partition\n")
        file.writelines("#SBATCH -G {} # GPU's\n".format(experiment['gpus_requested']))
        file.writelines("#SBATCH -t {} # time in d-hh:mm:ss\n".format(experiment['time_requested']))
        file.writelines("#SBATCH -q public # QOS\n")
        file.writelines("#SBATCH -o {}.%j.out # file to save job's STDOUT (%j = JobId)\n".format(os.path.join(experiment_directory, 'slurm')))
        file.writelines("#SBATCH -o {}.%j.err # file to save job's STDERR (%j = JobId)\n".format(os.path.join(experiment_directory, 'slurm')))
        file.writelines("#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails\n")   
        file.writelines("#SBATCH --export=NONE   # Purge the job-submitting shell environment\n\n")
        
        file.writelines("# Load required modules for job's environment\n")
        file.writelines("module load mamba/latest\n")
        file.writelines("# Using python, so source activate an appropriate environment\n")
        file.writelines("source activate BenchmarkingTransformers\n")
        file.writelines("#navigate to project directory\n")
        file.writelines("cd /data/jliang12/ayanan/Projects/BenchmarkCNNvTransformer \n\n")


        file.writelines("python main_classification.py \\\n")
        file.writelines("\t --data_set {} \\\n".format(experiment['data_set']))
        file.writelines("\t --model {} \\\n".format(experiment['model']))
        file.writelines("\t --init {} \\\n".format(experiment['init']))
        file.writelines("\t --data_dir {} \\\n".format(experiment['data_dir']))
        file.writelines("\t --train_list {} \\\n".format(experiment['train_list']))
        file.writelines("\t --test_list {} \\\n".format(experiment['test_list']))
        file.writelines("\t --val_list {} \\\n".format(experiment['val_list']))
        file.writelines("\t --lr {} \\\n".format(experiment['lr']))
        file.writelines("\t --opt {} \\\n".format(experiment['opt']))
        file.writelines("\t --epochs {} \\\n".format(experiment['epochs']))
        file.writelines("\t --warmup-epochs {} \\\n".format(experiment['warmup-epochs']))
        file.writelines("\t --batch_size {} \\\n".format(experiment['batch_size']))
        file.writelines("\t --exp_name {} \\\n".format(experiment['experiment']))
        file.writelines("\t --GPU {} \\\n".format(experiment['GPU']))
        file.writelines("\t --workers {} \\\n".format(experiment['workers']))
        file.writelines("\t --models_dir {} \\\n".format(experiment['models_dir']))
        file.writelines("\t --resume true")
        
    #submit sbatch script
    if args.submit_jobs:
        os.system("sbatch {}".format(experiment_sbatch_file_path))