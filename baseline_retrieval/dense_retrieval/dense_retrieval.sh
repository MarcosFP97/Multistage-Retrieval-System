#!/bin/bash
#SBATCH --job-name=dense_retr            # Job name
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=1            # -c Run 1 processor per task       
#SBATCH --gpus=2
##SBATCH --constraint=cpu_amd     
#SBATCH --mem=10gb                    # Job memory request
#SBATCH --time=4-04:00:00              # Time limit hrs:min:sec
#SBATCH --qos=regular                 # Cola
#SBATCH --output=log_%x_%j.log       # Standard output and error log

#conda install -c conda-forge pytorch faiss-gpu
#pip install typer pandas nltk pyserini querybuilder
#conda install pytorch cudatoolkit=11.3 -c pytorch

echo "starting job from $(hostname)"
module load "Java"
sudo mount_image.py /mnt/beegfs/groups/irgroup/software/image.ext4 --mount-point image --rw
rm /mnt/beegfs/groups/irgroup/software/image
ln -s /mnt/imagenes/marcos.fernandez.pichel/image /mnt/beegfs/groups/irgroup/software/image
conda init bash
source ~/.bashrc
conda activate
cd /mnt/beegfs/groups/irgroup
python /mnt/beegfs/home/marcos.fernandez.pichel/irgroup/trec-pipeline/1_baseline_retrieval/dense_retrieval/baseline_retrieval.py --out-dir /mnt/beegfs/home/marcos.fernandez.pichel/irgroup/trec-pipeline/runs/dense_retrieval
bash /mnt/beegfs/groups/irgroup/software/umount_image.sh


