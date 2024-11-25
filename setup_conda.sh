wget /root/https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash /root/Anaconda3-2024.02-1-Linux-x86_64.sh -b
/root/anaconda3/bin/conda init && source /root/.bashrc
/root/anaconda3/bin/conda create -n llm3 python=3.10 -y

#eval "$(/root/anaconda3/bin/conda shell.bash hook)"
#conda activate llm3
#
#/root/anaconda3/bin/conda init
#
#source /root/anaconda3/etc/profile.d/conda.sh
#
#source activate base
#
#/root/anaconda3/bin/conda activate llm

#pip install -r requirements.txt

