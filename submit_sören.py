########################################################################################################################
# This project is used to test whether Sören's simualtion scripts have a good agreement with Giulio's simulation results
########################################################################################################################
import subprocess
from jinja2 import Template
import os

from misc_tools import *
from analyse import *

flib = '/groups/sbinlab/fancao/IDPs_multi'  # Folder of master scripts
def write_config(flib, fbase,config,fconfig):
    with open(f'{flib}/{fbase}/{fconfig}','w') as stream:
        yaml.dump(config,stream)

def write_job(name, cycle, replica, temp,flib,fconfig):
    submission = Template("""#!/bin/bash
#SBATCH --job-name={{name}}_{{replica}}
#SBATCH --partition=sbinlab_ib
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=node156
#SBATCH -t 30:00:00
#SBATCH -o {{name}}/{{cycle}}/out_{{replica}}
#SBATCH -e {{name}}/{{cycle}}/err_{{replica}}

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

echo $PWD

echo $SLURM_CPUS_PER_TASK

echo $SLURM_CPUS_ON_NODE

python3 {{flib}}/simulate.py --config {{flib}}/{{name}}/{{cycle}}/{{fconfig}}""")
    with open(f'{flib}/{name}/{cycle}/job_{replica}.sh', 'w') as submit:
        submit.write(submission.render(name=name, cycle=cycle, replica=replica, temp=f'{temp:d}',flib=flib,fconfig=fconfig))

batch_sys = 'SLURM'
pfname = 'CPU'
ffasta = f'{flib}/fastabib.fasta'  # no fasta needed if pdb provided
# general simulation settings
calvados_version = 2
cutoff = 2.4  # set the cutoff for the nonionic interactions
# L = 25.0  # box edge length
# wfreq = 10000  # dcd writing frequency
# steps = 10000000  # number of simulation steps
runtime = 20  # hours (overwrites steps if runtime is > 0)

# multidomain settings
use_pdb = False
use_hnetwork = False
use_ssdomains = False
input_pae = None
k_restraint = 700.
fpdb = f'{flib}/pdbfolder' # pdb folder
fdomains = f'{flib}/domains.yaml'

# slab simulation parameters
slab = False

# protein and parameter list
# names = ["THB-C2","Ubq2","Ubq3","Gal3","TIA1","Ubq4","hnRNPA1","C5_C6_C7","hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
# temps = [277,293,293,303,300,293,300,298,300,293,293,293,293,293,293]
# ionics = [0.15,0.33,0.33,0.04,0.1,0.33,0.15,0.28,0.1,0.15,0.15,0.15,0.15,0.15,0.15]

names = ['Hst5']
temps = [298]
ionics = [0.15]

cycle = 0
replicas = 20

# residues = pd.read_csv(f"{flib}/residues.csv").set_index('one', drop=False)
proteins = initProteins(cycle)  # PRE data
proteinsRgs = initProteinsRgs(cycle)  # Rg data
allproteins = pd.concat((proteins, proteinsRgs), sort=True)  # merge dataset of PRE and Rg
allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))  # length of sequences
allproteins = allproteins.sort_values('N')  # sort in terms of sequence length


for name, temp, ionic in zip(names,temps,ionics):
    for replica in range(replicas):
        fconfig = f'config_{replica}.yaml'
        prot = allproteins.loc[name]
        if not os.path.isdir(f"{flib}/{prot.path}"):
            os.system(f"mkdir -p {flib}/{prot.path}")

        N_res = prot.N
        L = int(np.ceil((N_res - 1) * 0.38 + 4))
        N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
        N_steps = 4 * N_save

        fbase = f'{name}/{cycle}'
        subprocess.run(f'mkdir -p {flib}/{fbase}',shell=True)
        config = dict(flib=flib,calvados_version=calvados_version,pfname=pfname,
            name=name,temp=temp,ionic=ionic,cycle=cycle,replica=replica,
            cutoff=cutoff,L=L,wfreq=N_save,steps=N_steps,
            use_pdb=use_pdb,fpdb=fpdb,use_hnetwork=use_hnetwork,use_ssdomains=use_ssdomains,input_pae=input_pae,
            k_restraint=k_restraint,fdomains=fdomains,
            slab=slab,runtime=runtime,ffasta=ffasta)
        write_config(flib, fbase,config,fconfig=fconfig)
        write_job(name, cycle, replica, temp, flib, fconfig)
        subprocess.run(["sbatch", f'{flib}/{fbase}/job_{replica}.sh'])
    break
