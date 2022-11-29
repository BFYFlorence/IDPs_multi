########################################################################################################################
#                                  import modules                                                                     #
########################################################################################################################
# before using this script, make sure Deic:/groups/sbinlab/fancao/IDPs_multi/af2pre dir is copied into IDPs_multi dir
from utils import *
import subprocess
from analyse import *
import os
from collections import defaultdict
import MDAnalysis
import time

########################################################################################################################
#                              general simulation details                                                        #
########################################################################################################################
cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory
dataset = "IDPs_allmultidomain"  # onlyIDPs, IDPs_allmultidomain,
cycle = 0  # current training cycles
replicas = 20  # nums of replica for each sequence
pfname = 'CPU'  # platform to simulate
batch_sys = 'SLURM'  # schedule system
calvados_version = 2  # which version of calvados to use
discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded when merging replicas
cutoff = 2.4  # cutoff for the nonionic interactions, nm
runtime = 20  # hours (overwrites steps if runtime is > 0)
nframes = 200  # total number of frames to keep (exclude discarded frames)

########################################################################################################################
#                              multidomain simulation details                                                    #
########################################################################################################################
kb = 8.31451E-3  # unit:KJ/(mol*K);
k_restraint = 700.  # unit:KJ/(mol*nm^2);
fdomains = f'{cwd}/domains.yaml'
slab = False  # slab simulation parameters
# protein and parameter list
multidomain_names = ["THB_C2","Ubq2","Ubq3","Gal3","TIA1","Ubq4","hnRNPA1","C5_C6_C7","hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
# temps = [277,293,293,303,300,293,300,298,300,293,293,293,293,293,293]
# ionics = [0.15,0.33,0.33,0.04,0.1,0.33,0.15,0.28,0.1,0.15,0.15,0.15,0.15,0.15,0.15]

########################################################################################################################
#                                             setup simulations                                                        #
########################################################################################################################
if not os.path.isdir(f"{cwd}/{dataset}"):
    os.system(f"mkdir -p {cwd}/{dataset}")
if cycle == 0:
    r = pd.read_csv(f'{cwd}/residues_{cycle-1}.csv').set_index('one')
    r.lambdas = 0.5
    r = r[['three', 'MW', 'lambdas', 'sigmas', 'q']]
    r.to_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv')

proteinsPRE = initProteinsPRE(cycle)
proteinsRgs = initProteinsRgs(cycle)
allproteins = pd.concat((proteinsPRE,proteinsRgs),sort=True)
allproteins['N'] = allproteins['fasta'].apply(lambda x : len(x))
allproteins = allproteins.sort_values('N')
proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl')
allproteins.to_pickle(f'{cwd}/{dataset}/allproteins.pkl')
print("Properties used:\n", pd.read_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv'))

# simulate
jobid_1 = defaultdict(list)
for name, prot in allproteins.iterrows():
    if name in multidomain_names:
        ffasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
        input_pae = decide_best_pae(cwd, name)
        fpdb = f'{cwd}/af2pre/{name}/ranked_0.pdb'  # af2 predicted structure
        use_pdb = True
        use_hnetwork = True
        use_ssdomains = True
    else:
        ffasta = ""  # no fasta needed if pdb provided
        input_pae = None
        fpdb = ""  # af2 predicted structure
        use_pdb = False
        use_hnetwork = False
        use_ssdomains = False

    jobid_name = []
    if not os.path.isdir(f"{cwd}/{dataset}/{name}/{cycle}"):
        os.system(f"mkdir -p {cwd}/{dataset}/{name}/{cycle}")
    for replica in range(replicas):
        if (not os.path.isfile(f"{cwd}/{dataset}/{prot.path}/{replica}.dcd")) or len(MDAnalysis.coordinates.DCD.DCDReader(
                f"{cwd}/{dataset}/{name}/{cycle}/{replica}.dcd")) != int(nframes + discard_first_nframes):
            config_filename = f'config_{replica}.yaml'
            N_res = prot.N
            L = int(np.ceil((N_res - 1) * 0.38 + 4))
            N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
            N_steps = (nframes + discard_first_nframes) * N_save
            cpu_num = 1
            config_data = dict(flib=cwd, calvados_version=calvados_version, pfname=pfname, name=name, ffasta=ffasta,
                               temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, replica=replica,
                               cutoff=cutoff, L=L, wfreq=N_save, slab=slab, steps=N_steps, use_pdb=use_pdb, fpdb=fpdb,
                               use_hnetwork=use_hnetwork, fdomains=fdomains, use_ssdomains=use_ssdomains,
                               input_pae=input_pae, k_restraint=k_restraint, runtime=runtime, seq=prot.fasta,
                               dataset=dataset)
            write_config(cwd, dataset, prot.path, config_data, config_filename=config_filename)
            with open(f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_{replica}.sh", 'w') as submit:
                submit.write(submission_1.render(cwd=cwd, dataset=dataset, name=name,cycle=f'{cycle}', replica=f'{replica}',cutoff=f'{cutoff}',
                                                 overwrite=True, cpu_num=cpu_num))
            proc = subprocess.run(['sbatch', f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_{replica}.sh"],capture_output=True)
            print(proc)
            jobid_name.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
            time.sleep(0.25)

    jobid_1[name] = jobid_name
print(f'Simulating sequences: {jobid_1}')

# merge
jobid_2 = []
for name in allproteins.index:
    prot = allproteins.loc[name]
    if (not os.path.isfile(f"{cwd}/{dataset}/{prot.path}/{name}.dcd")) or len(
            MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{prot.path}/{name}.dcd")) != int(replicas * nframes):
        with open(f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}.sh", 'w') as submit:
            submit.write(submission_2.render(cwd=cwd, dataset=dataset, jobid=jobid_1[name],name=name,cycle=f'{cycle}', replicas=replicas,
                                             discard_first_nframes=discard_first_nframes))
        proc = subprocess.run(['sbatch',f'{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}.sh'],capture_output=True)
        print(proc)
        jobid_2.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
        time.sleep(0.25)

# optimize
with open(f'{cwd}/{dataset}/opt_{cycle}.sh', 'w') as submit:
    submit.write(submission_3.render(cwd=cwd, dataset=dataset, jobid=jobid_2,proteins=' '.join(proteinsPRE.index),
                 cycle=f'{cycle}',cutoff=f'{cutoff}'))
subprocess.run(['sbatch',f'{cwd}/{dataset}/opt_{cycle}.sh'])