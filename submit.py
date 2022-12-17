########################################################################################################################
#                                  import modules                                                                     #
########################################################################################################################
# before using this script, make sure Deic:/groups/sbinlab/fancao/IDPs_multi/af2pre dir is copied into IDPs_multi dir
import test_idps
from utils import *
import subprocess
from analyse import *
import os
from collections import defaultdict
import MDAnalysis
import time
from protein_repo import get_ssdomains

########################################################################################################################
#                              general simulation details                                                        #
########################################################################################################################
cwd_dict = {"Computerome": "/home/people/fancao/IDPs_multi", "Deic": "/groups/sbinlab/fancao/IDPs_multi"}
exclusion_GS_dict = {"IDPs_allmultidomain": False, "IDPs_multidomainExcludeGS": True}

validate = False
batch_sys = 'Deic'  # schedule system, Deic, Computerome
cwd = cwd_dict[batch_sys]  # current working directory
dataset = "IDPs_multidomainExcludeGS"  # onlyIDPs, IDPs_allmultidomain, test, IDPs_multidomainExcludeGS
exclusion_GS = exclusion_GS_dict[dataset]
cycle = 6  # current training cycles
replicas = 20  # nums of replica for each sequence
"""Index(['Hst5', 'Hst52', 'p532070', 'ACTR', 'Ash1', 'CTD2', 'Sic1', 'SH4UD',
       'ColNT', 'p15PAF', 'hNL3cyt', 'RNaseA', 'M10R', 'M9FP3Y', 'M9FP6Y',
       'P7FM7Y', 'M12FP12Y', 'A1', 'M4D', 'P4D', 'P2R', 'P7R', 'M3RP3K', 'M6R',
       'M8FP4Y', 'M10RP10K', 'P8D', 'P12D', 'M10FP7RP12D', 'M12FP12YM10R',
       'P7KP12Db', 'P7KP12D', 'M6RP6K', 'P12E', 'THB_C2', 'aSyn140', 'aSyn',
       'FhuA', 'Ubq2', 'FUS', 'FUS12E', 'K27', 'K10', 'K25', 'K32', 'OPN',
       'CAHSD', 'Ubq3', 'Gal3', 'K23', 'tau35', 'CoRNID', 'TIA1', 'K44',
       'Ubq4', 'hnRNPA1', 'C5_C6_C7', 'PNtS5', 'PNtS4', 'PNtS1', 'PNt',
       'PNtS6', 'GHRICD', 'hSUMO_hnRNPA1', 'GS0', 'GS8', 'GS16', 'GS24',
       'GS32', 'GS48'],
      dtype='object')"""
if not exclusion_GS:
    gpu_proteins = ["hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
else:
    gpu_proteins = ["C5_C6_C7", "PNtS5", "PNtS4", "PNtS1", "PNt", "PNtS6", "GHRICD", "hSUMO_hnRNPA1"]

gpu = False
gpu_id = -1
cpu_num = 1
calvados_version = 2  # which version of calvados to use
discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded when merging replicas
cutoff = 2.4  # cutoff for the nonionic interactions, nm
runtime = 20  # hours (overwrites steps if runtime is > 0)
nframes = 200  # total number of frames to keep for each replica (exclude discarded frames)

########################################################################################################################
#                              multidomain simulation details                                                    #
########################################################################################################################
# kb = 8.31451E-3  # unit:KJ/(mol*K);
k_restraint = 700.  # unit:KJ/(mol*nm^2);
fdomains = f'{cwd}/domains.yaml'
slab = False  # slab simulation parameters
# protein and parameter list
multidomain_names = ["THB_C2","Ubq2","Ubq3","Gal3","TIA1","Ubq4","hnRNPA1","C5_C6_C7","hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
test_multidomain_names = []
# gpu_proteins = ['hSUMO_hnRNPA1', 'GS0','GS8','GS16','GS24','GS32','GS48']
# temps = [277,293,293,303,300,293,300,298,300,293,293,293,293,293,293]
# ionics = [0.15,0.33,0.33,0.04,0.1,0.33,0.15,0.28,0.1,0.15,0.15,0.15,0.15,0.15,0.15]

########################################################################################################################
#                                             submit simulations                                                       #
########################################################################################################################
if not validate:
    if not os.path.isdir(f"{cwd}/{dataset}"):
        os.system(f"mkdir -p {cwd}/{dataset}")
    if cycle == 0:
        r = pd.read_csv(f'{cwd}/residues_{cycle-1}.csv').set_index('one')
        r.lambdas = 0.5
        r = r[['three', 'MW', 'lambdas', 'sigmas', 'q']]
        r.to_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv')

    proteinsPRE = initProteinsPRE(cycle)
    proteinsRgs = initProteinsRgs(cycle, exclusion_GS)
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
        if name in gpu_proteins:
            pass
            # gpu = True
        if name in multidomain_names:
            path2fasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
            input_pae = decide_best_pae(cwd, name)
            path2pdb = f'{cwd}/extract_relax/{name}_rank0_relax.pdb'  # af2 predicted structure
            use_pdb = True
            use_hnetwork = True
            use_ssdomains = True
        else:
            path2fasta = ""  # no fasta needed if pdb provided
            input_pae = None
            path2pdb = ""  # af2 predicted structure
            use_pdb = False
            use_hnetwork = False
            use_ssdomains = False

        jobid_name = []
        if not os.path.isdir(f"{cwd}/{dataset}/{name}/{cycle}"):
            os.system(f"mkdir -p {cwd}/{dataset}/{name}/{cycle}")

        N_res = prot.N
        L = int(np.ceil((N_res - 1) * 0.38 + 4))
        if name in multidomain_names:
            domain_len = 0
            for domain in get_ssdomains(name, fdomains):
                domain_len += len(domain)
            N_res -= domain_len
        N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
        N_steps = (nframes + discard_first_nframes) * N_save
        print("N_res:", N_res)
        for replica in range(replicas):
            if gpu:
                gpu_id += 1
            config_filename = f'config_{replica}.yaml'
            if (not os.path.isfile(f"{cwd}/{dataset}/{prot.path}/{replica}.dcd")) or len(MDAnalysis.coordinates.DCD.DCDReader(
                    f"{cwd}/{dataset}/{name}/{cycle}/{replica}.dcd")) != int(nframes + discard_first_nframes):
                config_data = dict(cwd=cwd, calvados_version=calvados_version, name=name, dataset=dataset, seq=prot.fasta,
                                   path2fasta=path2fasta, temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle,
                                   replica=replica, cutoff=cutoff, L=L, wfreq=N_save, slab=slab, steps=N_steps, gpu=gpu,
                                   use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains,
                                   use_ssdomains=use_ssdomains, input_pae=input_pae, k_restraint=k_restraint,
                                   runtime=runtime, gpu_id=gpu_id%4)
                write_config(cwd, dataset, prot.path, config_data, config_filename=config_filename)
                with open(f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_{replica}.{'sh' if batch_sys=='Deic' else 'pbs'}", 'w') as submit:
                    if batch_sys == "Deic":
                        submit.write(
                            submission_1.render(cwd=cwd, dataset=dataset, name=name, cycle=f'{cycle}', replica=f'{replica}',
                                                cutoff=f'{cutoff}', overwrite=True, cpu_num=2 if gpu else 1,
                                                node="sbinlab_gpu" if gpu else "sbinlab_ib2"))
                    else:
                        submit.write(
                            submission_5.render(cwd=cwd, dataset=dataset, name=name, cycle=f'{cycle}', replica=f'{replica}',
                                                cutoff=f'{cutoff}', overwrite=True, cpu_num=cpu_num,
                                                node="gpu" if gpu else "thinnode"))

                proc = subprocess.run([f"{'sbatch' if batch_sys=='Deic' else 'qsub'}", f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_{replica}.{'sh' if batch_sys=='Deic' else 'pbs'}"],capture_output=True)
                print(proc)
                jobid_name.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
                time.sleep(0.25)

        jobid_1[name] = jobid_name
    print(f'Simulating sequences: {jobid_1}')

    # merge
    jobid_2 = []
    for name in allproteins.index:
        if len(jobid_1[name]) == 0:
            continue
        prot = allproteins.loc[name]
        if (not os.path.isfile(f"{cwd}/{dataset}/{prot.path}/{name}.dcd")) or len(
                MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{prot.path}/{name}.dcd")) != int(replicas * nframes):
            with open(f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}.{'sh' if batch_sys=='Deic' else 'pbs'}", 'w') as submit:
                submit.write(submission_2.render(cwd=cwd, dataset=dataset, jobid=jobid_1[name],name=name,cycle=f'{cycle}', replicas=replicas,
                                                 discard_first_nframes=discard_first_nframes))
            proc = subprocess.run([f"{'sbatch' if batch_sys=='Deic' else 'qsub'}",f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}.{'sh' if batch_sys=='Deic' else 'pbs'}"],capture_output=True)
            print(proc)
            jobid_2.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
            time.sleep(0.25)

    # optimize
    config_filename = f'config_opt{cycle}.yaml'
    config_data = dict(cwd=cwd, log_path="LOG", dataset=dataset, cycle=cycle, num_cpus=62, cutoff=cutoff, exclusion_GS=exclusion_GS)
    with open(f'{cwd}/{dataset}/{config_filename}','w') as stream:
        yaml.dump(config_data,stream)

    with open(f"{cwd}/{dataset}/opt_{cycle}.{'sh' if batch_sys=='Deic' else 'pbs'}", 'w') as submit:
        submit.write(submission_3.render(cwd=cwd, dataset=dataset, jobid=jobid_2,proteins=' '.join(proteinsPRE.index),
                     cycle=f'{cycle}', path2config=f"{cwd}/{dataset}/{config_filename}"))
    subprocess.run([f"{'sbatch' if batch_sys=='Deic' else 'qsub'}",f"{cwd}/{dataset}/opt_{cycle}.{'sh' if batch_sys=='Deic' else 'pbs'}"])

