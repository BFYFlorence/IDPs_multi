from PDBedit import PDBedit
from simulate import *
from analyse import *

def force_constants():
    pae_inv = load_pae("/groups/sbinlab/fancao/IDPs_multi/PNt_pae35_368.json")
    print(pae_inv[-1])
    seq_len = 334
    pae_inv = pae_inv[34:34+seq_len][:,34:34+seq_len]

    # pdbedit = PDBedit()
    # plddt, fasta = list(pdbedit.readPDB_singleChain("/groups/sbinlab/fancao/IDPs_multi/AF-Q546U4-F1-model_v4.pdb").plddt)[34:34+seq_len]

def test_ENM():
    name = "Gal3"
    cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory

    cycle = 0  # current training cycles
    replicas = 20  # nums of replica for each sequence
    pfname = 'CPU'  # platform to simulate
    batch_sys = 'SLURM'  # schedule system
    ffasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
    calvados_version = 2  # which version of calvados to use
    discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded when merging replicas
    cutoff = 2.4  # cutoff for the nonionic interactions
    runtime = 20  # hours (overwrites steps if runtime is > 0)
    nframes = 200  # total number of frames to keep (exclude discarded frames)
    kb = 8.31451E-3  # unit:KJ/(mol*K);

    use_pdb = True
    use_hnetwork = True
    use_ssdomains = True

    input_pae = decide_best_pae(cwd, name)

    fpdb = f'{cwd}/af2pre/{name}/ranked_0.pdb'  # af2 predicted structure
    fdomains = f'{cwd}/domains.yaml'
    slab = False  # slab simulation parameters

    proteinsRgs = initProteinsRgs(cycle)
    proteinsRgs['N'] = proteinsRgs['fasta'].apply(lambda x: len(x))
    prot = proteinsRgs.loc[name]
    replica = 0
    N_res = prot.N
    L = int(np.ceil((N_res - 1) * 0.38 + 4))
    N_save = 260  # interval
    N_steps = 520
    k_restraint = kb*float(prot.temp)

    if not os.path.isdir(f"{cwd}/{name}/{cycle}"):
        os.system(f"mkdir -p {cwd}/{name}/{cycle}")

    config_data = dict(flib=cwd, calvados_version=calvados_version, pfname=pfname, name=name, ffasta=ffasta,
                       temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, replica=replica,
                       cutoff=cutoff, L=L, wfreq=N_save, slab=slab, steps=N_steps, use_pdb=use_pdb, fpdb=fpdb,
                       use_hnetwork=use_hnetwork, fdomains=fdomains, use_ssdomains=use_ssdomains,
                       input_pae=input_pae, k_restraint=k_restraint, runtime=runtime,
                       seq=prot.fasta)

    with open(f"{cwd}/{name}/{cycle}/{name}_config_{replica}.yaml",'w') as stream:
        yaml.dump(config_data,stream)

    with open(f"{cwd}/{name}/{cycle}/{name}_config_{replica}.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    simulate(config, True, 1)


if __name__ == '__main__':
    test_ENM()
    # force_constants()





