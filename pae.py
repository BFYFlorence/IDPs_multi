import os

import pandas as pd

from simulate import *
from analyse import *
import protein_repo

def force_constants():
    pae_inv = load_pae("/groups/sbinlab/fancao/IDPs_multi/PNt_pae35_368.json")
    print(pae_inv[-1])
    seq_len = 334
    pae_inv = pae_inv[34:34+seq_len][:,34:34+seq_len]

    # pdbedit = PDBedit()
    # plddt, fasta = list(pdbedit.readPDB_singleChain("/groups/sbinlab/fancao/IDPs_multi/AF-Q546U4-F1-model_v4.pdb").plddt)[34:34+seq_len]

def test_sim():
    # Com :      /home/people/fancao/IDPs_multi
    # Deic:      /groups/sbinlab/fancao/IDPs_multi
    cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory

    dataset = "test"  # onlyIDPs, IDPs_allmultidomain, test,
    name = "Hst5"
    allproteins = pd.read_pickle("/groups/sbinlab/fancao/IDPs_multi/test/allproteins.pkl")
    prot = allproteins.loc[name]
    cycle = 0  # current training cycles
    replica = 0
    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv').set_index('three')

    with open(f"{cwd}/{dataset}/{name}/{cycle}/config_{replica}.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    simulate(config, True, 1)

    """residues = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('one', drop=False)
    top = md.Topology()
    chain = top.add_chain()
    for resname in prot.fasta:
        residue = top.add_residue(residues.loc[resname, 'three'], chain)
        top.add_atom(residues.loc[resname, 'three'], element=md.element.carbon, residue=residue)
    for i in range(len(prot.fasta) - 1):
        top.add_bond(top.atom(i), top.atom(i + 1))
    traj = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/0.dcd", top=top)
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0, 0] / 2
    print(f'Number of frames: {traj.n_frames}')
    traj.save_dcd(f'{cwd}/{dataset}/{prot.path}/{name}.dcd')
    traj[0].save_pdb(f'{cwd}/{dataset}/{prot.path}/{name}.pdb')"""

    """calcDistSums(cwd, dataset, df, name, prot)
    np.save(f'{cwd}/{dataset}/{name}/{cycle}/{name}_AHenergy.npy',
            calcAHenergy(cwd, dataset, df, allproteins.loc[name]))"""

def test_mer():
    # Com :      /home/people/fancao/IDPs_multi
    # Deic:      /groups/sbinlab/fancao/IDPs_multi
    cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory

    dataset = "test"  # onlyIDPs, IDPs_allmultidomain, test,
    names = ["GS48", "OPN"]
    cycle = 0  # current training cycles

    for name in names:
        os.system(f"srun -w node589 -p sbinlab_ib2 python {cwd}/merge_replicas.py --cwd {cwd} --dataset {dataset} --name {name} --cycle {cycle} --replicas 1 --discard_first_nframes 0")

def test_pulchra():
    # Com :      /home/people/fancao/IDPs_multi
    # Deic:      /groups/sbinlab/fancao/IDPs_multi
    cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory

    dataset = "test"  # onlyIDPs, IDPs_allmultidomain, test,
    names = ["GS48", "OPN"]
    cycle = 0  # current training cycles
    for name in names:
        os.system(f"srun -w node589 -p sbinlab_ib2 python {cwd}/pulchra.py --cwd {cwd} --dataset {dataset} --name {name} --num_cpus 2 --pulchra /groups/sbinlab/fancao/pulchra")

def test_validate():
    # Com :      /home/people/fancao/IDPs_multi
    # Deic:      /groups/sbinlab/fancao/IDPs_multi
    cwd = "/groups/sbinlab/fancao/IDPs_multi"  # current working directory
    os.system(f"srun -w node150 -p sbinlab_gpu -c 10 --mem 6GB python3 {cwd}/validate.py")

if __name__ == '__main__':
    test_sim()
    # test_mer()
    # test_pulchra()
    # test_validate()