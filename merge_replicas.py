from analyse import *
import time
import itertools
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--cwd',nargs='?',const='.', type=str)
parser.add_argument('--dataset',nargs='?',const='', type=str)
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--cycle',nargs='?',const='', type=int)
parser.add_argument('--replicas',nargs='?',const='', type=int)
parser.add_argument('--discard_first_nframes',nargs='?',const='', type=int)
args = parser.parse_args()

def centerDCD(cwd,dataset,name,cycle, replicas, discard_first_nframes):
    residues = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv').set_index('one', drop=False)
    prot = pd.read_pickle(f'{cwd}/{dataset}/allproteins.pkl').loc[name]
    top = md.Topology()
    chain = top.add_chain()
    for resname in prot.fasta:
        residue = top.add_residue(residues.loc[resname,'three'], chain)
        top.add_atom(residues.loc[resname,'three'], element=md.element.carbon, residue=residue)
    for i in range(len(prot.fasta)-1):
        top.add_bond(top.atom(i),top.atom(i+1))
    traj = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/0.dcd",top=top)[discard_first_nframes:]
    for i in range(1,replicas):
        t = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/{i}.dcd",top=top)[discard_first_nframes:]
        traj = md.join([traj,t])
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0,0]/2
    print(f'Number of frames: {traj.n_frames}')
    traj.save_dcd(f'{cwd}/{dataset}/{prot.path}/{name}.dcd')
    traj[0].save_pdb(f'{cwd}/{dataset}/{prot.path}/{name}.pdb')
    # for i in range(replicas):
        # os.remove(f'{cwd}/{prot.path}/{i}.dcd')

starttime = time.time()  # begin timer
centerDCD(args.cwd, args.dataset, args.name, args.cycle, args.replicas, args.discard_first_nframes)
endtime = time.time()  # end timer
target_seconds = endtime - starttime  # total used time
print(f"{args.name} total merging used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")
