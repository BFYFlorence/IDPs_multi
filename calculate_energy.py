from protein_repo import get_ssdomains
import numpy as np
from MDAnalysis.analysis import distances
from analyse import *
import time
import itertools
import os
from argparse import ArgumentParser
import MDAnalysis as mda
import yaml

parser = ArgumentParser()
parser.add_argument('--cwd',nargs='?',const='.', type=str)
parser.add_argument('--dataset',nargs='?',const='', type=str)
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--cycle',nargs='?',const='', type=int)
args = parser.parse_args()

def geometry_from_pdb(pdb):
    """ positions in nm """
    u = mda.Universe(pdb)
    cas = u.select_atoms('name CA')

    cas.translate(-cas.center_of_mass())

    pos = cas.positions / 10.  # nm

    # print(pos.shape)
    return pos

def self_distances(N, pos):
    """ Self distance map for matrix of positions

    Input: Matrix of positions
    Output: Self distance map
    """
    dmap = np.zeros((N, N))
    d = distances.self_distance_array(pos)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            dmap[i, j] = d[k]
            dmap[j, i] = d[k]
            k += 1
    return dmap

def calcAHenergy(cwd,dataset,name, cycle):
    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
    term_1 = np.load(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_1_exclude_domain.npy')
    term_2 = np.load(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_2_exclude_domain.npy')
    unique_pairs = np.load(f'{cwd}/{dataset}/{name}/{cycle}/unique_pairs_exclude_domain.npy')
    print("lambdas = (df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)/2......")
    lambdas = (df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)/2
    return term_1+np.nansum(lambdas*term_2,axis=1)

def calcDistSums(cwd,dataset,name, cycle):
    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
    rc = 2.4
    cs_cutoff = 0.9
    # if not os.path.isfile(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy'):
    traj = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd",f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")
    traj = traj[:2]
    fasta = [res.name for res in traj.top.atoms]
    N = len(fasta)  # number of residues
    input_pdb = f'{cwd}/extract_relax/{name}_rank0_relax.pdb'  # af2 predicted structure
    # pos = geometry_from_pdb(input_pdb)
    pairs = traj.top.select_pairs('all','all')
    fdomains = f'{cwd}/domains.yaml'
    ssdomains = get_ssdomains(name, fdomains)

    dmap = self_distances(N, pos)
    for i in range(N):
        for j in range(i + 1, N):
            if np.abs(i - j) == 1:
                mask.append(False)
            elif ssdomains != None:  # use fixed domain boundaries for network
                ss = False
                for ssdom in ssdomains:
                    if i in ssdom and j in ssdom:
                        ss = True
                if ss:  # both residues in structured domains
                    mask.append(False)
                else:
                    mask.append(True)
            else:
                mask.append(True)
    mask = np.array(mask)

    print(pairs.shape)
    print(mask.shape)

    # exclude bonded and harmonic
    """mask = []
    dmap = self_distances(N, pos)
    for i in range(N):
        for j in range(i+1,N):
            if np.abs(i-j) == 1:
                mask.append(False)
            elif ssdomains != None:  # use fixed domain boundaries for network
                ss = False
                for ssdom in ssdomains:
                    if i in ssdom and j in ssdom:
                        ss = True
                if ss:  # both residues in structured domains
                    if dmap[i,j] < cs_cutoff:  # nm
                        mask.append(False)
                    else:
                        mask.append(True)
                else:
                    mask.append(True)
            else:
                mask.append(True)
    mask = np.array(mask)

    print(pairs.shape)
    print(mask.shape)"""

    # only exclude bonded
    # mask = np.abs(pairs[:,0]-pairs[:,1])>1 # exclude bonds

    pairs = pairs[mask]  # index
    d = md.compute_distances(traj,pairs)
    d[d>rc] = np.inf # cutoff
    r = np.copy(d)
    n1 = np.zeros(r.shape,dtype=np.int8)
    n2 = np.zeros(r.shape,dtype=np.int8)
    pairs = np.array(list(itertools.combinations(fasta,2)))
    pairs = pairs[mask] # exclude bonded  three words
    sigmas = 0.5*(df.loc[pairs[:,0]].sigmas.values+df.loc[pairs[:,1]].sigmas.values)
    for i,sigma in enumerate(sigmas):
        mask = r[:,i]>np.power(2.,1./6)*sigma
        r[:,i][mask] = np.inf # cutoff
        n1[:,i][~mask] = 1
        n2[:,i][np.isfinite(d[:,i])] = 1

    unique_pairs = np.unique(pairs,axis=0)
    pairs = np.core.defchararray.add(pairs[:,0],pairs[:,1])

    d12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),
                          axis=1, arr=np.power(d,-12.))
    d6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),
                          axis=1, arr=-np.power(d,-6.))
    r12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=np.power(r,-12.))
    r6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=-np.power(r,-6.))
    ncut1 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n1)
    ncut2 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n2)
    sigmas = 0.5*(df.loc[unique_pairs[:,0]].sigmas.values+df.loc[unique_pairs[:,1]].sigmas.values)
    sigmas6 = np.power(sigmas,6)
    sigmas12 = np.power(sigmas6,2)
    eps = 0.2*4.184
    term_1 = eps*(ncut1+4*(sigmas6*r6 + sigmas12*r12))
    term_2 = sigmas6*(d6+ncut2/np.power(rc,6)) + sigmas12*(d12+ncut2/np.power(rc,12))
    term_2 = 4*eps*term_2 - term_1
    np.save(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_1_exclude_domain.npy',term_1.sum(axis=1))
    np.save(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_2_exclude_domain.npy',term_2)
    np.save(f'{cwd}/{dataset}/{name}/{cycle}/unique_pairs_exclude_domain.npy',unique_pairs)


cwd = args.cwd
dataset = args.dataset
name = args.name
cycle = args.cycle

calcDistSums(cwd, dataset, name, cycle)
np.save(f'{cwd}/{dataset}/{name}/{cycle}/{name}_AHenergy_exclude_domain.npy',
        calcAHenergy(cwd, dataset, name, cycle))

