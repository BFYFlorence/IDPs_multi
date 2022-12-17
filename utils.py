from PDBedit import PDBedit
import os
from collections import defaultdict
import Bio.PDB.PDBParser
import MDAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
import json
from misc_tools import *
from protein_repo import *
import yaml
from simtk import openmm, unit
import mdtraj as md
import itertools

sasa_max = {
"GLY": 86.2,
"ALA": 109.3,
"ARG": 255.5,  # +1 net charge
"ASN": 171.5,
"ASP": 170.2,  # COO-, because in CALVADOS2, it has -1 net charge
"CYS": 140.1,
"GLN": 189.2,
"GLU": 198.6,  # COO-, because in CALVADOS2, it has -1 net charge
"HIS": 193.8,  # not charged
"ILE": 173.3,
"LEU": 181.6,
"LYS": 215.0,  # +1 net charge
"MET": 193.0,
"PHE": 205.8,
"PRO": 134.3,
"SER": 132.4,
"THR": 148.8,
"TRP": 253.1,
"TYR": 230.5,
"VAL": 148.4,
}  # Angstroms^2, extended GGXGG (G, glycine; X any residue)

#SBATCH --exclusive
submission_1 = Template(
"""#!/bin/bash
#SBATCH --job-name={{name}}_{{replica}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={{cpu_num}}
#SBATCH --partition={{node}}
#SBATCH --mem=1GB
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/out_{{replica}}
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/err_{{replica}}

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

python3 {{cwd}}/simulate.py --config {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/config_{{replica}}.yaml --cpu_num {{cpu_num}} --overwrite {{overwrite}}""")

submission_2 = Template(
"""#!/bin/bash
#SBATCH --job-name={{name}}_{{cycle}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=sbinlab_ib2
#SBATCH --dependency=afterok{% for id in jobid %}:{{id}}{% endfor %}
#SBATCH --mem=10GB
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_out
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

python {{cwd}}/merge_replicas.py --cwd {{cwd}} --dataset {{dataset}} --name {{name}} --cycle {{cycle}} --replicas {{replicas}} --discard_first_nframes {{discard_first_nframes}}""")

submission_3 = Template(
"""#!/bin/bash
#SBATCH --job-name=opt_{{cycle}}
#SBATCH --nodes=1
#SBATCH --partition=sbinlab_ib2
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=62
#SBATCH --dependency=afterok{% for id in jobid %}:{{id}}{% endfor %}
#SBATCH -o {{cwd}}/{{dataset}}/{{cycle}}_out
#SBATCH -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

declare -a proteinsPRE_list=({{proteins}})

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --num_cpus 62 --pulchra /groups/sbinlab/fancao/pulchra
done

python {{cwd}}/optimize.py  --path2config {{path2config}}""")

submission_4 = Template(
"""#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=sbinlab
#SBATCH --mem=50GB
#SBATCH -t 30:00:00
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/out_0
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/err_0

source /groups/sbinlab/fancao/.bashrc

path2python3=~/miniconda3/envs/calvados/bin

echo $SLURM_CPUS_PER_TASK

echo $SLURM_CPUS_ON_NODE

$path2python3/python3 {{cwd}}/calculate_energy.py --cwd {{cwd}} --dataset {{dataset}} --name {{name}} --cycle {{cycle}}""")

submission_5 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{replica}}
#PBS -l nodes=1:ppn=1:{{node}}
#PBS -l walltime=48:00:00
#PBS -l mem=2gb
#PBS -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/out_{{replica}}
#PBS -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/err_{{replica}}

echo $CUDA_VISIBLE_DEVICES

source /home/people/fancao/.bashrc

path2python3=/home/projects/ku_10001/people/fancao/miniconda3/envs/calvados/bin

$path2python3/python3 {{cwd}}/simulate.py --config {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/config_{{replica}}.yaml --cpu_num {{cpu_num}} --overwrite {{overwrite}}""")

submission_6 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{cycle}}
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=2:00:00
#PBS -l mem=10gb
#PBS -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_out
#PBS -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_err

source /home/people/fancao/.bashrc

conda activate calvados

python {{cwd}}/merge_replicas.py --cwd {{cwd}} --dataset {{dataset}} --name {{name}} --cycle {{cycle}} --replicas {{replicas}} --discard_first_nframes {{discard_first_nframes}}""")

submission_7 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N opt_{{cycle}}
#PBS -l nodes=1:ppn=40:thinnode
#PBS -l walltime=45:00:00
#PBS -l mem=188gb
#PBS -o {{cwd}}/{{dataset}}/{{cycle}}_out
#PBS -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /home/people/fancao/.bashrc

conda activate calvados

declare -a proteinsPRE_list=({{proteins}})

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --num_cpus $SLURM_CPUS_ON_NODE --pulchra /groups/sbinlab/fancao/pulchra
done

python {{cwd}}/optimize.py --cwd {{cwd}} --dataset {{dataset}} --log LOG --cycle {{cycle}} --num_cpus $SLURM_CPUS_ON_NODE --cutoff {{cutoff}}""")

def nested_dictlist():
    return defaultdict(list)

def processIon(aa) -> str:  # Dealing with protonation conditions
    if aa in ['ASH']:
        return 'ASP'
    if aa in ['HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP']:
        return 'HIS'
    return aa

def space_data(li, interval=20):  # interval
    y = []
    for i in range(len(li)):
        if i % interval == 0:
            y.append(li[i])
    return y

def write_config(cwd, dataset, fbase,config_data,config_filename):
    with open(f'{cwd}/{dataset}/{fbase}/{config_filename}','w') as stream:
        yaml.dump(config_data,stream)

def set_harmonic_network(N,pos,pae_inv,yu,ah,ssdomains=None,cs_cutoff=0.9,k_restraint=700.):
    cs = openmm.openmm.HarmonicBondForce()
    dmap = self_distances(N,pos)
    for i in range(N-2):
        for j in range(i+2,N):
            if ssdomains != None:  # use fixed domain boundaries for network
                ss = False
                for ssdom in ssdomains:
                    if i+1 in ssdom and j+1 in ssdom:
                        ss = True
                if ss:  # both residues in structured domains
                    if dmap[i,j] < cs_cutoff:  # nm
                        cs.addBond(i, j, dmap[i, j] * unit.nanometer,
                                   k_restraint * unit.kilojoules_per_mole / (unit.nanometer ** 2))
                        yu.addExclusion(i, j)
                        ah.addExclusion(i, j)
            elif isinstance(pae_inv, np.ndarray):  # use alphafold PAE matrix for network
                k = k_restraint * pae_inv[i,j]**2
                if k > 0.0:
                    cs.addBond(i,j, dmap[i,j]*unit.nanometer,
                                k*unit.kilojoules_per_mole/(unit.nanometer**2))
                    yu.addExclusion(i, j)
                    ah.addExclusion(i, j)
            else:
                raise
    cs.setUsesPeriodicBoundaryConditions(True)
    return cs, yu, ah

def set_interactions(system, residues, prot, calvados_version, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N, n_chains=1):
    hb = openmm.openmm.HarmonicBondForce()
    # interactions
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    if calvados_version in [1, 2]:
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    elif calvados_version == 3:  # interactions scaled aromatics + R + H
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=sqrt(l1*l2)+m1*m2*0.8; shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    elif calvados_version == 4:  # scaled charges
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=sqrt(l1*l2)+m1*m2*0.5; shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    else:
        raise

    ah.addGlobalParameter('eps', lj_eps * unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc', float(cutoff) * unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    if calvados_version in [3, 4]:
        ah.addPerParticleParameter('m')

    print('rc', cutoff * unit.nanometer)

    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa', yukawa_kappa / unit.nanometer)
    yu.addGlobalParameter('shift', np.exp(-yukawa_kappa * 4.0) / 4.0 / unit.nanometer)
    yu.addPerParticleParameter('q')

    for j in range(n_chains):
        begin = j * N
        end = j * N + N
        for a, e in zip(prot.fasta, yukawa_eps):
            yu.addParticle([e * unit.nanometer * unit.kilojoules_per_mole])
            if calvados_version in [3, 4]:
                m = 1.0 if a in ['R', 'H', 'F', 'Y', 'W'] else 0.0
                ah.addParticle([residues.loc[a].sigmas * unit.nanometer, residues.loc[a].lambdas * unit.dimensionless,
                                m * unit.dimensionless])
            else:
                ah.addParticle([residues.loc[a].sigmas * unit.nanometer, residues.loc[a].lambdas * unit.dimensionless])

        for i in range(begin, end - 1):
            hb.addBond(i, i + 1, 0.38 * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
            yu.addExclusion(i, i + 1)
            ah.addExclusion(i, i + 1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4 * unit.nanometer)
    ah.setCutoffDistance(cutoff * unit.nanometer)
    return hb, yu, ah

def add_particles(system,residues,prot,n_chains=1):
    for i_chain in range(n_chains):
        system.addParticle((residues.loc[prot.fasta[0]].MW+2)*unit.amu)
        for a in prot.fasta[1:-1]:
            system.addParticle(residues.loc[a].MW*unit.amu)
        system.addParticle((residues.loc[prot.fasta[-1]].MW+16)*unit.amu)
    return system

def build_box(Lx,Ly,Lz):
    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = Lx * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = Ly * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    return a, b, c

def build_topology(fasta,n_chains=1):
    # build CG topology
    top = md.Topology()
    for i_chain in range(n_chains):
        chain = top.add_chain()
        for resname in fasta:
            residue = top.add_residue(resname, chain)
            top.add_atom(resname, element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top.add_bond(chain.atom(i),chain.atom(i+1))
    return top

def p2c(r, phi):
    """
    polar to cartesian
    """
    return (r * np.cos(phi), r * np.sin(phi))

def xy_spiral_array(n, delta=0, arc=.38, separation=.7):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i in range(n):
        coords.append(list(p2c(r, phi))+[0])
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta

def geometry_from_pdb(pdb):
    """ positions in nm """
    u = mda.Universe(pdb)
    cas = u.select_atoms('name CA')

    cas.translate(-cas.center_of_mass())

    pos = cas.positions / 10.  # nm

    # print(pos.shape)
    return pos

def slab_xy(L,margin):
    xy = np.empty(0)
    xy = np.append(xy,np.random.rand(2)*(L-margin)-(L-margin)/2).reshape((-1,2))
    for x,y in np.random.rand(1000,2)*(L-margin)-(L-margin)/2:
        x1 = x-L if x>0 else x+L
        y1 = y-L if y>0 else y+L
        if np.all(np.linalg.norm(xy-[x,y],axis=1)>.7):
            if np.all(np.linalg.norm(xy-[x1,y],axis=1)>.7):
                if np.all(np.linalg.norm(xy-[x,y1],axis=1)>.7):
                    xy = np.append(xy,[x,y]).reshape((-1,2))
        if xy.shape[0] == 100:
            break
    n_chains = xy.shape[0]
    return xy, n_chains

def slab_dimensions(N):
    if N > 350:
        L = 25.
        Lz = 300.
        margin = 8
        Nsteps = int(2e7)
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
        Nsteps = int(6e7)
    else:
        L = 15.
        Lz = 150.
        margin = 2
        Nsteps = int(6e7)
    return L, Lz, margin, Nsteps

def load_parameters(flib, dataset, cycle, calvados_version):
    if calvados_version in [1,2]:
        residues = pd.read_csv(f'{flib}/{dataset}/residues_{cycle-1}.csv').set_index('one')

        """if calvados_version == 1:
            r.lambdas = r['CALVADOS1'] # select CALVADOS1 or CALVADOS2 stickiness parameters
        else:
            r.lambdas = r['CALVADOS2'] # select CALVADOS1 or CALVADOS2 stckiness parameters"""

        residues.lambdas = residues['lambdas']  # use 0.5 if cycle==0
    elif calvados_version == 3:
        residues = pd.read_csv(f'{flib}/residues_RHYFW.csv').set_index('one')
    elif calvados_version == 4:
        residues = pd.read_csv(f'{flib}/residues_RHYFW_075.csv').set_index('one')
    return residues

def load_pae(input_pae, cutoff=0.25):
    """ pae as pkl file (AF2 format) """
    pae = pd.read_pickle(input_pae)['predicted_aligned_error']/10. + .0001  # (N,N) N is the num of res, nm
    print(pae)
    # pae = np.where(pae == 0, 1, pae)  # avoid division by zero (for i = j)
    pae_inv = 1 / pae  # inverse pae
    pae_inv = np.where(pae_inv > cutoff, pae_inv, 0)
    return pae_inv

def compare_lambdas(cwd):
    cycles = 5
    datasets = ["IDPs_allmultidomain", "IDPs_multidomainExcludeGS", "IDPs"]
    marker_dict = {"IDPs_allmultidomain": "^", "IDPs_multidomainExcludeGS": "v", "IDPs": "s"}
    color_dict = {"IDPs_allmultidomain": "blue", "IDPs_multidomainExcludeGS": "green", "IDPs": "orange"}

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("λ among different datasets", fontsize=20)
    ax1.set_ylabel("lambda values", fontsize=20)
    ax1.set_xlabel("Residues", fontsize=20)

    residues_publication = pd.read_csv(f"{cwd}/residues_pub.csv")
    ax1.scatter(list(residues_publication.one), list(residues_publication.lambdas), s=60, edgecolors="red", label="CALVADOS2",
                marker="o", c="none")
    for dataset in datasets:
        lambda_my = pd.read_csv(f"{cwd}/{dataset}/residues_{cycles}.csv")
        print(list(residues_publication.one))
        ax1.scatter(list(lambda_my.one), list(lambda_my[f"lambdas_{cycles}"]), s=60, marker=marker_dict[dataset], label=dataset, c="none", edgecolors=color_dict[dataset])
        """for cycle in range(-1, cycles+1):
            print(list(lambda_my.one))
            if cycle != cycles:
                ax1.scatter(list(lambda_my.one), list(lambda_my[f"lambdas_{cycle}"]), s=10, color="blue", marker="^", alpha=(cycle+2)/(cycles+2))
            else:
                ax1.scatter(list(lambda_my.one), list(lambda_my[f"lambdas_{cycle}"]), s=60, color="blue",
                            label=f"fan{cycle}", marker="^", alpha=(cycle+2)/(cycles+2))"""


    plt.legend(fontsize=12)
    plt.show()

def visualize_traj(cwd, dataset, name, cycle):
    traj_dcd = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd", f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")
    traj_dcd = traj_dcd.superpose(traj_dcd[0])
    # L = np.squeeze(traj_dcd[0].unitcell_lengths)[0]  # nm
    # the origin is at the point of box
    # traj_dcd.xyz = traj_dcd.xyz + L/2  # nm  <centering>
    traj_dcd.save_trr(f"{cwd}/{dataset}/{name}/{cycle}/{name}.trr")

def plot_loss():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    dataset = "IDPs_multidomainExcludeGS"  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    total_cycles = 5
    multidomain = True

    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    multidomain_ExcludeGS_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1"]

    multidomain_dict = {"IDPs": [], "IDPs_allmultidomain": allmultidomain_names, "IDPs_multidomainExcludeGS": multidomain_ExcludeGS_names}
    # exclusion_GS_dict = {"IDPs_allmultidomain": False, "IDPs_multidomainExcludeGS": True}
    # ExcludeGS = exclusion_GS_dict[dataset]
    multidomain_names = multidomain_dict[dataset]

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(f"{dataset}-{total_cycles+1}cycles", fontsize=20)
    ax1.set_ylabel("loss", fontsize=20)
    ax1.set_xlabel("Iterations", fontsize=20)

    start_x = 0
    points_x = []
    chi2_rg = []
    chi2_multi_rg = []
    chi2_pre = []
    theta_prior = []
    for cycle in range(total_cycles+1):
        res = pd.read_pickle(f"{cwd}/{dataset}/{cycle}_chi2.pkl")
        # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
        points_x += (np.array(res.index)[1:]+start_x).tolist()
        start_x = points_x[-1]
        if multidomain:
            chi2_multi_rg += list(np.mean(
                pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl")[multidomain_names].to_numpy(), axis=1))
        chi2_rg += list(res["chi2_rg"])[1:]
        chi2_pre += list(res["chi2_pre"])[1:]
        theta_prior += list(res["theta_prior"])[1:]

    ax1.scatter(points_x, chi2_rg, s=5, color="blue", label="chi2_rg", marker="o")
    ax1.scatter(points_x, chi2_pre, s=5, color="orange", label="chi2_pre", marker="o")
    ax1.scatter(points_x, -np.array(theta_prior), s=5, color="black", label="-theta_prior", marker="o")
    if multidomain:
        ax1.scatter(points_x, chi2_multi_rg, s=5, color="green", label="chi2_multi_rg", marker="o")
    plt.legend(fontsize=20, markerscale=6)
    plt.show()

def scp_PredPdbFromCom():
    cwd = "/Users/erik/PycharmProjects/IDPs_multi"
    submission_1 = Template(
        """#!/bin/bash
        
    
        python3 {{cwd}}/simulate.py --config {{cwd}}/{{name}}/{{cycle}}/config_{{replica}}.yaml --cpu_num {{cpu_num}} --overwrite {{overwrite}}""")

def decide_best_pae(cwd, name) -> str:
    ranking_confidences = []
    for i in range(1, 6):
        ranking_confidences.append(pd.read_pickle(f"{cwd}/af2pre/{name}/result_model_{i}_ptm_pred_0.pkl")["ranking_confidence"])
    ranking_confidences = np.array(ranking_confidences)
    return f"{cwd}/af2pre/{name}/result_model_{np.argmax(ranking_confidences)+1}_ptm_pred_0.pkl"

def plotAHenergy():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    dataset = "IDPs_allmultidomain"  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    cycles = 5
    nframes = 4000
    # dt = 0.01  # ps
    multidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("Ashbaugh-Hatch potential, all domains excluded", fontsize=20)
    ax1.set_ylabel("Energy [KJ/mol]", fontsize=20)
    ax1.set_xlabel("Frames", fontsize=20)
    allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")

    for name in allproteins.index:
        if name in multidomain_names:
            energy_traj = []
            for cycle in range(cycles+1):
                energy_traj += np.load(f"{cwd}/{dataset}/{name}/{cycle}/{name}_AHenergy.npy").tolist()
                # energy_traj = energy_traj/np.mean(energy_traj)
                # energy_traj = space_data(list(energy_traj), interval=10)
            ax1.plot([i for i in range(nframes*(cycles+1))], energy_traj, label=name)
            print(name, np.mean(energy_traj[-4000:]))

    plt.legend()
    plt.show()

def compare_prior():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    datasets = ["IDPs", "IDPs_allmultidomain", "IDPs_multidomainExcludeGS"]  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    total_cycles = 5
    # exclusion_GS_dict = {"IDPs_allmultidomain": False, "IDPs_multidomainExcludeGS": True}
    # ExcludeGS = exclusion_GS_dict[dataset]

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(f"-prior_loss {total_cycles + 1}cycles", fontsize=20)
    ax1.set_ylabel("-prior_loss", fontsize=20)
    ax1.set_xlabel("Iterations", fontsize=20)

    for dataset in datasets:
        start_x = 0
        points_x = []
        theta_prior = []
        for cycle in range(total_cycles + 1):
            res = pd.read_pickle(f"{cwd}/{dataset}/{cycle}_chi2.pkl")
            # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
            points_x += (np.array(res.index)[1:] + start_x).tolist()
            start_x = points_x[-1]
            theta_prior += list(res["theta_prior"])[1:]

        ax1.scatter(points_x, -np.array(theta_prior), s=5, label=dataset, marker="o")

    plt.legend(fontsize=20, markerscale=6)
    plt.show()

def multi_loss():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    datasets = ["IDPs_allmultidomain", "IDPs_multidomainExcludeGS"]  # IDPs_allmultidomain, IDPs_multidomainExcludeGS
    total_cycles = 5

    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1",
                            'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    multidomain_ExcludeGS_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7",
                                   "hSUMO_hnRNPA1"]

    multidomain_dict = {"IDPs_allmultidomain": allmultidomain_names, "IDPs_multidomainExcludeGS": multidomain_ExcludeGS_names}
    # exclusion_GS_dict = {"IDPs_allmultidomain": False, "IDPs_multidomainExcludeGS": True}
    # ExcludeGS = exclusion_GS_dict[dataset]

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(f"chi2_multi_rg {total_cycles + 1}cycles", fontsize=20)
    ax1.set_ylabel("chi2_multi_rg", fontsize=20)
    ax1.set_xlabel("Iterations", fontsize=20)

    for dataset in datasets:
        print("dataset:", dataset)
        multidomain_names = multidomain_dict[dataset]
        start_x = 0
        points_x = []
        chi2_multi_rg = []
        for cycle in range(total_cycles + 1):
            res = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl")[multidomain_names]
            chi2_multi_rg += list(np.mean(res.to_numpy(),axis=1))
            # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
            points_x += (np.array(res.index) + start_x).tolist()
            start_x = points_x[-1]
        print("last loss:", chi2_multi_rg[-1])
        ax1.scatter(points_x, np.array(chi2_multi_rg), s=5, label=dataset, marker="o")

    plt.legend(fontsize=20, markerscale=6)
    plt.show()

def IDPs_loss():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    datasets = ["IDPs_allmultidomain", "IDPs_multidomainExcludeGS"]  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    total_cycles = 5

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(f"chi2_IDPs_rg {total_cycles + 1}cycles", fontsize=20)
    ax1.set_ylabel("chi2_IDPs_rg", fontsize=20)
    ax1.set_xlabel("Iterations", fontsize=20)

    for dataset in datasets:
        print("dataset:", dataset)
        start_x = 0
        points_x = []
        chi2_IDPs_rg = []
        for cycle in range(total_cycles + 1):
            res = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl")
            # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
            points_x += (np.array(res.index) + start_x).tolist()
            start_x = points_x[-1]
            chi2_IDPs_rg += list(np.mean(res.to_numpy(),axis=1))

        ax1.scatter(points_x, chi2_IDPs_rg, s=5, label=dataset, marker="o")
        print("last loss:", chi2_IDPs_rg[-1])
    plt.legend(fontsize=20, markerscale=6)
    plt.show()

def RgBoxplot():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    dataset = "IDPs_multidomainExcludeGS"  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    cycles = 5
    # multidomain = True

    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1",
                            'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    multidomain_ExcludeGS_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7",
                                   "hSUMO_hnRNPA1"]

    multidomain_dict = {"IDPs": [], "IDPs_allmultidomain": allmultidomain_names,
                        "IDPs_multidomainExcludeGS": multidomain_ExcludeGS_names}
    multidomain_names = multidomain_dict[dataset]
    # nframes = 4000

    fig = plt.figure(num=1, figsize=(15, 20), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("")
    ax1.set_ylabel(f"{dataset}: Rg_BoxPlot over {cycles}cycles [nm]", fontsize=20)
    ax1.set_xlabel("Multidomain names", fontsize=20)

    rgarrays = []
    xticklabels = []
    for name in multidomain_names:
        print(name)
        ssdomains = get_ssdomains(name, f'{cwd}/domains.yaml')
        for ssdomain_idx in range(len(ssdomains)):
            xticklabels.append(f"{name}/D{ssdomain_idx}")
            tmp = []
            for cycle in range(cycles+1):
                df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
                t = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd", f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")  # nm
                residues = [res.name for res in t.top.atoms]
                masses = df.loc[residues, 'MW'].values
                masses[0] += 2
                masses[-1] += 16
                if not os.path.isfile(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy"):
                    print("calculating Rg..........")
                    ssdomain = ssdomains[ssdomain_idx]
                    mask = np.zeros(len(residues), dtype=bool)  # used to filter residues within domains
                    # ssdomains = np.array(sum(ssdomains, []))  # the numbe of residues within domains
                    mask[np.array(ssdomain)-1] = True  # the number of residues, (N,)
                    # calculate the center of mass
                    # print(t.xyz.shape)  (N_traj, N_res, 3)
                    filter_traj = np.array([traj[mask] for traj in t.xyz])  # (N_traj, N_res-N_notdomain, 3)
                    filter_masses = masses[mask]
                    cm = np.sum(filter_traj * filter_masses[np.newaxis, :, np.newaxis], axis=1) / filter_masses.sum()  # (N_traj, 3)

                    # calculate residue-cm distances
                    si = np.linalg.norm(filter_traj - cm[:, np.newaxis, :], axis=2)
                    # calculate rg
                    rgarray = np.sqrt(np.sum(si ** 2 * filter_masses, axis=1) / filter_masses.sum())
                    np.save(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy", rgarray)
                    rgarray = rgarray.tolist()
                else:
                    rgarray = np.load(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy")

                tmp += rgarray.tolist()
            rgarrays.append(np.array(tmp))
    ax1.boxplot(rgarrays)
    plt.setp(ax1, xticks=[i+1 for i in range(len(xticklabels))], xticklabels=xticklabels)
    # plt.legend(fontsize=20, markerscale=6)
    plt.xticks(rotation=90, size=14)
    plt.yticks(rotation=90, size=14)
    plt.show()

def RgTraj_plot():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    dataset = "IDPs_allmultidomain"  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    cycles = 5
    # multidomain = True

    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1",
                            'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    multidomain_ExcludeGS_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7",
                                   "hSUMO_hnRNPA1"]

    multidomain_dict = {"IDPs": [], "IDPs_allmultidomain": allmultidomain_names,
                        "IDPs_multidomainExcludeGS": multidomain_ExcludeGS_names}
    multidomain_names = multidomain_dict[dataset]
    nframes = 4000

    fig = plt.figure(num=1, figsize=(15, 20), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylabel(f"Rg [nm]", fontsize=20)
    ax1.set_xlabel("Frames", fontsize=20)

    for name in multidomain_names:
        if name == "Gal3":
            ax1.set_title(f"{name}: Rg_Traj over {cycles}cycles", fontsize=20)
            print(name)
            ssdomains = get_ssdomains(name, f'{cwd}/domains.yaml')
            for ssdomain_idx in range(len(ssdomains)):
                rgarrays = []
                for cycle in range(cycles + 1):
                    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
                    t = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd",
                                    f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")  # nm
                    residues = [res.name for res in t.top.atoms]
                    masses = df.loc[residues, 'MW'].values
                    masses[0] += 2
                    masses[-1] += 16
                    if not os.path.isfile(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy"):
                        ssdomain = ssdomains[ssdomain_idx]
                        mask = np.zeros(len(residues), dtype=bool)  # used to filter residues within domains
                        # ssdomains = np.array(sum(ssdomains, []))  # the numbe of residues within domains
                        mask[np.array(ssdomain) - 1] = True  # the number of residues, (N,)
                        # calculate the center of mass
                        # print(t.xyz.shape)  (N_traj, N_res, 3)
                        filter_traj = np.array([traj[mask] for traj in t.xyz])  # (N_traj, N_res-N_notdomain, 3)
                        filter_masses = masses[mask]
                        cm = np.sum(filter_traj * filter_masses[np.newaxis, :, np.newaxis],
                                    axis=1) / filter_masses.sum()  # (N_traj, 3)

                        # calculate residue-cm distances
                        si = np.linalg.norm(filter_traj - cm[:, np.newaxis, :], axis=2)
                        # calculate rg
                        rgarray = np.sqrt(np.sum(si ** 2 * filter_masses, axis=1) / filter_masses.sum())
                        np.save(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy", rgarray)
                    else:
                        rgarray = np.load(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy")

                    rgarrays.append(rgarray)
                ax1.scatter([x for x in range((cycles+1)*nframes)], rgarrays, label=f"{name}/D{ssdomain_idx}")

    plt.legend(fontsize=20, markerscale=2)
    plt.xticks(rotation=90, size=14)
    plt.yticks(rotation=90, size=14)
    plt.show()

def Rg_exp_vs_Rg_calc():
    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    datasets = ["IDPs", "IDPs_allmultidomain", "IDPs_multidomainExcludeGS"]  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    cycle = 5  # use current lambdas to test accuracy of next cycle
    shape_dict = {"IDPs": 'o', "IDPs_allmultidomain": '^', "IDPs_multidomainExcludeGS": 's'}

    fig = plt.figure(num=1, figsize=(15, 15), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("cal vs exp Rg", fontsize=20)
    ax1.set_ylabel("cal_Rgvalues [nm]", fontsize=20)
    ax1.set_xlabel("exp_Rgvalues [nm]", fontsize=20)

    for dataset in datasets:
        allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")
        idps_exp, idps_cal = [], []
        GS_exp, GS_cal = [], []
        multi_exp, multi_cal = [], []
        for name in allproteins.index:
            exp_value = allproteins.loc[name]["expRg"]
            if not os.path.isfile(f"{cwd}/{dataset}/{name}/{cycle+1}/Rg_traj.npy"):
                df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
                t = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle+1}/{name}.dcd", f"{cwd}/{dataset}/{name}/{cycle+1}/{name}.pdb")
                residues = [res.name for res in t.top.atoms]
                masses = df.loc[residues, 'MW'].values
                masses[0] += 2
                masses[-1] += 16
                # calculate the center of mass
                cm = np.sum(t.xyz * masses[np.newaxis, :, np.newaxis], axis=1) / masses.sum()
                # calculate residue-cm distances
                si = np.linalg.norm(t.xyz - cm[:, np.newaxis, :], axis=2)
                # calculate rg
                rgarray = np.sqrt(np.sum(si ** 2 * masses, axis=1) / masses.sum())
                np.save(f"{cwd}/{dataset}/{name}/{cycle+1}/Rg_traj.npy", rgarray)
                cal_value = np.round(np.mean(rgarray), 2)
            else:
                cal_value = np.round(np.mean(np.load(f"{cwd}/{dataset}/{name}/{cycle+1}/Rg_traj.npy")), 2)
            if not name in allmultidomain_names:
                idps_exp.append(exp_value)
                idps_cal.append(cal_value)
            elif name[:2] == "GS":
                GS_exp.append(exp_value)
                GS_cal.append(cal_value)
            else:
                multi_exp.append(exp_value)
                multi_cal.append(cal_value)

        ax1.scatter(idps_exp, idps_cal, label=f"IDPs in {{{dataset}}}", c="none", edgecolors="blue", s=70, linewidths=3, marker=shape_dict[dataset])
        if dataset != "IDPs":
            ax1.scatter(multi_exp, multi_cal, label=f"multidomain in {{{dataset}}}", c="none", edgecolors="green", s=70, linewidths=3, marker=shape_dict[dataset])
        if dataset == "IDPs_allmultidomain":
            ax1.scatter(GS_exp, GS_cal, label=f"GS-pro in {{{dataset}}}", c="none", edgecolors="orange", s=70, linewidths=3, marker=shape_dict[dataset])

    ax1.plot([0,7],[0,7], color="red")
    plt.legend(fontsize=15)
    plt.show()

def outputrelax_stride():
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    multidomain_names = ["THB_C2","Ubq2","Ubq3","Gal3","TIA1","Ubq4","hnRNPA1","C5_C6_C7","hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
    for name in multidomain_names:
        os.system(f"stride {cwd}/extract_relax/{name}_rank0_relax.pdb > {cwd}/relax_stride/{name}_rank0_relax_stride.txt")

def calrelative_SASA():
    # Sometimes polar residues whose sidechains pointing outside could also have very little values,
    # because of interactions between their sidechains and others, resulting in their sidechains buried
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    dir = "relax_stride"
    relative_sasa_cutoff = 15  # 15%
    relative_SASA_dict = defaultdict(nested_dictlist)
    allmultidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1",
                            'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    # allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")
    for name in allmultidomain_names:
        name_dict = defaultdict(list)
        with open(f"{cwd}/{dir}/{name}_rank0_relax_stride.txt") as file:
            for line in file.readlines():
                record = line.strip()
                if record[:3] == "ASG":
                    record = record.split()
                    resName = processIon(record[1])
                    resSeq = record[3]
                    sasa = float(record[-2])  # Angstroms^2

                    relative_sasa = np.round(sasa / sasa_max[resName], 3)
                    # print(f"{resName}-{resSeq}: {relative_sasa} %")
                    name_dict[f"{resName}-{resSeq}"] = relative_sasa
        relative_SASA_dict[name] = name_dict

    np.save(f"{cwd}/relative_SASA_dict.npy", relative_SASA_dict)  # rember to import utils before opening this saved dict

if __name__ == '__main__':
    # calrelative_SASA()
    # res = np.load("/groups/sbinlab/fancao/IDPs_multi/relative_SASA_dict.npy", allow_pickle=True).item()
    # outputrelax_stride()
    # Rg_exp_vs_Rg_calc()
    # RgTraj_plot()
    # RgBoxplot()
    # IDPs_loss()
    # multi_loss()
    # compare_prior()
    # plotAHenergy()
    # print(decide_best_pae("/groups/sbinlab/fancao/IDPs_multi", "Gal3"))
    # plot_loss()
    # visualize_traj("/groups/sbinlab/fancao/IDPs_multi", "IDPs_multidomainExcludeGS", "C5_C6_C7", "5")
    # load_pae("/groups/sbinlab/fancao/IDPs_multi/PNt_pae35_368.json")  # https://alphafold.com/entry/Q546U4
    # compare_lambdas("/groups/sbinlab/fancao/IDPs_multi")
    pass