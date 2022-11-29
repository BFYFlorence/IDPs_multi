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

#SBATCH --exclusive
submission_1 = Template(
"""#!/bin/bash
#SBATCH --job-name={{name}}_{{replica}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={{cpu_num}}
#SBATCH --partition=sbinlab_ib2
#SBATCH --mem=1GB
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/out_{{replica}}
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/err_{{replica}}

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

echo "Number of cpus requested per task:" $SLURM_CPUS_PER_TASK
echo "Number of tasks requested per core:" $SLURM_NTASKS_PER_CORE
echo $SLURM_CPUS_ON_NODE

python3 {{cwd}}/simulate.py --config {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/config_{{replica}}.yaml --cpu_num {{cpu_num}} --overwrite {{overwrite}}""")

submission_2 = Template(
"""#!/bin/bash
#SBATCH --job-name={{name}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=sbinlab_ib2
#SBATCH --dependency=afterok{% for id in jobid %}:{{id}}{% endfor %}
#SBATCH --mem=10GB
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_out
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/merge_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

echo "Number of cpus requested per task:" $SLURM_CPUS_PER_TASK
echo "Number of tasks requested per core:" $SLURM_NTASKS_PER_CORE
echo $SLURM_CPUS_ON_NODE

python {{cwd}}/merge_replicas.py --cwd {{cwd}} --dataset {{dataset}} --name {{name}} --cycle {{cycle}} --replicas {{replicas}} --discard_first_nframes {{discard_first_nframes}}""")

submission_3 = Template(
"""#!/bin/bash
#SBATCH --job-name=opt_{{cycle}}
#SBATCH --nodes=1
#SBATCH --partition=sbinlab_ib2
#SBATCH --mem=90GB
#SBATCH --dependency=afterok{% for id in jobid %}:{{id}}{% endfor %}
#SBATCH -o {{cwd}}/{{dataset}}/{{cycle}}_out
#SBATCH -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

echo "Number of cpus requested per task:" $SLURM_CPUS_PER_TASK
echo "Number of tasks requested per core:" $SLURM_NTASKS_PER_CORE
echo $SLURM_CPUS_ON_NODE

declare -a proteinsPRE_list=({{proteins}})

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --num_cpus $SLURM_CPUS_ON_NODE --pulchra /groups/sbinlab/fancao/pulchra
done

python {{cwd}}/optimize.py --cwd {{cwd}} --dataset {{dataset}} --log LOG --cycle {{cycle}} --num_cpus $SLURM_CPUS_ON_NODE --cutoff {{cutoff}}""")

def write_config(cwd, dataset, fbase,config_data,config_filename):
    with open(f'{cwd}/{dataset}/{fbase}/{config_filename}','w') as stream:
        yaml.dump(config_data,stream)

def set_harmonic_network(N,pos,pae_inv,yu,ah,ssdomains=None,cs_cutoff=0.9,k_restraint=700):
    cs = openmm.openmm.HarmonicBondForce()
    dmap = self_distances(N,pos)
    for i in range(N-2):
        for j in range(i+2,N):
            if ssdomains != None:  # use fixed domain boundaries for network
                ss = False
                for ssdom in ssdomains:
                    if i in ssdom and j in ssdom:
                        ss = True
                if ss:  # both residues in structured domains
                    if dmap[i,j] < cs_cutoff:  # nm
                        k = k_restraint
                        print(i,j,k)
                        cs.addBond(i,j, dmap[i,j]*unit.nanometer,
                                k*unit.kilojoules_per_mole/(unit.nanometer**2))
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
    """ positions in nm"""
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
    cycles = 6
    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("fan's VS giulio's", fontsize=20)
    ax1.set_ylabel("lambda values", fontsize=20)
    ax1.set_xlabel("Residues", fontsize=20)
    residues_publication = pd.read_csv(f"{cwd}/residues_publication.csv")
    lambda_my = pd.read_csv(f"residues_{cycles}.csv")
    ax1.scatter(list(residues_publication.one), list(residues_publication.lambdas), s=60, color="red",
                label="giulio", marker="o")
    print(list(residues_publication.one))

    for cycle in range(-1, cycles+1):
        print(list(lambda_my.one))
        if cycle != cycles:
            ax1.scatter(list(lambda_my.one), list(lambda_my[f"lambdas_{cycle}"]), s=10, color="blue", marker="^", alpha=(cycle+2)/(cycles+2))
        else:
            ax1.scatter(list(lambda_my.one), list(lambda_my[f"lambdas_{cycle}"]), s=60, color="blue",
                        label=f"fan{cycle}", marker="^", alpha=(cycle+2)/(cycles+2))


    plt.legend(fontsize=20)
    plt.show()

def visualize_traj(path2traj, path2top):
    traj_dcd = md.load_dcd(path2traj, path2top)
    traj_dcd = traj_dcd.superpose(traj_dcd[0])
    L = np.squeeze(traj_dcd[0].unitcell_lengths)[0]  # nm
    # the origin is at the point of box
    traj_dcd.xyz = traj_dcd.xyz + L/2  # nm  <centering>
    traj_dcd.save_trr("/groups/sbinlab/fancao/test_ENM/Gal3/7/0.trr")

def plot_loss(cwd):
    cycles = 6

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("", fontsize=20)
    ax1.set_ylabel("loss", fontsize=20)
    ax1.set_xlabel("Iterations", fontsize=20)
    start_x = 0
    points_x = []
    chi2_rg = []
    chi2_pre = []
    theta_prior = []
    for cycle in range(cycles):
        res = pd.read_pickle(f"{cwd}/{cycle}_chi2.pkl")
        # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
        points_x += (np.array(res.index)+start_x).tolist()
        start_x = points_x[-1]

        chi2_rg += list(res["chi2_rg"])
        chi2_pre += list(res["chi2_pre"])
        theta_prior += list(res["theta_prior"])

    ax1.scatter(points_x, chi2_rg, s=5, color="blue", label="chi2_rg", marker="^")
    ax1.scatter(points_x, chi2_pre, s=5, color="orange", label="chi2_pre", marker="s")
    ax1.scatter(points_x, theta_prior, s=5, color="green", label="-theta_prior", marker="x")
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

if __name__ == '__main__':
    print(decide_best_pae("/groups/sbinlab/fancao/IDPs_multi", "Gal3"))
    # plot_loss("/groups/sbinlab/fancao/IDPs_multi")
    # visualize_traj("/groups/sbinlab/fancao/test_ENM/Gal3/7/0.dcd", "/groups/sbinlab/fancao/test_ENM/Gal3/7/top_0.pdb")
    # load_pae("/groups/sbinlab/fancao/IDPs_multi/PNt_pae35_368.json")  # https://alphafold.com/entry/Q546U4
    # compare_lambdas("/groups/sbinlab/fancao/IDPs_multi")