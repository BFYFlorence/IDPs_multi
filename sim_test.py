import time
from argparse import ArgumentParser
import os

import numpy as np
from openmm import app
from utils import *
import itertools

def simulate(config, overwrite, cpu_num):
    # os.environ["OPENMM_CPU_THREADS"] = str(cpu_num)
    """ Simulate openMM Calvados

    * config is a dictionary """

    # parse config
    dataset = config["dataset"]
    name, temp, ionic = config['name'], config['temp'], config['ionic']
    cutoff, steps, wfreq = config['cutoff'], config['steps'], config['wfreq']
    L = config['L']
    # gpu_id = config['gpu_id']
    # gpu = config['gpu']
    gpu = False
    seq = config['seq']
    cycle = config["cycle"]
    replica = config["replica"]
    calvados_version = config['calvados_version']
    cwd, path2fasta = config['cwd'], config['path2fasta']
    use_pdb, path2pdb = config['use_pdb'], config['path2pdb']
    slab, runtime = config['slab'], config['runtime']
    steps = config["steps"]
    print("use_pdb: ", use_pdb)
    if use_pdb:
        use_hnetwork = config['use_hnetwork']
        if use_hnetwork:
            k_restraint = config['k_restraint']
            use_ssdomains = config['use_ssdomains']
            if use_ssdomains:
                fdomains = config['fdomains']
                ssdomains = get_ssdomains(name, fdomains)
                pae = None
            else:
                input_pae = config['input_pae']
                pae = load_pae(input_pae)
                ssdomains = None
    else:
        path2pdb = ''
        use_hnetwork = False
        pae = None
        ssdomains = None


    print(f'calvados version: {calvados_version}')
    # load residue parameters
    residues = load_parameters(cwd,dataset, cycle, calvados_version)
    print(residues)

    # build protein dataframe
    proteins = initProteins()  # name,input_pdb,ionic=ionic,use_pdb=use_pdb)
    proteins = addProtein(proteins, name, seq=seq, use_pdb=use_pdb, pdb=path2pdb, temp=temp, ionic=ionic, path2fasta=path2fasta)
    prot = proteins.loc[name]

    # LJ and YU parameters
    lj_eps, fasta, types, MWs = genParamsLJ(residues, name, prot)
    yukawa_eps, yukawa_kappa = genParamsDH(residues, name, prot, temp, calvados_version)

    N = len(fasta)  # number of residues

    if slab:  # overrides L from config for now
        L, Lz, margin, Nsteps = slab_dimensions(N)
        xy, n_chains = slab_xy(L, margin)
    else:
        n_chains = 1
        Lz = L

    # get input geometry
    if slab:
        if use_pdb:
            raise
        else:
            pos = []
            for x, y in xy:
                pos.append([[x, y, Lz / 2 + (i - N / 2.) * .38] for i in range(N)])
            pos = np.array(pos).reshape(n_chains * N, 3)
    else:
        if use_pdb:
            print(f'Starting from pdb structure {path2pdb}')
            pos = geometry_from_pdb(path2pdb)
        else:
            spiral = True
            if spiral:
                pos = xy_spiral_array(N)
            else:
                pos = [[L / 2, L / 2, L / 2 + (i - N / 2.) * .38] for i in range(N)]
            pos = np.array(pos)

    dmap = self_distances(N, pos)
    pos = pos + np.array([L/2, L/2, L/2])
    # np.save(f"{cwd}/{dataset}/{name}/{cycle}/dmap.npy", dmap)
    # np.save(f"{cwd}/{dataset}/{name}/{cycle}/pos.npy", pos)
    # build mdtraj topology from fasta
    top = build_topology(fasta, n_chains=n_chains)
    # print(top.n_bonds)
    pdb_cg = f'{cwd}/{dataset}/{name}/{cycle}/top_{replica}.pdb'

    a = md.Trajectory(pos, top, 0, [L, L, Lz], [90, 90, 90])
    a.save_pdb(pdb_cg, force_overwrite=True)

    # build openmm system
    system = openmm.System()

    # box
    a, b, c = build_box(L, L, Lz)
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # load topology into system
    pdb = app.pdbfile.PDBFile(pdb_cg)

    # print(pdb.topology)

    # particles and termini
    system = add_particles(system, residues, prot, n_chains=n_chains)

    # interactions
    hb, yu, ah = set_interactions(system, residues, prot, calvados_version, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N,
                                  n_chains=n_chains)
    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    print("use_hnetwork: ", use_hnetwork)
    # harmonic network (currently unavailable for slab sim)
    if use_hnetwork:
        if slab:
            raise
        else:
            cs, yu, ah = set_harmonic_network(N, pos, pae, yu, ah, ssdomains=ssdomains, k_restraint=k_restraint)
            system.addForce(cs)

    # use langevin integrator
    integrator = openmm.openmm.LangevinIntegrator(temp * unit.kelvin, 0.01 / unit.picosecond, 0.01 * unit.picosecond)
    print(integrator.getFriction(), integrator.getTemperature())

    # assemble simulation
    if gpu:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # feasible
        os.system("echo $CUDA_VISIBLE_DEVICES")
        # os.system("export CUDA_VISIBLE_DEVICES=1")
        # platform = openmm.Platform.getPlatformByName("CUDA")
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, openmm.Platform.getPlatformByName("CUDA"),
                                               {"DeviceIndex": f"{gpu_id}"})
    else:
        platform = openmm.Platform.getPlatformByName('CPU')
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads=str(cpu_num)))

    fcheck = f'restart_{replica}.chk'


    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    pos2 = state.getPositions(asNumpy=True)
    # dmap2 = self_distances(N, pos2)
    # np.save(f"{flib}/{dataset}/{name}/{cycle}/dmap2.npy", dmap2)
    # np.save(f"{flib}/{dataset}/{name}/{cycle}/pos2.npy", pos2)
    simulation.reporters.append(app.dcdreporter.DCDReporter(f'{cwd}/{dataset}/{name}/{cycle}/{replica}.dcd', wfreq, enforcePeriodicBox=False))

    simulation.reporters.append(
        app.statedatareporter.StateDataReporter(f'{cwd}/{dataset}/{name}/{cycle}/statedata_{replica}.log', int(wfreq),
                                                step=True, speed=True, elapsedTime=True, separator='\t', progress=True,
                                                remainingTime=True, totalSteps=steps))

    starttime = time.time()  # begin timer
    simulation.step(steps)
    endtime = time.time()  # end timer
    target_seconds = endtime - starttime  # total used time
    print(
        f"{name} total simulations used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")
    # run simulation
    """if runtime > 0:  # in hours
        simulation.runForClockTime(runtime * unit.hour, checkpointFile=fcheck, checkpointInterval=30 * unit.minute)
    else:
        step = int(steps / 10)
        for i in progressbar.progressbar(range(10), min_poll_interval=1):
            simulation.step(step)

    simulation.saveCheckpoint(fcheck)"""
    # allproteins = pd.read_pickle(f'{cwd}/{dataset}/allproteins.pkl')
    """starttime = time.time()  # begin timer
    residues = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('one', drop=False)
    top = md.Topology()
    chain = top.add_chain()
    for resname in prot.fasta:
        residue = top.add_residue(residues.loc[resname, 'three'], chain)
        top.add_atom(residues.loc[resname, 'three'], element=md.element.carbon, residue=residue)
    for i in range(len(prot.fasta) - 1):
        top.add_bond(top.atom(i), top.atom(i + 1))
    traj = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/0.dcd", top=top)
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0, 0] / 2
    print(f'Number of frames: {traj.n_frames}')
    traj.save_dcd(f'{cwd}/{dataset}/{name}/{cycle}/{name}.dcd')
    traj[0].save_pdb(f'{cwd}/{dataset}/{name}/{cycle}/{name}.pdb')

    calcDistSums(cwd, dataset, name, cycle)
    np.save(f'{cwd}/{dataset}/{name}/{cycle}/{name}_AHenergy.npy',
            calcAHenergy(cwd, dataset, name, cycle))

    endtime = time.time()  # end timer
    target_seconds = endtime - starttime  # total used time
    print(
        f"{name} total calculation used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")"""


with open("/groups/sbinlab/fancao/IDPs_multi/test/Hst5/0/config_0.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

simulate(config, True, 2)


