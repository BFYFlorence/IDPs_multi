import time
from argparse import ArgumentParser
import os
from simtk.openmm import app
from utils import *

def simulate(config, overwrite, cpu_num):
    os.environ["OPENMM_CPU_THREADS"] = str(cpu_num)
    """ Simulate openMM Calvados

    * config is a dictionary """

    # parse config
    name, temp, ionic = config['name'], config['temp'], config['ionic']
    cutoff, steps, wfreq = config['cutoff'], config['steps'], config['wfreq']
    L = config['L']
    seq = config['seq']
    cycle = config["cycle"]
    replica = config["replica"]
    calvados_version, pfname = config['calvados_version'], config['pfname']
    flib, ffasta = config['flib'], config['ffasta']
    use_pdb, fpdb = config['use_pdb'], config['fpdb']
    slab, runtime = config['slab'], config['runtime']
    steps = config["steps"]
    print("use_pdb: ", use_pdb)
    if use_pdb:
        input_pdb = fpdb
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
        input_pdb = ''
        use_hnetwork = False
        pae = None
        ssdomains = None


    print(f'calvados version: {calvados_version}')
    # load residue parameters
    residues = load_parameters(flib, cycle, calvados_version)
    print(residues)

    # build protein dataframe
    proteins = initProteins()  # name,input_pdb,ionic=ionic,use_pdb=use_pdb)
    proteins = addProtein(proteins, name, seq=seq, use_pdb=use_pdb, pdb=input_pdb, temp=temp, ionic=ionic, ffasta=ffasta)
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
            print(f'Starting from pdb structure {input_pdb}')
            pos = geometry_from_pdb(input_pdb)
        else:
            spiral = True
            if spiral:
                pos = xy_spiral_array(N)
            else:
                pos = [[L / 2, L / 2, L / 2 + (i - N / 2.) * .38] for i in range(N)]
            pos = np.array(pos)
    # build mdtraj topology from fasta
    top = build_topology(fasta, n_chains=n_chains)
    # print(top.n_bonds)
    pdb_cg = f'{flib}/{name}/{cycle}/top_{replica}.pdb'

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
    platform = openmm.Platform.getPlatformByName(pfname)
    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads=str(cpu_num)))

    fcheck = f'restart_{replica}.chk'

    if os.path.isfile(fcheck) and not overwrite:
        print(f'Reading check point file {fcheck}')
        simulation.loadCheckpoint(fcheck)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{flib}/{name}/{cycle}/{replica}.dcd', wfreq, append=True, enforcePeriodicBox=False))
    else:
        # print(pdb.positions)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True)
        pos2 = state.getPositions(asNumpy=True)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{flib}/{name}/{cycle}/{replica}.dcd', wfreq, enforcePeriodicBox=False, append=False))

    simulation.reporters.append(app.statedatareporter.StateDataReporter(f'{flib}/{name}/{cycle}/statedata_{replica}.log', int(wfreq * 10),
                                                                        step=True, speed=True, elapsedTime=True,
                                                                        separator='\t'))

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', nargs='?', default='config.yaml', const='config.yaml', type=str)
    parser.add_argument('--cpu_num', nargs='?', default=1, const=1, type=int)
    parser.add_argument('--overwrite', nargs='?', default=False, const=True, type=bool)
    args = parser.parse_args()

    with open(f'{args.config}', 'r') as stream:
        config = yaml.safe_load(stream)

    simulate(config, args.overwrite, args.cpu_num)
