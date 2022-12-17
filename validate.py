import test_idps
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from analyse import *
import MDAnalysis
import time
import os
import glob
import sys
from DEERPREdict.PRE import PREpredict
from argparse import ArgumentParser
from sklearn.neighbors import KernelDensity
import ray
import logging
import shutil
import yaml

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)

cwd = "/groups/sbinlab/fancao/IDPs_multi"
log_path = "LOG"
dataset = "test"
cycle = 0
num_cpus = 10  # can be 64
cutoff = 2.4

ray.init(num_cpus=num_cpus)
if not os.path.isdir(f"{cwd}/{dataset}/{log_path}"):
    os.system(f"mkdir -p {cwd}/{dataset}/{log_path}")
logging.basicConfig(filename=f'{cwd}/{dataset}/{log_path}/log',level=logging.INFO)


@ray.remote(num_cpus=1)
def evaluatePRE(cwd, dataset, label, name, prot):
    if not os.path.isdir(f"{cwd}/{dataset}/{log_path}/{name}"):
        os.system(f"mkdir -p {cwd}/{dataset}/{log_path}/{name}")

    prefix = f'{cwd}/{dataset}/{prot.path}/calcPREs/res'
    filename = prefix+'-{:d}.pkl'.format(label)
    if isinstance(prot.weights, np.ndarray):  # it seems that this never is True
        u = MDAnalysis.Universe(f'{cwd}/{dataset}/{prot.path}/allatom.pdb')
        load_file = filename
    elif isinstance(prot.weights, bool):
        u = MDAnalysis.Universe(f'{cwd}/{dataset}/{prot.path}/allatom.pdb',f'{cwd}/{dataset}/{prot.path}/allatom.dcd')
        load_file = False
    else:
        raise ValueError('Weights argument is a '+str(type(prot.weights)))
    PRE = PREpredict(u, label, log_file=f'{cwd}/{dataset}/{log_path}/{name}/log', temperature=prot.temp, atom_selection='N', sigma_scaling=1.0)
    PRE.run(output_prefix=prefix, weights=prot.weights, load_file=load_file, tau_t=1e-10, tau_c=prot.tau_c*1e-09, r_2=10, wh=prot.wh)

@ray.remote(num_cpus=1)
def calcDistSums(cwd,dataset, df,name,prot, rc):
    # if not os.path.isfile(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy'):
    # print("name:", name)
    traj = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/{name}.dcd",f"{cwd}/{dataset}/{prot.path}/{name}.pdb")
    # traj = traj[:2]
    fasta = [res.name for res in traj.top.atoms]  # ['MET', 'HIS', 'GLN', 'ASP'...'VAL', 'THR', 'ARG']
    pairs = traj.top.select_pairs('all','all')  # (24090, traj_len)
    mask = np.abs(pairs[:,0]-pairs[:,1])>1 # exclude bonds, (24090,)
    pairs = pairs[mask]  # (23871, traj_len)
    d = md.compute_distances(traj,pairs)  # (traj_len, 23871)
    d[d>rc] = np.inf  # cutoff
    r = np.copy(d)  # changing d will not change r, vice versa, (traj_len, 23871)
    n1 = np.zeros(r.shape,dtype=np.int8)
    n2 = np.zeros(r.shape,dtype=np.int8)
    pairs = np.array(list(itertools.combinations(fasta,2)))  # (24090, traj_len)
    # [['MET' 'HIS']
    #  ['MET' 'GLN']
    #  ...
    #  ['VAL' 'THR']
    #  ['VAL' 'ARG']
    #  ['THR' 'ARG']]
    pairs = pairs[mask]  # exclude bonded, (23871, traj_len)
    # [['MET' 'GLN']
    #  ['MET' 'ASP']
    #  ['MET' 'HIS']
    #  ...
    #  ['GLU' 'THR']
    #  ['GLU' 'ARG']
    #  ['VAL' 'ARG']]
    # print(df.loc[pairs[:,0]])
    #       one      MW  lambdas  sigmas    q
    # three
    # MET     M  131.20      0.5   0.618  0.0
    # MET     M  131.20      0.5   0.618  0.0
    # MET     M  131.20      0.5   0.618  0.0
    # ...
    # GLU     E  129.11      0.5   0.592 -1.0
    # GLU     E  129.11      0.5   0.592 -1.0
    # VAL     V   99.13      0.5   0.586  0.0
    # print(type(df.loc[pairs[:,0]].sigmas.values))  # [0.618 0.618 0.618 ... 0.592 0.592 0.586], <class 'numpy.ndarray'>
    # pair-wise sigmas
    sigmas = 0.5*(df.loc[pairs[:,0]].sigmas.values+df.loc[pairs[:,1]].sigmas.values)  # (23871,)
    """lambdas = (df.loc[pairs[:, 0]].lambdas.values + df.loc[pairs[:, 1]].lambdas.values) / 2
    for frame in d:
        u = 0
        u_ct = 0
        u_cut = 0
        for dis_idx in range(len(frame)):
            if frame[dis_idx] <= np.power(2, 1/6.)*sigmas[dis_idx]:
                u += 4*0.2*4.184*(np.power(sigmas[dis_idx]/frame[dis_idx], 12)-np.power(sigmas[dis_idx]/frame[dis_idx], 6))-lambdas[dis_idx]*4*0.2*4.184*(np.power(sigmas[dis_idx]/rc, 12)-np.power(sigmas[dis_idx]/rc, 6))+0.2*4.184*(1-lambdas[dis_idx])
                # u += 4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / frame[dis_idx], 12) - np.power(sigmas[dis_idx] / frame[dis_idx], 6)) - lambdas[dis_idx] * 4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / rc, 12) + np.power(sigmas[dis_idx] / rc,6)) + 0.2 * 4.184 * (1 - lambdas[dis_idx])
                # u += 4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / frame[dis_idx], 12) - np.power(sigmas[dis_idx] / frame[dis_idx], 6)) + 0.2 * 4.184 * (1 - lambdas[dis_idx])
                u_ct += lambdas[dis_idx]*4*0.2*4.184*(np.power(sigmas[dis_idx]/rc, 12)-np.power(sigmas[dis_idx]/rc, 6))
                u_cut += lambdas[dis_idx] * 4 * 0.2 * 4.184 * (
                            np.power(sigmas[dis_idx] / rc, 12) + np.power(sigmas[dis_idx] / rc, 6))
            elif frame[dis_idx] < rc:
                u += lambdas[dis_idx]*((4*0.2*4.184*(np.power(sigmas[dis_idx]/frame[dis_idx], 12)-np.power(sigmas[dis_idx]/frame[dis_idx], 6)))-(4*0.2*4.184*(np.power(sigmas[dis_idx]/rc, 12)-np.power(sigmas[dis_idx]/rc, 6))))
                # u += lambdas[dis_idx] * ((4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / frame[dis_idx], 12) - np.power(sigmas[dis_idx] / frame[dis_idx],6))) - (4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / rc, 12) + np.power(sigmas[dis_idx] / rc, 6))))
                # u += lambdas[dis_idx] * ((4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx] / frame[dis_idx], 12) - np.power(sigmas[dis_idx] / frame[dis_idx], 6))))
                u_ct += lambdas[dis_idx] * 4 * 0.2 * 4.184 * (np.power(sigmas[dis_idx]/rc, 12) - np.power(sigmas[dis_idx]/rc, 6))
                u_cut += lambdas[dis_idx] * 4 * 0.2 * 4.184 * (
                            np.power(sigmas[dis_idx] / rc, 12) + np.power(sigmas[dis_idx] / rc, 6))
        print(u, u_ct, u_cut)"""
    # r, (traj_len, 23871)
    for i,sigma in enumerate(sigmas):
        mask = r[:,i]>np.power(2.,1./6)*sigma  # [ True  True], (traj_len, )
        r[:,i][mask] = np.inf  # distances bigger than Rmin will be inf
        n1[:,i][~mask] = 1  # distances within Rmin have 1
        n2[:,i][np.isfinite(d[:,i])] = 1  # distances within cutoff have 1, because distances in d bigger than cutoff have been assigned as inf

    unique_pairs = np.unique(pairs,axis=0)  # (339, traj_len)
    # [['ALA' 'ALA']
    #  ['ALA' 'ARG']
    #  ...
    #  ['VAL' 'TYR']
    #  ['VAL' 'VAL']]
    pairs = np.core.defchararray.add(pairs[:,0],pairs[:,1])  # (23871,)
    # ['METGLN' 'METASP' 'METHIS' ... 'GLUTHR' 'GLUARG' 'VALARG']

    # axis=1 means extract distances every single frame; then calculate the sum of np.power(d,-12.) of every unique pari (like METGLN)
    d12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(), axis=1, arr=np.power(d,-12.))
    # [[6.03472023e+01 1.20000923e+02 1.29472687e+02 ... 1.14875392e-03 2.68038521e+01 4.90650101e+01]], (traj_len, 339)

    # same as d12
    d6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(), axis=1, arr=-np.power(d,-6.))

    # same as the above
    r12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=np.power(r,-12.))
    r6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=-np.power(r,-6.))

    ncut1 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n1)
    ncut2 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n2)

    # unique-pair-wise sigmas
    sigmas = 0.5*(df.loc[unique_pairs[:,0]].sigmas.values+df.loc[unique_pairs[:,1]].sigmas.values)
    sigmas6 = np.power(sigmas,6)
    sigmas12 = np.power(sigmas6,2)


    eps = 0.2*4.184
    term_1 = eps*(ncut1+4*(sigmas6*r6 + sigmas12*r12))  # ncut1 is used to calculate the constant values when r <= 2^(1/6)*sigma
    term_2 = sigmas6*(d6+ncut2/np.power(rc,6)) + sigmas12*(d12-ncut2/np.power(rc,12))  # ncut2 is used to calculate the constant values of shifted energy
    term_2 = 4*eps*term_2 - term_1  # together to calculate AH energy, better to write down to really understand it

    np.save(f'{cwd}/{dataset}/{prot.path}/energy_sums_1.npy',term_1.sum(axis=1))
    np.save(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy',term_2)
    np.save(f'{cwd}/{dataset}/{prot.path}/unique_pairs.npy',unique_pairs)
    """lambdas = (df.loc[unique_pairs[:, 0]].lambdas.values + df.loc[unique_pairs[:, 1]].lambdas.values) / 2
    print("giulio's cutoff correct:", np.nansum(lambdas * 4*eps*(-sigmas6*ncut2/np.power(rc,6)+sigmas12*ncut2/np.power(rc,12)), axis=1))
    print("giulio's cutoff wrong:",
          np.nansum(lambdas * 4 * eps * (sigmas6 * ncut2 / np.power(rc, 6) + sigmas12 * ncut2 / np.power(rc, 12)),
                    axis=1))
    print(term_1.sum(axis=1))
    print(np.sum(lambdas * term_2, axis=1))
    print(term_1.sum(axis=1) + np.nansum(lambdas * term_2, axis=1))"""

def calcAHenergy(cwd,dataset, df,prot):
    term_1 = np.load(f'{cwd}/{dataset}/{prot.path}/energy_sums_1.npy')
    term_2 = np.load(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy')
    unique_pairs = np.load(f'{cwd}/{dataset}/{prot.path}/unique_pairs.npy')  # (339, traj_len)
    # unique-pair-wise lambdas
    lambdas = (df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)/2
    return term_1+np.nansum(lambdas*term_2,axis=1)

@ray.remote(num_cpus=1)
def calcWeights(cwd,dataset, df,name,prot):
    new_ah_energy = calcAHenergy(cwd,dataset, df,prot)
    ah_energy = np.load(f'{cwd}/{dataset}/{prot.path}/{name}_AHenergy.npy')
    kT = 8.3145*prot.temp*1e-3
    weights = np.exp((ah_energy-new_ah_energy)/kT)
    weights /= weights.sum()
    eff = np.exp(-np.sum(weights*np.log(weights*weights.size)))
    return name,weights,eff

def reweight(cwd, dataset,dp,df,proteinPRE,proteinsRgs, proc_PRE):
    multidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    IDP_names = np.setdiff1d(list(proteinsRgs.index), multidomain_names).tolist()
    Rgloss4multidomain = np.ones(shape=len(multidomain_names))
    Rgloss4IDPs = np.ones(shape=len(list(proteinsRgs.index))-len(multidomain_names))
    trial_proteinsPRE = proteinPRE.copy()
    trial_proteinsRgs = proteinsRgs.copy()
    trial_df = df.copy()
    res_sel = np.random.choice(trial_df.index, 5, replace=False)
    trial_df.loc[res_sel,'lambdas'] += np.random.normal(0,dp,res_sel.size)
    f_out_of_01 = lambda df : df.loc[(df.lambdas<=0)|(df.lambdas>1),'lambdas'].index
    out_of_01 = f_out_of_01(trial_df)
    trial_df.loc[out_of_01,'lambdas'] = df.loc[out_of_01,'lambdas']

    # calculate AH energies, weights and fraction of effective frames
    weights = ray.get([calcWeights.remote(cwd,dataset, trial_df,name,prot) for name,prot in pd.concat((trial_proteinsPRE,trial_proteinsRgs),sort=True).iterrows()])
    for name,w,eff in weights:
        if eff < 0.6:
            return False, df, proteinPRE, proteinsRgs, Rgloss4multidomain, Rgloss4IDPs
        if name in trial_proteinsPRE.index:
            trial_proteinsPRE.at[name,'weights'] = w
            trial_proteinsPRE.at[name,'eff'] = eff
        else:
            trial_proteinsRgs.at[name,'weights'] = w
            trial_proteinsRgs.at[name,'eff'] = eff

    # calculate PREs and cost function
    ray.get([evaluatePRE.remote(cwd,dataset,label,name,trial_proteinsPRE.loc[name]) for n,(label,name) in enumerate(proc_PRE)])
    for name in trial_proteinsPRE.index:
        trial_proteinsPRE.at[name,'chi2_pre'] = calcChi2(cwd,dataset, trial_proteinsPRE.loc[name])
    for name in trial_proteinsRgs.index:
        rg, chi2_rg = reweightRg(df,name,trial_proteinsRgs.loc[name])
        if name in multidomain_names:
            Rgloss4multidomain[np.where(np.array(multidomain_names)==name)[0][0]] = chi2_rg
        else:
            Rgloss4IDPs[np.where(np.array(IDP_names)==name)[0][0]] = chi2_rg
        trial_proteinsRgs.at[name,'Rg'] = rg
        trial_proteinsRgs.at[name,'chi2_rg'] = chi2_rg

    return True, trial_df, trial_proteinsPRE, trial_proteinsRgs, Rgloss4multidomain, Rgloss4IDPs

def validate():
    rc = cutoff
    """
    parser = ArgumentParser()
    parser.add_argument('--path2config',dest='path2config',type=str,required=True)
    args = parser.parse_args()

    with open(f'{args.path2config}', 'r') as stream:
        config = yaml.safe_load(stream)
    cwd, log_path, dataset, cycle, num_cpus, cutoff = config["cwd"], config["log_path"],config["dataset"],config["cycle"],config["num_cpus"], config["cutoff"]
    exclusion_GS = config["exclusion_GS"]

    if not os.path.isdir(f"{cwd}/{dataset}/{log_path}"):
        os.system(f"mkdir -p {cwd}/{dataset}/{log_path}")

    # normal test set
    testProteinRgs = test_idps.initProteinsRgs(cycle)
    allproteins = testProteinRgs
    allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))
    allproteins = allproteins.sort_values('N')"""


    ########################################################################################################################
    #                                  test                                                                     #
    ########################################################################################################################

    # rc = cutoff

    outdir = '/{:d}'.format(cycle)
    proteinsRgs = pd.DataFrame(columns=['temp','expRg','expRgErr','Rg','rgarray','eff','chi2_rg','weights','pH','ionic','fasta','path'],dtype=object)
    fasta_GS48 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSG
SGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSKLMVSKGEEDNMASLPATHELHI
FGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQFSPWILVPHIGYGFHQYLPYPDGM
SPFQAAMVDGSGYQVHRTMQFEDGASLTVNYRYTYEGSHIKGEAQVKGTGFPADGPVMTN
SLTAADWCRSKKTYPNDKTIISTFKWSYTTGNGKRYRSTARTTYTFAKPMAANYLKNQPM
YVFRKTELKHSKTELNFKEWQKAFTD""".replace('\n', '')  # 566
    proteinsRgs.loc['GS48'] = dict(temp=293, expRg=4.11, expRgErr=0.02, pH=7.0, fasta=list(fasta_GS48), ionic=0.15,
                                        path='GS48' + outdir)  # Tórur's thesis, ph is not checked yet

    proteinsPRE= pd.DataFrame(columns=['labels','wh','tau_c','temp','obs','pH','ionic','expPREs','initPREs','eff','chi2_pre','fasta','weights','path'],dtype=object)
    fasta_OPN = """MHQDHVDSQSQEHLQQTQNDLASLQQTHYSSEENADVPEQPDFPDV
PSKSQETVDDDDDDDNDSNDTDESDEVFTDFPTEAPVAPFNRGDNAGRGDSVAYGFRAKA
HVVKASKIRKAARKLIEDDATTEDGDSQPAGLWWPKESREQNSRELPQHQSVENDSRPKF
DSREVDGGDSKASAGVDSRESQGSVPAVDASNQTLESAEDAEDRHSIENNEVTR""".replace('\n', '')
    proteinsPRE.loc['OPN'] = dict(labels=[10, 33, 64, 88, 117, 130, 144, 162, 184, 203], tau_c=3.0,
                                   wh=800,temp=298,obs='rate',pH=6.5,fasta=list(fasta_OPN),ionic=0.15,weights=False,path='OPN'+outdir)
    proteinsPRE = proteinsPRE.astype(object)  # .astype(object) allows to pass dataframes to column_value
    ########################################################################################################################
    #                                  test                                                                     #
    ########################################################################################################################

    for _, prot in proteinsPRE.iterrows():
        if not os.path.isdir(f'{cwd}/{dataset}/{prot.path}'):
            os.mkdir(f'{cwd}/{dataset}/{prot.path}')
        if not os.path.isdir(f'{cwd}/{dataset}/{prot.path}/calcPREs'):
            os.mkdir(f'{cwd}/{dataset}/{prot.path}/calcPREs')

    proc_PRE = [(label,name) for name,prot in proteinsPRE.iterrows() for label in prot.labels]
    # [(10, 'OPN'), (33, 'OPN'), (64, 'OPN'), (88, 'OPN'), (117, 'OPN'), (130, 'OPN'), (144, 'OPN'), (162, 'OPN'), (184, 'OPN'), (203, 'OPN')]

    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv').set_index('three')
    # print(df.columns) Index(['one', 'MW', 'lambdas', 'sigmas', 'q'], dtype='object')
    # print(df.index) Index(['ARG', 'ASP', 'ASN', 'GLU', 'LYS', 'HIS', 'GLN', 'SER', 'CYS', 'GLY', 'THR', 'ALA', 'MET', 'TYR', 'VAL', 'TRP', 'LEU', 'ILE', 'PRO', 'PHE'], dtype='object', name='three')
    logging.info(df.lambdas)

    for name in proteinsPRE.index:
        # load all expPREs values
        proteinsPRE.at[name,'expPREs'] = loadExpPREs(cwd,dataset, name,proteinsPRE.loc[name])

    time0 = time.time()
    # print(proteinsPRE.at["OPN", "weights"])  # False
    # Ideally speaking, evaluatePRE can calculate 64 labels on sbinlab_ib2 at the same time
    # write a config file to record calculation process so that it will not be calculated again
    # cf
    # ray.get([evaluatePRE.remote(cwd,dataset,label,name,proteinsPRE.loc[name]) for n,(label,name) in enumerate(proc_PRE)])
    logging.info('Timing evaluatePRE {:.3f}'.format(time.time()-time0))

    time0 = time.time()
    # cf
    # ray.get([calcDistSums.remote(cwd, dataset, df, name, prot, rc) for name, prot in pd.concat((proteinsPRE, proteinsRgs), sort=True).iloc[:1].iterrows()])
    logging.info('Timing calcDistSums {:.3f}'.format(time.time() - time0))
    # cf
    """for name in proteinsPRE.index:
        np.save(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/{name}_AHenergy.npy', calcAHenergy(cwd, dataset, df, proteinsPRE.loc[name]))
        # calculate the optimal tau_c and chi2_pre with uniform weights
        # the optimal tau_c is fixed during optimization because the ideal situation is that the uniform weights are used to calculate optimal tauC
        # the fixed value of tauC also helps to tell changes of chi_pre are from biased weights
        # After pre-computed r3, r6 and angular, left thing should be using biased weights to recalculate PRE data
        tau_c, chi2_pre = optTauC(cwd, dataset, proteinsPRE.loc[name])
        proteinsPRE.at[name, 'tau_c'] = tau_c
        proteinsPRE.at[name, 'chi2_pre'] = chi2_pre
        # save PRE data calculated using uniform weights
        proteinsPRE.at[name, 'initPREs'] = loadInitPREs(cwd, dataset, name, proteinsPRE.loc[name])
        if os.path.exists(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs'):
            shutil.rmtree(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs')
        shutil.copytree(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/calcPREs',
                        f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs')
    proteinsPRE.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsPRE.pkl')"""

    for name in proteinsRgs.index:
        np.save(f'{cwd}/{dataset}/{proteinsRgs.loc[name].path}/{name}_AHenergy.npy', calcAHenergy(cwd, dataset, df, proteinsRgs.loc[name]))
        rgarray, rg, chi2_rg = calcRg(cwd, dataset, df, name, proteinsRgs.loc[name])
        proteinsRgs.at[name, 'rgarray'] = rgarray
        proteinsRgs.at[name, 'Rg'] = rg
        proteinsRgs.at[name, 'chi2_rg'] = chi2_rg
    proteinsRgs.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsRgs.pkl')


if __name__ == '__main__':
    validate()