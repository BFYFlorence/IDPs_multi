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
import protein_repo
import yaml

time_total = time.time()

parser = ArgumentParser()
parser.add_argument('--path2config',dest='path2config',type=str,required=True)
args = parser.parse_args()

with open(f'{args.path2config}', 'r') as stream:
    config = yaml.safe_load(stream)
cwd, log_path, dataset, cycle, num_cpus, cutoff = config["cwd"], config["log_path"],config["dataset"],config["cycle"],config["num_cpus"], config["cutoff"]
exclusion_GS = config["exclusion_GS"]

if not os.path.isdir(f"{cwd}/{dataset}/{log_path}"):
    os.system(f"mkdir -p {cwd}/{dataset}/{log_path}")


multidomain_names = ["THB_C2","Ubq2","Ubq3","Gal3","TIA1","Ubq4","hnRNPA1","C5_C6_C7","hSUMO_hnRNPA1",'GS0','GS8','GS16','GS24','GS32','GS48']
logging.basicConfig(filename=f'{cwd}/{dataset}/{log_path}/log',level=logging.INFO)

dp = 0.05
theta = .05
eta = .1
xi_0 = .1
rc = cutoff

os.environ["NUMEXPR_MAX_THREADS"]="1"

proteinsPRE = initProteinsPRE(cycle)
proteinsRgs = initProteinsRgs(cycle, exclusion_GS=exclusion_GS)
proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl')
proteinsPRE = pd.read_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl').astype(object)
proteinsRgs = pd.read_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl').astype(object)

for _, prot in proteinsPRE.iterrows():
    if not os.path.isdir(f'{cwd}/{dataset}/{prot.path}'):
        os.mkdir(f'{cwd}/{dataset}/{prot.path}')
    if not os.path.isdir(f'{cwd}/{dataset}/{prot.path}/calcPREs'):
        os.mkdir(f'{cwd}/{dataset}/{prot.path}/calcPREs')

proc_PRE = [(label,name) for name,prot in proteinsPRE.iterrows() for label in prot.labels]
# [(24, 'aSyn'), (42, 'aSyn'), (62, 'aSyn'), (87, 'aSyn'), (103, 'aSyn'), (10, 'OPN'), (33, 'OPN'), (64, 'OPN'), (88, 'OPN'), (117, 'OPN'), (130, 'OPN'), (144, 'OPN'), (162, 'OPN'), (184, 'OPN'), (203, 'OPN'), (16, 'FUS'), (86, 'FUS'), (142, 'FUS'), (16, 'FUS12E'), (86, 'FUS12E'), (142, 'FUS12E')]

ray.init(num_cpus=num_cpus)

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
def calcDistSums(cwd,dataset, df,name,prot):
    multidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    if not os.path.isfile(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy'):
        traj = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/{name}.dcd",f"{cwd}/{dataset}/{prot.path}/{name}.pdb")
        fasta = [res.name for res in traj.top.atoms]
        pairs = traj.top.select_pairs('all','all')  # index starts with 0

        # exclude all ssdomain and bonds
        # this method is more reasonable, but much slower than simple loops and requests too much memory;
        """mask = np.ones(int(((N - 1) * N) / 2), dtype=bool)
        for domain in ssdomains:
            folded = np.array(list(itertools.combinations(range(domain[0] - 1, domain[-1]), 2)))
            mask &= ~(pairs[:, None] == folded).all(-1).any(-1)
        mask &= np.squeeze(np.abs(pairs[:, 0] - pairs[:, 1]) > 1)"""
        if name in multidomain_names:
            N = len(fasta)  # number of residues
            ssdomains = protein_repo.get_ssdomains(name, f'{cwd}/domains.yaml')
            mask = []
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
        else:  # IDP
            mask = np.abs(pairs[:, 0] - pairs[:, 1]) > 1  # exclude bonds
        pairs = pairs[mask]
        d = md.compute_distances(traj,pairs)
        d[d>rc] = np.inf  # cutoff
        r = np.copy(d)
        n1 = np.zeros(r.shape,dtype=np.int8)
        n2 = np.zeros(r.shape,dtype=np.int8)
        pairs = np.array(list(itertools.combinations(fasta,2)))
        pairs = pairs[mask] # exclude bonded
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
        term_2 = sigmas6*(d6+ncut2/np.power(rc,6)) + sigmas12*(d12-ncut2/np.power(rc,12))
        term_2 = 4*eps*term_2 - term_1
        np.save(f'{cwd}/{dataset}/{prot.path}/energy_sums_1.npy',term_1.sum(axis=1))
        np.save(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy',term_2)
        np.save(f'{cwd}/{dataset}/{prot.path}/unique_pairs.npy',unique_pairs)

def calcAHenergy(cwd,dataset, df,prot):
    term_1 = np.load(f'{cwd}/{dataset}/{prot.path}/energy_sums_1.npy')
    term_2 = np.load(f'{cwd}/{dataset}/{prot.path}/energy_sums_2.npy')
    unique_pairs = np.load(f'{cwd}/{dataset}/{prot.path}/unique_pairs.npy')
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

def reweight(cwd, dataset,dp,df,proteinPRE,proteinsRgs):
    multidomain_names = ["THB_C2", "Ubq2", "Ubq3", "Gal3", "TIA1", "Ubq4", "hnRNPA1", "C5_C6_C7", "hSUMO_hnRNPA1", 'GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']
    IDP_names = np.setdiff1d(list(proteinsRgs.index)+list(proteinPRE.index), multidomain_names).tolist()
    Rgloss4multidomain = np.ones(shape=len(multidomain_names))
    Rgloss4IDPs = np.ones(shape=len(IDP_names))
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

df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle-1}.csv').set_index('three')
logging.info(df.lambdas)

for name in proteinsPRE.index:
    # type(proteinsPRE.loc["aSyn"]["expPREs"])-> <class 'pandas.core.frame.DataFrame'>
    proteinsPRE.at[name,'expPREs'] = loadExpPREs(cwd,dataset, name,proteinsPRE.loc[name])

time0 = time.time()
# proteinsPRE["weights"]
# aSyn      False
# OPN       False
# FUS       False
# FUS12E    False

ray.get([evaluatePRE.remote(cwd,dataset,label,name,proteinsPRE.loc[name]) for n,(label,name) in enumerate(proc_PRE)])
logging.info('Timing evaluatePRE {:.3f}'.format(time.time()-time0))


time0 = time.time()

ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[:10].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[10:20].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[20:30].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[30:40].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[40:50].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[50:60].iterrows()])
ray.get([calcDistSums.remote(cwd,dataset,df,name,prot) for name,prot in pd.concat((proteinsPRE,proteinsRgs),sort=True).iloc[60:].iterrows()])
logging.info('Timing calcDistSums {:.3f}'.format(time.time()-time0))

for name in proteinsPRE.index:
    np.save(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/{name}_AHenergy.npy',
        calcAHenergy(cwd,dataset, df,proteinsPRE.loc[name]))
    tau_c, chi2_pre = optTauC(cwd,dataset, proteinsPRE.loc[name])
    proteinsPRE.at[name,'tau_c'] = tau_c
    proteinsPRE.at[name,'chi2_pre'] = chi2_pre
    proteinsPRE.at[name,'initPREs'] = loadInitPREs(cwd, dataset, name,proteinsPRE.loc[name])
    if os.path.exists(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs'):
        shutil.rmtree(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs')  # delete files recursively
    shutil.copytree(f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/calcPREs',f'{cwd}/{dataset}/{proteinsPRE.loc[name].path}/initPREs')  # copy dir
proteinsPRE.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsPRE.pkl')

for name in proteinsRgs.index:
    np.save(f'{cwd}/{dataset}/{proteinsRgs.loc[name].path}/{name}_AHenergy.npy',
        calcAHenergy(cwd,dataset, df,proteinsRgs.loc[name]))
    rgarray, rg, chi2_rg = calcRg(cwd,dataset, df,name,proteinsRgs.loc[name])
    proteinsRgs.at[name,'rgarray'] = rgarray
    proteinsRgs.at[name,'Rg'] = rg
    proteinsRgs.at[name,'chi2_rg'] = chi2_rg
proteinsRgs.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsRgs.pkl')

logging.info('Initial Chi2 PRE {:.3f} +/- {:.3f}'.format(proteinsPRE.chi2_pre.mean(),proteinsPRE.chi2_pre.std()))
logging.info('Initial Chi2 Gyration Radius {:.3f} +/- {:.3f}'.format(proteinsRgs.chi2_rg.mean(),proteinsRgs.chi2_rg.std()))

selHPS = pd.read_csv(f'{cwd}/selHPS.csv',index_col=0)

kde = KernelDensity(kernel='gaussian',bandwidth=0.05).fit(selHPS.T.values)
theta_prior = theta * kde.score_samples(df.lambdas.values.reshape(1, -1))[0]

xi = xi_0

variants = ['A1','M12FP12Y','P7FM7Y','M9FP6Y','M8FP4Y','M9FP3Y','M10R','M6R','P2R','P7R','M3RP3K','M6RP6K','M10RP10K','M4D','P4D','P8D','P12D','P12E','P7KP12D','P7KP12Db','M12FP12YM10R','M10FP7RP12D']

logging.info('Initial theta*prior {:.2f}'.format(theta_prior))
logging.info('theta {:.2f}'.format(theta))
logging.info('xi {:g}'.format(xi))

dfchi2 = pd.DataFrame(columns=['chi2_pre','chi2_rg','theta_prior','lambdas','xi'])
dfchi2.loc[0] = [proteinsPRE.chi2_pre.mean(),proteinsRgs.chi2_rg.mean(),theta_prior,df.lambdas,xi]

time0 = time.time()

micro_cycle = 0

logging.info(df.lambdas)
df[f'lambdas_{cycle-1}'] = df.lambdas
Rgloss_multidomain = pd.DataFrame(columns=[multidomain_names])
Rgloss_IDP = pd.DataFrame(columns=np.setdiff1d(list(proteinsRgs.index)+list(proteinsPRE.index), multidomain_names).tolist())
for k in range(2,200000):  # 200000
    if (xi<1e-8):
        xi = xi_0
        micro_cycle += 1
        if (micro_cycle==10):
            logging.info('xi {:g}'.format(xi))
            break

    xi = xi * .99
    passed, trial_df, trialPRE, trialRgs, Rgloss4multidomain, Rgloss4IDPs = reweight(cwd, dataset, dp,df,proteinsPRE,proteinsRgs)
    if passed:
        theta_prior = theta * kde.score_samples(trial_df.lambdas.values.reshape(1, -1))[0]
        delta1 = eta*trialPRE.chi2_pre.mean() + trialRgs.chi2_rg.mean() - theta_prior
        delta2 = eta*proteinsPRE.chi2_pre.mean() + proteinsRgs.chi2_rg.mean() - dfchi2.iloc[-1]['theta_prior']
        delta = delta1 - delta2
        if ( np.exp(-delta/xi) > np.random.rand() ):
            proteinsPRE = trialPRE.copy()
            proteinsRgs = trialRgs.copy()
            Rgloss_multidomain.loc[k-1] = Rgloss4multidomain
            Rgloss_IDP.loc[k-1] = Rgloss4IDPs
            df = trial_df.copy()
            dfchi2.loc[k-1] = [trialPRE.chi2_pre.mean(),trialRgs.chi2_rg.mean(),theta_prior,df.lambdas,xi]
            logging.info('Acc Iter {:d}, micro cycle {:d}, xi {:g}, Chi2 PRE {:.2f}, Chi2 Rg {:.2f}, theta*prior {:.2f}'.format(k-1,micro_cycle,xi,trialPRE.chi2_pre.mean(),trialRgs.chi2_rg.mean(),theta_prior))

logging.info('Timing Reweighting {:.3f}'.format(time.time()-time0))
logging.info('Theta {:.3f}'.format(theta))

dfchi2['cost'] = dfchi2.chi2_rg + eta*dfchi2.chi2_pre - dfchi2.theta_prior
dfchi2.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_chi2.pkl')
df.lambdas = dfchi2.loc[pd.to_numeric(dfchi2['cost']).idxmin()].lambdas
logging.info(df.lambdas)
df[f'lambdas_{cycle}'] = df.lambdas
proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl')
Rgloss_multidomain.to_pickle(f'{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl')
Rgloss_IDP.to_pickle(f'{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl')
df.to_csv(f'{cwd}/{dataset}/residues_{cycle}.csv')
logging.info('Cost at 0: {:.2f}'.format(dfchi2.loc[0].cost))
logging.info('Min Cost at {:d}: {:.2f}'.format(pd.to_numeric(dfchi2['cost']).idxmin(),dfchi2.cost.min()))

time_end = time.time()
target_seconds = time_end - time_total  # total used time
logging.info(f"total optimization used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")