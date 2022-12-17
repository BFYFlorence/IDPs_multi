import pandas as pd
import numpy as np
import mdtraj as md
import itertools
from DEERPREdict.utils import Operations

def loadExpPREs(cwd,dataset, name,prot):
    value = {}
    error = {}
    resnums = np.arange(1,len(prot.fasta)+1)
    for label in prot.labels:
        value[label], error[label] = np.loadtxt(f'{cwd}/{dataset}/{name}/expPREs/exp-{label}.dat',unpack=True)
    v = pd.DataFrame(value,index=resnums)
    v.rename_axis('residue', axis='index', inplace=True)
    v.rename_axis('label', axis='columns',inplace=True)
    e = pd.DataFrame(error,index=resnums)
    e.rename_axis('residue', axis='index', inplace=True)
    e.rename_axis('label', axis='columns',inplace=True)
    return pd.concat(dict(value=v,error=e),axis=1)

def loadInitPREs(cwd, dataset, name,prot):
    obs = 1 if prot.obs=='ratio' else 2
    value = {}
    resnums = np.arange(1,len(prot.fasta)+1)
    for label in prot.labels:
        value[label] = np.loadtxt(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.dat')[:,obs]
    v = pd.DataFrame(value,index=resnums)
    v.rename_axis('residue', inplace=True)
    v.rename_axis('label', axis='columns',inplace=True)
    return v

def calcChi2(cwd,dataset, prot):
    obs = 1 if prot.obs=='ratio' else 2
    chi2 = 0
    for label in prot.labels:
        y = np.loadtxt(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.dat')[:,obs]
        chi = (prot.expPREs.value[label].values - y) / prot.expPREs.error[label].values
        chi = chi[~np.isnan(chi)]
        chi2 += np.nansum( np.power( chi, 2) ) / chi.size
    return chi2 / len(prot.labels)

def optTauC(cwd,dataset, prot):
    obs = 1 if prot.obs == 'ratio' else 2
    chi2list = []
    tau_c = np.arange(2,10.05,1)
    for tc in tau_c:
        chi2 = 0
        for label in prot.labels:
            # the first two columns in res-{label}.dat
            x,y = np.loadtxt(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.dat',usecols=(0,1),unpack=True)
            # x: [  1.   2.   3. ... 218. 219. 220.] (number of residue)
            # y: [nan 0.14382832 0.28971916 ... 0.99804923 0.99705705 0.99578599] (number of residue)

            # index of residues with a real value (start with 0)
            measured_resnums = np.where(~np.isnan(y))[0]
            data = pd.read_pickle(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.pkl', compression='gzip')
            # the reason why nan shows up is that the corresponding steric partition function is under cutoff.
            # So it is discarded
            # print(data)
            # r3[[nan, ... nan], ... [1.47580169e-04, ... 3.25930228e-06]]  (traj_len, measured_resnums_len)
            # r6 similar to r3, but distributions of nan are not necessarily the same
            # angular the similar as the above
            # print(data.index)  Index(['r3', 'r6', 'angular'], dtype='object')
            gamma_2_av = np.full(y.size, fill_value=np.NaN)
            s_pre = np.power(data['r3'], 2)/data['r6']*data['angular']  # calculate following the formula
            gamma_2 = Operations.calc_gamma_2(data['r6'], s_pre, tau_c = tc * 1e-9, tau_t = 1e-10, wh = prot.wh, k = 1.23e16)  # calculate following the formula
            # print(gamma_2.shape)  (traj_len, measured_resnums_len)
            gamma_2 = np.ma.MaskedArray(gamma_2, mask = np.isnan(gamma_2))
            gamma_2_av[measured_resnums] = np.ma.average(gamma_2, axis=0).data  # averaged over traj
            # For samples with particularly high PRE rates it can be infeasible to obtain Γ2 from 174 multiple time-point measurements,
            # https://doi.org/10.1101/2020.08.09.243030
            if prot.obs == 'ratio':
                y = 10 * np.exp(-gamma_2_av * 0.01) / ( 10 + gamma_2_av )
            else:
                y = gamma_2_av

            # calculate chi
            chi = (prot.expPREs.value[label].values - y) / prot.expPREs.error[label].values
            chi = chi[~np.isnan(chi)]
            chi2 += np.nansum( np.power( chi, 2) ) / chi.size
        chi2list.append(chi2 / len(prot.labels))

    tc_min = tau_c[np.argmin(chi2list)]  # pick up the smallest value

    for label in prot.labels:
        x,y = np.loadtxt(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.dat',usecols=(0,1),unpack=True)
        measured_resnums = np.where(~np.isnan(y))[0]
        data = pd.read_pickle(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.pkl', compression='gzip')
        gamma_2_av = np.full(y.size, fill_value=np.NaN)
        s_pre = np.power(data['r3'], 2)/data['r6']*data['angular']
        gamma_2 = Operations.calc_gamma_2(data['r6'], s_pre, tau_c = tc_min * 1e-9, tau_t = 1e-10, wh = prot.wh, k = 1.23e16)
        gamma_2 = np.ma.MaskedArray(gamma_2, mask = np.isnan(gamma_2))
        gamma_2_av[measured_resnums] = np.ma.average(gamma_2, axis=0).data
        i_ratio = 10 * np.exp(-gamma_2_av * 0.01) / ( 10 + gamma_2_av )
        np.savetxt(f'{cwd}/{dataset}/{prot.path}/calcPREs/res-{label}.dat',np.c_[x,i_ratio,gamma_2_av])

    return tc_min, calcChi2(cwd, dataset, prot)

def reweightRg(df,name,prot):
    #rg = np.sqrt( np.dot(np.power(prot.rgarray,2), prot.weights) )
    rg = np.dot(prot.rgarray, prot.weights)
    chi2_rg = np.power((prot.expRg-rg)/prot.expRgErr,2)
    #chi2_rg = np.power((prot.expRg-rg)/(prot.expRg*0.03),2)
    return rg, chi2_rg

def calcRg(cwd,dataset, df,name,prot):
    t = md.load_dcd(f"{cwd}/{dataset}/{prot.path}/{name}.dcd",f"{cwd}/{dataset}/{prot.path}/{name}.pdb")
    residues = [res.name for res in t.top.atoms]
    masses = df.loc[residues,'MW'].values
    masses[0] += 2
    masses[-1] += 16
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rgarray = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    np.save(f"{cwd}/{dataset}/{prot.path}/Rg_traj.npy", rgarray)
    #rg = np.sqrt( np.power(rgarray, 2).mean() )
    rg = rgarray.mean()
    chi2_rg = np.power((prot.expRg-rg)/prot.expRgErr,2)
    #chi2_rg = np.power((prot.expRg-rg)/(prot.expRg*0.03),2)
    return rgarray, rg, chi2_rg

def initProteinsPRE(cycle):
    outdir = f'/{cycle}'
    proteins = pd.DataFrame(columns=['labels','wh','tau_c','temp','obs','pH','ionic','expPREs','initPREs','eff','chi2_pre','fasta','weights','path'],dtype=object)
    fasta_OPN = """MHQDHVDSQSQEHLQQTQNDLASLQQTHYSSEENADVPEQPDFPDV
PSKSQETVDDDDDDDNDSNDTDESDEVFTDFPTEAPVAPFNRGDNAGRGDSVAYGFRAKA
HVVKASKIRKAARKLIEDDATTEDGDSQPAGLWWPKESREQNSRELPQHQSVENDSRPKF
DSREVDGGDSKASAGVDSRESQGSVPAVDASNQTLESAEDAEDRHSIENNEVTR""".replace('\n', '')
    fasta_FUS = """MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ
SQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSS
SYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')
    fasta_FUS12E = """GMASNDYEQQAEQSYGAYPEQPGQGYEQQSEQPYGQQSYSGYEQSTDTSGYGQSSYSSYGQ
EQNTGYGEQSTPQGYGSTGGYGSEQSEQSSYGQQSSYPGYGQQPAPSSTSGSYGSSEQSS
SYGQPQSGSYEQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')
    fasta_aSyn = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK
EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDP
DNEAYEMPSEEGYQDYEPEA""".replace('\n', '')
    proteins.loc['aSyn'] = dict(labels=[24, 42, 62, 87, 103], tau_c=1.0,
                               wh=700,temp=283,obs='ratio',pH=7.4,fasta=list(fasta_aSyn),ionic=0.2,weights=False,path='aSyn'+outdir)
    proteins.loc['OPN'] = dict(labels=[10, 33, 64, 88, 117, 130, 144, 162, 184, 203], tau_c=3.0,
                               wh=800,temp=298,obs='rate',pH=6.5,fasta=list(fasta_OPN),ionic=0.15,weights=False,path='OPN'+outdir)
    proteins.loc['FUS'] = dict(labels=[16, 86, 142], tau_c=10.0,
                               wh=850,temp=298,obs='rate',pH=5.5,fasta=list(fasta_FUS),ionic=0.15,weights=False,path='FUS'+outdir)
    proteins.loc['FUS12E'] = dict(labels=[16, 86, 142], tau_c=10.0,
                               wh=850,temp=298,obs='rate',pH=5.5,fasta=list(fasta_FUS12E),ionic=0.15,weights=False,path='FUS12E'+outdir)
    return proteins

def initProteinsRgs(cycle, exclusion_GS):
    outdir = '/{:d}'.format(cycle)
    proteins = pd.DataFrame(columns=['temp','expRg','expRgErr','Rg','rgarray','eff','chi2_rg','weights','pH','ionic','fasta','path'],dtype=object)
    fasta_GHRICD = """SKQQRIKMLILPPVPVPKIKGIDPDLLKEGKLEEVNTILAIHDSYKPEFHSDDSWVEFIELDIDEPDEKTEESDTDRLLSSDHEKSHSNL
GVKDGDSGRTSCCEPDILETDFNANDIHEGTSEVAQPQRLKGEADLLCLDQKNQNNSPYHDACPATQQPSVIQAEKNKPQPLPTEGAESTHQAAH
IQLSNPSSLSNIDFYAQVSDITPAGSVVLSPGQKNKAGMSQCDMHPEMVSLCQENFLMDNAYFCEADAKKCIPVAPHIKVESHIQPSLNQEDIYI
TTESLTTAAGRPGTGEHVPGSEMPVPDYTSIHIVQSPQGLILNATALPLPDKEFLSSCGYVSTDQLNKIMP""".replace('\n', '')
    fasta_Ash1 = """SASSSPSPSTPTKSGKMRSRSSSPVRPKAYTPSPRSPNYHRFALDSPPQSPRRSSNSSITKKGSRRSSGSSPTRHTTRVCV"""
    fasta_CTD2 = """FAGSGSNIYSPGNAYSPSSSNYSPNSPSYSPTSPSYSPSSPSYSPTSPCYSPTSPSYSPTSPNYTPVTPSYSPTSPNYSASPQ"""
    fasta_Hst52 = """DSHAKRHHGYKRKFHEKHHSHRGYDSHAKRHHGYKRKFHEKHHSHRGY""".replace('\n', '') # DOI: 10.1021/acs.jpcb.0c09635
    fasta_Hst5 = """DSHAKRHHGYKRKFHEKHHSHRGY""".replace('\n', '')
    fasta_aSyn140 = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK
EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA""".replace('\n', '')
    fasta_PNt = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSS
GQLSDDGIRRFLGTVTVKAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGG
VQIERGANVTVQRSAIVDGGLHIGALQSLQPEDLPPSRVVLRDTNVTAVPASGAPAAVSVLGAS
ELTLDGGHITGGRAAGVAAMQGAVVHLQRATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGP
VLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAP
QAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS1 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSS
GQKSDDGIRRFLGTVTVLAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGG
VQIERGANVTVQRSAIVLGGLHIGALQSLQPEDDPPSRVVLRDTNVTAVPASGAPAAVSVLGAS
LLTLDGGHITGGRAAGVAAMQGAVVHEQRATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGP
VLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAP
QAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS4 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSS
GQLSFVGITRDLGRDTVKAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGG
VQIERGADVRVQREAIVDGGLHNGALQSLQPSILPPSTVVLRDTNVTAVPASGAPAAVLVSGAS
GLRLDGGHIHEGRAAGVAAMQGAVVTLQTATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGP
VLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAP
QAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS5 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSS
GQLSDDGIEDFLGTVTVDAGELVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGG
VQIEDGANVTVQESAIVDGGLHIGALQSLQPRRLPPSRVVLRKTNVTAVPASGAPAAVSVLGAS
KLTLRGGHITGGRAAGVAAMQGAVVHLQRATIRRGRALAGGAVPGGAVPGGAVPGGFGPGGFGP
VLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAP
QAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS6 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSS
GQLSDRGIDRFLGTVTVEAGKLVADHATLANVGDTWDKDGIALYVAGRQAQASIADSTLQGAGG
VQIREGANVTVQRSAIVDGGLHIGALQSLQPERLPPSDVVLRDTNVTAVPASGAPAAVSVLGAS
RLTLDGGHITGGDAAGVAAMQGAVVHLQRATIERGEALAGGAVPGGAVPGGAVPGGFGPGGFGP
VLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAP
QAAPLSITLQAGAH""".replace('\n','')
    fasta_ACTR = """GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_RNaseA = """KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSI
TDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV""".replace('\n', '')
    fasta_p15PAF = """MVRTKADSVPGTYRKVVAARAPRKVLGSSTSATNSTSVSSRKAENKYAGGNPVCVRPTPKWQKGIGEFFR
LSPKDSEKENQIPEEAGSSGLGKAKRKACPLQPDHTNDEKE""".replace('\n', '') # DOI: 10.1016/j.bpj.2013.12.046
    fasta_CoRNID = """GPHMQVPRTHRLITLADHICQIITQDFARNQVPSQASTSTFQTSPSALSSTPVRTKTSSRYS
PESQSQTVLHPRPGPRVSPENLVDKSRGSRPGKSPERSHIPSEPYEPISPPQGPAVHEKQDSMLLLSQRGVDPAEQRSDSRSP
GSISYLPSFFTKLESTSPMVKSKKQEIFRKLNSSGGGDSDMAAAQPGTEIFNLPAVTTSGAVSSRSHSFADPASNLGLEDIIR
KALMGSFDDKVEDHGVVMSHPVGIMPGSASTSVVTSSEARRDE""".replace('\n', '') # SASDF34
    fasta_Sic1 = """GSMTPSTPPRSRGTRYLAQPSGNTSSSALMQGQKTPQKPS
QNLVPVTPSTTKSFKNAPLLAPPNSNMGMTSPFNGLTSPQRSPFPKSSVKRT""".replace('\n', '')
    fasta_FhuA = """SESAWGPAATIAARQSATGTKTDTPIQKVPQSISVVTAEEMALHQPKSVKEALSYTPGVSVGTRGASNTYDHLIIRGFAAEGQS
QNNYLNGLKLQGNFYNDAVIDPYMLERAEIMRGPVSVLYGKSSPGGLLNMVSKRPTTEPL""".replace('\n', '')
    fasta_K44 =  """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKA
KGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRL
QTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIE""".replace('\n', '')
    fasta_K10 = """MQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVK
SEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_K27 = """MSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIV
YKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVY""".replace('\n', '')
    fasta_K25 = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDK
KAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRL""".replace('\n', '')
    fasta_K32 = """MSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQII
NKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVY""".replace('\n', '')
    fasta_K23 = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDK
KAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRL
THKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M12FP12Y = """GSMASASSSQRGRSGSGNYGGGRGGGYGGNDNYGRGGNYSGRGGYGGSRGGGGYGGSGDGYNGYGNDGSNYGGGGSYNDYGNYNNQ
SSNYGPMKGGNYGGRSSGGSGGGGQYYAKPRNQGGYGGSSSSSSYGSGRRY""".replace('\n', '')
    fasta_P7FM7Y = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGFGGSGDGFNGFGNDGSNFGGGGSFNDFGNFNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQFFAKPRNQGGFGGSSSSSSFGSGRRF""".replace('\n', '')
    fasta_M9FP6Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNYGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNYNNQ
SSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRY""".replace('\n', '')
    fasta_M8FP4Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNYNNQ
SSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M9FP3Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNGNNQ
SSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRS""".replace('\n', '')
    fasta_M10R = """GSMASASSSQGGSSGSGNFGGGGGGGFGGNDNFGGGGNFSGSGGFGGSGGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGGF""".replace('\n', '')
    fasta_M6R = """GSMASASSSQGGRSGSGNFGGGRGGGFGGNDNFGGGGNFSGSGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGRF""".replace('\n', '')
    fasta_P2R = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFRNDGSNFGGGGRYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P7R = """GSMASASSSQRGRSGRGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGRYGGSGDRYNGFGNDGRNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFRGRSSGPYGRGGQYFAKPRNQGGYGGSSSSRSYGSGRRF""".replace('\n', '')
    fasta_M3RP3K = """GSMASASSSQRGKSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRKF""".replace('\n', '')
    fasta_M6RP6K = """GSMASASSSQKGKSGSGNFGGGRGGGFGGNDNFGKGGNFSGRGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGKSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRKF""".replace('\n', '')
    fasta_M10RP10K = """GSMASASSSQKGKSGSGNFGGGKGGGFGGNDNFGKGGNFSGKGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGKSSGGSGGGGQYFAKPKNQGGYGGSSSSSSYGSGKKF""".replace('\n', '')
    fasta_M4D = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNGNFGRGGNFSGRGGFGGSRGGGGYGGSGGGYNGFGNSGSNFGGGGSYNGFGNYNNQ
SSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P4D = """GSMASASSSQRDRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGDFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQGGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_P8D = """GSMASASSSQRDRSGSGNFGGGRDGGFGGNDNFGRGDNFSGRGDFGGSRDGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQDGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_P12D = """GSMASADSSQRDRDDSGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGGYGGDGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFDPMKGGNFGDRSSGPYDGGGQYFAKPRNQGGYGGSSSSSSYGSDRRF""".replace('\n', '')
    fasta_P12E = """GSMASAESSQREREESGNFGEGRGGGFGGNDNFGRGGNFSERGGFGGSRGEGGYGGEGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFEPMKGGNFGERSSGPYEGGGQYFAKPRNQGGYGGSSSSSSYGSERRF""".replace('\n', '')
    fasta_P7KP12D = """GSMASADSSQRDRDDKGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGKYGGDGDKYNGFGNDGKNFGGGGSYNDFGNYNNQ
SSNFDPMKGGNFKDRSSGPYDKGGQYFAKPRNQGGYGGSSSSKSYGSDRRF""".replace('\n', '')
    fasta_P7KP12Db = """GSMASAKSSQRDRDDDGNFGKGRGGGFGGNKNFGRGGNFSKRGGFGGSRGKGKYGGKGDDYNGFGNDGDNFGGGGSYNDFGNYNNQ
SSNFDPMDGGNFDDRSSGPYDDGGQYFADPRNQGGYGGSSSSKSYGSKRRF""".replace('\n', '')
    fasta_M12FP12YM10R = """GSMASASSSQGGSSGSGNYGGGGGGGYGGNDNYGGGGNYSGSGGYGGSGGGGGYGGSGDGYNGYGNDGSNYGGGGSYNDYGNYNNQ
SSNYGPMKGGNYGGSSSGPYGGGGQYYAKPGNQGGYGGSSSSSSYGSGGGY""".replace('\n', '')
    fasta_M10FP7RP12D = """GSMASADSSQRDRDDRGNFGDGRGGGGGGNDNFGRGGNGSDRGGGGGSRGDGRYGGDGDRYNGGGNDGRNGGGGGSYNDGGNYNNQ
SSNGDPMKGGNGRDRSSGPYDRGGQYGAKPRNQGGYGGSSSSRSYGSDRRG""".replace('\n', '')
    fasta_SH4UD = """MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAA
EPKLFGGFNSSDTVTSPQRAGPLAGGSAWSHPQFEK""".replace('\n', '') # DOI:
    fasta_hNL3cyt = """MYRKDKRRQEPLRQPSPQRGAWAPELGAAPEEELAALQLGPTHHECEAG
PPHDTLRLTALPDYTLTLRRSPDDIPLMTPNTITMIPNSLVG
LQTLHPYNTFAAGFNSTGLPHSHSTTRV""".replace('\n', '') # DOI: 10.1529/biophysj.107.126995
    fasta_ColNT = """MGSNGADNAHNNAFGGGKNPGIGNTSGAGSNGSASSNRGNSNGWSWSNKPHKNDGFHSDGSYHITFHGDNNSKPKPGGNSGNRGNNGDGASSHHHHHH""".replace('\n', '') # SASDC53
    fasta_tau35 = """EPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAP
VPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGS
VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTD
HGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_CAHSD = """MSGRNVESHMERNEKVVVNNSGHADVKKQQQQVEHTEFTHTEVKAPLIHPAPPIISTGAAGLA
EEIVGQGFTASAARISGGTAEVHLQPSAAMTEEARRDQERYRQEQESIAKQQEREMEKKTEAYRKT
AEAEAEKIRKELEKQHARDVEFRKDLIESTIDRQKREVDLEAKMAKRELDREGQLAKEALERSRLA
TNVEVNFDSAAGHTVSGGTTVSTSDKMEIKRN""".replace('\n','')
    fasta_p532070 = """GPGSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAALEHHHHHH""" # DOI: 10.1038/s41467-021-21258-5

    # full-length hnRNPA1 (FL-A1)
    fasta_hnRNPA1 = """MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFV
TYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHH
LRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKA
LSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGS
GDGYNGFGNDGSNFGGGGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYG
GSSSSSSYGSGRRF""".replace('\n', '')
    fasta_THB_C2 = """GPGSEDVWEILRQAPPSEYERIAFQYGVTDLRGMLKRLKGMRRDEKKSTAFQKKLEPAYQ
VSKGHKIRLTVELADHDAEVKWLKNGQEIQMSGSKYIFESIGAKRTLTISQCSLADDAAY
QCVVGGEKCSTELFVKE""".replace('\n', '')
    fasta_Ubq2 = """MASHHHHHHGAQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQL
EDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKE
GIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG""".replace('\n', '')
    fasta_Ubq3 = """MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN
IQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLI
FAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIENVKA
KIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG""".replace('\n', '')
    fasta_Gal3 = """MADNFSLHDALSGSGNPNPQGWPGAWGNQPAGAGGYPGASYPGAYPGQAPPGAYPGQAPP
GAYHGAPGAYPGAPAPGVYPGPPSGPGAYPSSGQPSAPGAYPATGPYGAPAGPLIVPYNL
PLPGGVVPRMLITILGTVKPNANRIALDFQRGNDVAFHFNPRFNENNRRVIVCNTKLDNN
WGREERQSVFPFESGKPFKIQVLVEPDHFKVAVNDAHLLQYNHRVKKLNEISKLGISGDI
DLTSASYTMI""".replace('\n', '')
    fasta_TIA1 = """GEDEMPKTLYVGNLSRDVTEALILQLFSQIGPCKNCKMIMDTAGNDPYCFVEFHEHRHAA
AALAAMNGRKIMGKEVKVNWATTPSSQKLPQTGNHFHVFVGDLSPEITTEDIKAAFAPFG
RISDARVVKDMATGKSKGYGFVSFFNKWDAENAIQQMGGQWLGGRQIRTNWATRKPPAPK
STYESNTKQLSYDEVVNQSSPSNCTVYCGGVTSGLTEQLMRQTFSPFGQIMEIRVFPDKG
YSFVRFNSHESAAHAIVSVNGTTIEGHVVKCYWGK""".replace('\n', '')
    fasta_Ubq4 = """MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN
IQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLI
FAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIENVKA
KIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKT
ITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLR
LRGG""".replace('\n', '')
    fasta_C5_C6_C7 = """GPGSRQEPPKIHLDCPGRIPDTIVVVAGNKLRLDVPISGDPAPTVIWQKAITQGNKAPAR
PAPDAPEDTGDSDEWVFDKKLLCETEGRVRVETTKDRSIFTVEGAEKEDEGVYTVTVKNP
VGEDQVNLTVKVIDVPDAPAAPKISNVGEDSCTVQWEPPAYDGGQPILGYILERKKKKSY
RWMRLNFDLIQELSHEARRMIEGVVYEMRVYAVNAIGMSRPSPASQPFMPIGPPSEPTHL
AVEDVSDTTVSLKWRPPERVGAGGLDGYSVEYCPEGCSEWVAALQGLTEHTSILVKDLPT
GARLLFRVRAHNMAGPGAPVTTTEPVTV""".replace('\n', '')
    fasta_hSUMO_hnRNPA1 = """MGSSHHHHHHGSGLVPRGSASMSDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRLMEAFAKRQGKEMDSLRFLYDGIRIQADQTPEDLDMEDNDIIEAHREQIGGMSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVT
YATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHL
RDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKAL
SKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSG
DGYNGFGNDGSNFGGGGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGG
SSSSSSYGSGRRF""".replace('\n', '')
    fasta_GS0 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SKLMVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQ
FSPWILVPHIGYGFHQYLPYPDGMSPFQAAMVDGSGYQVHRTMQFEDGASLTVNYRYTYE
GSHIKGEAQVKGTGFPADGPVMTNSLTAADWCRSKKTYPNDKTIISTFKWSYTTGNGKRY
RSTARTTYTFAKPMAANYLKNQPMYVFRKTELKHSKTELNFKEWQKAFTD""".replace('\n', '')
    fasta_GS8 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SGSGSGSGSGSGSGSGSKLMVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPN
DGYEELNLKSTKGDLQFSPWILVPHIGYGFHQYLPYPDGMSPFQAAMVDGSGYQVHRTMQ
FEDGASLTVNYRYTYEGSHIKGEAQVKGTGFPADGPVMTNSLTAADWCRSKKTYPNDKTI
ISTFKWSYTTGNGKRYRSTARTTYTFAKPMAANYLKNQPMYVFRKTELKHSKTELNFKEW
QKAFTD""".replace('\n', '')
    fasta_GS16 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSKLMVSKGEEDNMASLPATHELHIFGSI
NGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQFSPWILVPHIGYGFHQYLPYPDGMSPFQ
AAMVDGSGYQVHRTMQFEDGASLTVNYRYTYEGSHIKGEAQVKGTGFPADGPVMTNSLTA
ADWCRSKKTYPNDKTIISTFKWSYTTGNGKRYRSTARTTYTFAKPMAANYLKNQPMYVFR
KTELKHSKTELNFKEWQKAFTD""".replace('\n', '')
    fasta_GS24 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSKLMVSKGEEDN
MASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQFSPWILVPHIGY
GFHQYLPYPDGMSPFQAAMVDGSGYQVHRTMQFEDGASLTVNYRYTYEGSHIKGEAQVKG
TGFPADGPVMTNSLTAADWCRSKKTYPNDKTIISTFKWSYTTGNGKRYRSTARTTYTFAK
PMAANYLKNQPMYVFRKTELKHSKTELNFKEWQKAFTD""".replace('\n', '')
    fasta_GS32 = """SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV
TTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN
RIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADH
YQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGL
SGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSG
SGSGSKLMVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTK
GDLQFSPWILVPHIGYGFHQYLPYPDGMSPFQAAMVDGSGYQVHRTMQFEDGASLTVNYR
YTYEGSHIKGEAQVKGTGFPADGPVMTNSLTAADWCRSKKTYPNDKTIISTFKWSYTTGN
GKRYRSTARTTYTFAKPMAANYLKNQPMYVFRKTELKHSKTELNFKEWQKAFTD""".replace('\n', '')
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


    if not exclusion_GS:
        proteins.loc['GS48'] = dict(temp=293, expRg=4.11, expRgErr=0.02, pH=7.0, fasta=list(fasta_GS48), ionic=0.15,
                                    path='GS48' + outdir)  # Tórur's thesis, ph is not checked yet
        proteins.loc['GS32'] = dict(temp=293, expRg=3.78, expRgErr=0.02, pH=7.0, fasta=list(fasta_GS32), ionic=0.15,
                                    path='GS32' + outdir)  # Tórur's thesis, ph is not checked yet
        proteins.loc['GS24'] = dict(temp=293, expRg=3.57, expRgErr=0.01, pH=7.0, fasta=list(fasta_GS24), ionic=0.15,
                                    path='GS24' + outdir)  # Tórur's thesis, ph is not checked yet
        proteins.loc['GS16'] = dict(temp=293, expRg=3.45, expRgErr=0.01, pH=7.0, fasta=list(fasta_GS16), ionic=0.15,
                                    path='GS16' + outdir)  # Tórur's thesis, ph is not checked yet
        proteins.loc['GS8'] = dict(temp=293, expRg=3.37, expRgErr=0.01, pH=7.0, fasta=list(fasta_GS8), ionic=0.15,
                                   path='GS8' + outdir)  # Tórur's thesis, ph is not checked yet
        proteins.loc['GS0'] = dict(temp=293, expRg=3.20, expRgErr=0.01, pH=7.0, fasta=list(fasta_GS0), ionic=0.15,
                                   path='GS0' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['hSUMO_hnRNPA1'] = dict(temp=300, expRg=3.37, expRgErr=0.01, pH=7.0, fasta=list(fasta_hSUMO_hnRNPA1),
                                         ionic=0.1,
                                         path='hSUMO_hnRNPA1' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['C5_C6_C7'] = dict(temp=298, expRg=3.75, expRgErr=0.01, pH=7.0, fasta=list(fasta_C5_C6_C7), ionic=0.28, path='C5_C6_C7' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['Ubq4'] = dict(temp=293, expRg=3.19, expRgErr=0.02, pH=7.0, fasta=list(fasta_Ubq4), ionic=0.33, path='Ubq4' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['TIA1'] = dict(temp=300, expRg=2.75, expRgErr=0.03, pH=7.0, fasta=list(fasta_TIA1), ionic=0.1, path='TIA1' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['Gal3'] = dict(temp=303, expRg=2.91, expRgErr=0.01, pH=7.0, fasta=list(fasta_Gal3), ionic=0.04, path='Gal3' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['Ubq3'] = dict(temp=293, expRg=2.62, expRgErr=0.01, pH=7.0, fasta=list(fasta_Ubq3), ionic=0.33, path='Ubq3' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['Ubq2'] = dict(temp=293, expRg=2.19, expRgErr=0.02, pH=7.0, fasta=list(fasta_Ubq2), ionic=0.33, path='Ubq2' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['THB_C2'] = dict(temp=277, expRg=1.909, expRgErr=0.003, pH=7.0, fasta=list(fasta_THB_C2), ionic=0.15, path='THB_C2' + outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['hnRNPA1'] = dict(temp=300,expRg=3.12,expRgErr=0.02,pH=7.0,fasta=list(fasta_hnRNPA1),ionic=0.15,path='hnRNPA1'+outdir)  # Tórur's thesis, ph is not checked yet
    proteins.loc['tau35'] = dict(temp=293.2,expRg=4.64,expRgErr=0.1,pH=7.4,fasta=list(fasta_tau35),ionic=0.15,path='tau35'+outdir)  # checked, 6/12/2022
    proteins.loc['CAHSD'] = dict(temp=293,expRg=4.84,expRgErr=0.2,pH=7.0,fasta=list(fasta_CAHSD),ionic=0.07,path='CAHSD'+outdir)
    proteins.loc['GHRICD'] = dict(temp=298,expRg=6.0,expRgErr=0.5,pH=7.3,fasta=list(fasta_GHRICD),ionic=0.35,path='GHRICD'+outdir)
    proteins.loc['p532070'] = dict(eps_factor=0.2,temp=277,expRg=2.39,expRgErr=0.05,pH=7,fasta=list(fasta_p532070),ionic=0.1,path='p532070'+outdir)
    proteins.loc['Ash1'] = dict(temp=293,expRg=2.9,expRgErr=0.05,pH=7.5,fasta=list(fasta_Ash1),ionic=0.15,path='Ash1'+outdir)
    proteins.loc['CTD2'] = dict(temp=293,expRg=2.614,expRgErr=0.05,pH=7.5,fasta=list(fasta_CTD2),ionic=0.12,path='CTD2'+outdir)
    proteins.loc['ColNT'] = dict(temp=277,expRg=2.83,expRgErr=0.1,pH=7.6,fasta=list(fasta_ColNT),ionic=0.4,path='ColNT'+outdir)
    proteins.loc['hNL3cyt'] = dict(temp=293,expRg=3.15,expRgErr=0.2,pH=8.5,fasta=list(fasta_hNL3cyt),ionic=0.3,path='hNL3cyt'+outdir)
    proteins.loc['SH4UD'] = dict(temp=293,expRg=2.71,expRgErr=0.1,pH=8.0,fasta=list(fasta_SH4UD),ionic=0.2,path='SH4UD'+outdir)
    proteins.loc['Sic1'] = dict(temp=293,expRg=3.0,expRgErr=0.4,pH=7.5,fasta=list(fasta_Sic1),ionic=0.2,path='Sic1'+outdir)
    proteins.loc['FhuA'] = dict(temp=298,expRg=3.34,expRgErr=0.1,pH=7.5,fasta=list(fasta_FhuA),ionic=0.15,path='FhuA'+outdir)
    proteins.loc['K10'] = dict(temp=288,expRg=4.0,expRgErr=0.1,pH=7.4,fasta=list(fasta_K10),ionic=0.15,path='K10'+outdir)
    proteins.loc['K27'] = dict(temp=288,expRg=3.7,expRgErr=0.2,pH=7.4,fasta=list(fasta_K27),ionic=0.15,path='K27'+outdir)
    proteins.loc['K25'] = dict(temp=288,expRg=4.1,expRgErr=0.2,pH=7.4,fasta=list(fasta_K25),ionic=0.15,path='K25'+outdir)
    proteins.loc['K32'] = dict(temp=288,expRg=4.2,expRgErr=0.3,pH=7.4,fasta=list(fasta_K32),ionic=0.15,path='K32'+outdir)
    proteins.loc['K23'] = dict(temp=288,expRg=4.9,expRgErr=0.2,pH=7.4,fasta=list(fasta_K23),ionic=0.15,path='K23'+outdir)
    proteins.loc['K44'] = dict(temp=288,expRg=5.2,expRgErr=0.2,pH=7.4,fasta=list(fasta_K44),ionic=0.15,path='K44'+outdir)
    proteins.loc['A1'] = dict(temp=298,expRg=2.76,expRgErr=0.02,pH=7.0,fasta=list(fasta_A1),ionic=0.15,path='A1'+outdir)
    proteins.loc['M12FP12Y'] = dict(temp=298,expRg=2.60,expRgErr=0.02,pH=7.0,fasta=list(fasta_M12FP12Y),ionic=0.15,path='M12FP12Y'+outdir)
    proteins.loc['P7FM7Y'] = dict(temp=298,expRg=2.72,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7FM7Y),ionic=0.15,path='P7FM7Y'+outdir)
    proteins.loc['M9FP6Y'] = dict(temp=298,expRg=2.66,expRgErr=0.01,pH=7.0,fasta=list(fasta_M9FP6Y),ionic=0.15,path='M9FP6Y'+outdir)
    proteins.loc['M8FP4Y'] = dict(temp=298,expRg=2.71,expRgErr=0.01,pH=7.0,fasta=list(fasta_M8FP4Y),ionic=0.15,path='M8FP4Y'+outdir)
    proteins.loc['M9FP3Y'] = dict(temp=298,expRg=2.68,expRgErr=0.01,pH=7.0,fasta=list(fasta_M9FP3Y),ionic=0.15,path='M9FP3Y'+outdir)
    proteins.loc['M10R'] = dict(temp=298,expRg=2.67,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10R),ionic=0.15,path='M10R'+outdir)
    proteins.loc['M6R'] = dict(temp=298,expRg=2.57,expRgErr=0.01,pH=7.0,fasta=list(fasta_M6R),ionic=0.15,path='M6R'+outdir)
    proteins.loc['P2R'] = dict(temp=298,expRg=2.62,expRgErr=0.02,pH=7.0,fasta=list(fasta_P2R),ionic=0.15,path='P2R'+outdir)
    proteins.loc['P7R'] = dict(temp=298,expRg=2.71,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7R),ionic=0.15,path='P7R'+outdir)
    proteins.loc['M3RP3K'] = dict(temp=298,expRg=2.63,expRgErr=0.02,pH=7.0,fasta=list(fasta_M3RP3K),ionic=0.15,path='M3RP3K'+outdir)
    proteins.loc['M6RP6K'] = dict(temp=298,expRg=2.79,expRgErr=0.01,pH=7.0,fasta=list(fasta_M6RP6K),ionic=0.15,path='M6RP6K'+outdir)
    proteins.loc['M10RP10K'] = dict(temp=298,expRg=2.85,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10RP10K),ionic=0.15,path='M10RP10K'+outdir)
    proteins.loc['M4D'] = dict(temp=298,expRg=2.64,expRgErr=0.01,pH=7.0,fasta=list(fasta_M4D),ionic=0.15,path='M4D'+outdir)
    proteins.loc['P4D'] = dict(temp=298,expRg=2.72,expRgErr=0.03,pH=7.0,fasta=list(fasta_P4D),ionic=0.15,path='P4D'+outdir)
    proteins.loc['P8D'] = dict(temp=298,expRg=2.69,expRgErr=0.01,pH=7.0,fasta=list(fasta_P8D),ionic=0.15,path='P8D'+outdir)
    proteins.loc['P12D'] = dict(temp=298,expRg=2.80,expRgErr=0.01,pH=7.0,fasta=list(fasta_P12D),ionic=0.15,path='P12D'+outdir)
    proteins.loc['P12E'] = dict(temp=298,expRg=2.85,expRgErr=0.01,pH=7.0,fasta=list(fasta_P12E),ionic=0.15,path='P12E'+outdir)
    proteins.loc['P7KP12D'] = dict(temp=298,expRg=2.92,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7KP12D),ionic=0.15,path='P7KP12D'+outdir)
    proteins.loc['P7KP12Db'] = dict(temp=298,expRg=2.56,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7KP12Db),ionic=0.15,path='P7KP12Db'+outdir)
    proteins.loc['M12FP12YM10R'] = dict(temp=298,expRg=2.61,expRgErr=0.02,pH=7.0,fasta=list(fasta_M12FP12YM10R),ionic=0.15,path='M12FP12YM10R'+outdir)
    proteins.loc['M10FP7RP12D'] = dict(temp=298,expRg=2.86,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10FP7RP12D),ionic=0.15,path='M10FP7RP12D'+outdir)
    proteins.loc['Hst5'] = dict(temp=293,expRg=1.38,expRgErr=0.05,pH=7.5,fasta=list(fasta_Hst5),ionic=0.15,path='Hst5'+outdir)
    proteins.loc['Hst52'] = dict(temp=298,expRg=1.87,expRgErr=0.05,pH=7.0,fasta=list(fasta_Hst52),ionic=0.15,path='Hst52'+outdir)
    proteins.loc['aSyn140'] = dict(temp=293,expRg=3.55,expRgErr=0.1,pH=7.4,fasta=list(fasta_aSyn140),ionic=0.2,path='aSyn140'+outdir)
    proteins.loc['ACTR'] = dict(temp=278,expRg=2.63,expRgErr=0.1,pH=7.4,fasta=list(fasta_ACTR),ionic=0.2,path='ACTR'+outdir)
    proteins.loc['RNaseA'] = dict(temp=298,expRg=3.36,expRgErr=0.1,pH=7.5,fasta=list(fasta_RNaseA),ionic=0.15,path='RNaseA'+outdir)
    proteins.loc['p15PAF'] = dict(temp=298,expRg=2.81,expRgErr=0.1,pH=7.0,fasta=list(fasta_p15PAF),ionic=0.15,path='p15PAF'+outdir)
    proteins.loc['CoRNID'] = dict(temp=293,expRg=4.7,expRgErr=0.2,pH=7.5,fasta=list(fasta_CoRNID),ionic=0.2,path='CoRNID'+outdir)
    proteins.loc['PNt'] = dict(temp=298,expRg=5.11,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNt),ionic=0.15,path='PNt'+outdir)
    proteins.loc['PNtS1'] = dict(temp=298,expRg=4.92,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS1),ionic=0.15,path='PNtS1'+outdir)
    proteins.loc['PNtS4'] = dict(temp=298,expRg=5.34,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS4),ionic=0.15,path='PNtS4'+outdir)
    proteins.loc['PNtS5'] = dict(temp=298,expRg=4.87,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS5),ionic=0.15,path='PNtS5'+outdir)
    proteins.loc['PNtS6'] = dict(temp=298,expRg=5.26,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS6),ionic=0.15,path='PNtS6'+outdir)
    return proteins
