from PDBedit import PDBedit
from simulate import *
from analyse import *

def force_constants():
    pae_inv = load_pae("/groups/sbinlab/fancao/IDPs_multi/PNt_pae35_368.json")
    print(pae_inv[-1])
    seq_len = 334
    pae_inv = pae_inv[34:34+seq_len][:,34:34+seq_len]

    # pdbedit = PDBedit()
    # plddt, fasta = list(pdbedit.readPDB_singleChain("/groups/sbinlab/fancao/IDPs_multi/AF-Q546U4-F1-model_v4.pdb").plddt)[34:34+seq_len]

if __name__ == '__main__':
    pass
    # force_constants()





