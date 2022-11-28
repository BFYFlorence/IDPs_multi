from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import pandas as pd
import MDAnalysis as mda
from Bio import SeqIO, SeqUtils
import yaml

def initProteins(eps_factor=0.2,pH=7.0,ionic=0.15,temp=300,fname=None):
    """ Initialize protein dataframe with default values """
    df = pd.DataFrame(columns=['eps_factor','pH','ionic','temp','fasta'],dtype=object)
    return df

def fasta_from_pdb(pdb):
    """ Generate fasta from pdb entries """
    u = mda.Universe(pdb)
    res3 = "".join(u.residues.resnames)
    fastapdb = SeqUtils.seq1(res3)
    return fastapdb

def addProtein(df,name,seq=None,use_pdb=False,pdb=None,ffasta=None,eps_factor=0.2,pH=7.0,ionic=0.15,temp=300):
    """if use_pdb:
        fasta = fasta_from_pdb(pdb)
    else:
        records = read_fasta(ffasta)
        fasta = records[name].seq"""
    df.loc[name] = dict(pH=pH,ionic=ionic,temp=temp,eps_factor=eps_factor,fasta=seq)
    return df

def modProtein(df,name,**kwargs):
    for key,val in kwargs.items():
        print(key,val)
        if key not in df.columns:
            df[key] = None # initialize property, does not work with lists
        df.loc[name,key] = val
    return df

def delProtein(df,name):
    df = df.drop(name)
    return df

def subset(df,names):
    df2 = df.loc[names]
    return df2

def read_fasta(ffasta):
    records = SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))
    return records

def get_ssdomains(name,fdomains):
    with open(f'{fdomains}','r') as f:
        stream = f.read()
        domainbib = yaml.safe_load(stream)

    domains = domainbib[name]
    print(f'Using domains {domains}')
    ssdomains = []

    for domain in domains:
        ssdomains.append(list(range(domain[0],domain[1])))
    
    return ssdomains

def output_fasta(cwd, path2pdbfolder):
    """Output all sequences unber ${path} to every single fasta file in ${multidomain_fasta}"""
    # https://biopython.org/wiki/SeqRecord
    os.system(f"ls {path2pdbfolder} > {path2pdbfolder}/pdbnames.txt")
    with open(f"{path2pdbfolder}/pdbnames.txt", 'r') as file:
        for line in file.readlines():
            record = line.strip().split(".")
            if record[-1] == "pdb":
                fasta = fasta_from_pdb(f"{path2pdbfolder}/{record[0]}.pdb")
                print(record[0], fasta)
                fasta2save = SeqRecord(Seq(fasta), id=record[0], name=record[0], description=record[0])
                with open(f"{cwd}/multidomain_fasta/{record[0]}.fasta", "w") as output_handle:
                    SeqIO.write(fasta2save, output_handle, "fasta")


if __name__ == '__main__':
    cwd = "/groups/sbinlab/fancao/IDPs_multi"
    print(get_ssdomains("Gal3", f'{cwd}/domains.yaml'))
    # path2pdbfolder = f"{cwd}/pdbfolder"
    # output_fasta(cwd, path2pdbfolder)