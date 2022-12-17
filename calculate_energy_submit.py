import subprocess
cwd = "/groups/sbinlab/fancao/IDPs_multi"
dataset = "test"
cycle = 0
from utils import *
# allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")
multidomain_names = ['GS48']
# multidomain_names = ["THB_C2"]


for name in multidomain_names:
    with open(f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_calE.sh", 'w') as submit:
        submit.write(submission_4.render(cwd=cwd, dataset=dataset, name=name, cycle=f'{cycle}'))
    proc = subprocess.run(['sbatch', f"{cwd}/{dataset}/{name}/{cycle}/{name}_{cycle}_calE.sh"], capture_output=True)
    print(proc)



