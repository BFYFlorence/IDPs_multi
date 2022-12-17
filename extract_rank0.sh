#!/bin/bash

multidomain_names=(THB_C2 Ubq2 Ubq3 Gal3 TIA1 Ubq4 hnRNPA1 C5_C6_C7 hSUMO_hnRNPA1 GS0 GS8 GS16 GS24 GS32 GS48)
cwd=/groups/sbinlab/fancao/IDPs_multi
for name in ${multidomain_names[*]}; do
cp $cwd/af2pre/$name/ranked_0.pdb $cwd/extract_rank0/${name}_rank0.pdb
done