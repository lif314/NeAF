#!/bin/bash

# id=225
# seq=001
# mic=1  # 1 OR 2


ids=(225 234 238  245  248 253 363 374)
seqs=(001)
mic=1

for id in "${ids[@]}"
do
    for seq_id in "${seqs[@]}"
    do
        bash run_vctk.sh "$id" "$seq_id" "$mic"
    done
done