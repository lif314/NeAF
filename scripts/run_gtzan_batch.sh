#!/bin/bash

# type=blues
# seq_id=00000


types=(blues classical country disco hiphop jazz metal pop reggae rock)
seq_ids=(00000 00033 00099)


for type in "${types[@]}"
do
    for seq_id in "${seq_ids[@]}"
    do
        bash run_gtzan.sh "$type" "$seq_id"
    done
done