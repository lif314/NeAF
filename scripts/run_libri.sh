# read_id=19
# chapter_id=198
# seq_id=0000
read_id=$1
chapter_id=$2
seq_id=$3

data_path=data/LibriSpeech/train100/LibriSpeech/train-clean-100/$read_id/$chapter_id/$read_id-$chapter_id-$seq_id.flac
save_dir=logs/libsi/$read_id-$chapter_id-$seq_id

# 263 K
python train.py --arch siren \
    --save_dir $save_dir \
    --exp_name siren \
    --audio_path $data_path

# 254 K
python train.py --arch fourier \
    --save_dir $save_dir \
    --exp_name fourier \
    --audio_path $data_path