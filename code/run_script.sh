KERAS_BACKEND=tensorflow python train.py \
--emb /Users/tian/Documents/aspect-extraction/aspect-extraction/dataset/GoogleNews-vectors-negative300.bin \
--embdim 300 \
--domain restaurant \
--epochs 30 \
--batch-size 50 \
-o output