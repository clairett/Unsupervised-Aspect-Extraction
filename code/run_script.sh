KERAS_BACKEND=theano THEANO_FLAGS="device=cpu,floatX=float32" python2.7 train.py \
--vocab-size 1600 \
--emb ../preprocessed_data/mobile/w2v_embedding \
--domain mobile \
--epochs 10 \
--aspect-size 10 \
-o output_dir \

