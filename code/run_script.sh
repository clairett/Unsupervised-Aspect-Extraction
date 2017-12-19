KERAS_BACKEND=theano THEANO_FLAGS="device=cpu,floatX=float32" python2.7 train.py \
--emb ../preprocessed_data/mobile/w2v_embedding \
--domain mobile \
-o output_dir \

