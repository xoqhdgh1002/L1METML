from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import slice
from tensorflow.keras import initializers
import qkeras
from qkeras.qlayers import QDense, QActivation
import numpy as np


def dense_embedding(n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    with_bias=True,
                    t_mode=0,
                    units=[64, 32, 16]):
    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation(activation=activation)(x)

    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model


def dense_embedding_quantized(n_features=6,
                              n_features_cat=2,
                              number_of_pupcandis=100,
                              embedding_input_dim={0: 13, 1: 3},
                              emb_out_dim=2,
                              with_bias=True,
                              t_mode=0,
                              logit_total_bits=7,
                              logit_int_bits=2,
                              activation_total_bits=7,
                              logit_quantizer='quantized_bits',
                              activation_quantizer='quantized_relu',
                              activation_int_bits=2,
                              alpha=1,
                              use_stochastic_rounding=False,
                              units=[64, 32, 16]):
    n_dense_layers = len(units)

    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(logit_total_bits, logit_int_bits, alpha=alpha, use_stochastic_rounding=use_stochastic_rounding)
    activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(activation_total_bits, activation_int_bits)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = QDense(units[i_dense], kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = QActivation(activation=activation_quantizer)(x)

    if t_mode == 0:
        x = qkeras.qpooling.QGlobalAveragePooling1D(name='pool', quantizer=logit_quantizer)(x)
        # pool size?
        outputs = QDense(2, name='output', bias_quantizer=logit_quantizer, kernel_quantizer=logit_quantizer, activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = QDense(2, name='met_bias', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = QDense(1, name='met_weight', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model


def graph_embedding(De=64,
                    scale_e=2,
                    scale_n=2,
                    n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8):
    name = 'met'

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    N = number_of_pupcandis
    P = n_features+n_features_cat
    Nr = N*(N-1)  # number of relations (edges)

    x = BatchNormalization()(x)

    # Swap axes of input data (batch,nodes,features) -> (batch,features,nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    # Marshaling function
    ORr = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_1'.format(name))(x)
    ORs = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_2'.format(name))(x)
    B = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
    # Outputis new array = [batch, 2x features, edges]

    # Edges MLP (takes as inputs nodes features and fully conected graph edges)
    # Transpose input matrix permutating columns 1&2
    inp_e = Permute((2, 1), input_shape=B.shape[1:])(B)
    # Output is new array = [batch, edges, 2x features]

    # NN inference: run the NN on each edge (each pair of nodes) and output for each edge has a vector
    # Define the Edges MLP layers
    nhidden_e = int((2 * P)*scale_e)
    h = Dense(nhidden_e)(inp_e)
    h = Activation(activation)(h)
    h = Dense(int(nhidden_e/2))(h)
    h = Activation(activation)(h)
    h = Dense(De)(h)
    out_e = Activation(activation)(h)

    # Transpose output and permutes columns 1&2
    out_e = Permute((2, 1))(out_e)

    # Multiply edges MLP output by receiver nodes matrix Rr
    out_e = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3'.format(name))(out_e)

    # Nodes MLP (takes as inputs node features and embeding from edges MLP)
    inp_n = Concatenate(axis=1)([x, out_e])

    # Transpose input and permutes columns 1&2
    inp_n = Permute((2, 1), input_shape=inp_n.shape[1:])(inp_n)

    # 2nd NN inference
    # Define the Nodes MLP layers
    nhidden_n = int((P + De)*scale_n)  # number of neurons in Nodes MLP hidden layer
    h = Dense(nhidden_n)(inp_n)
    h = Activation(activation)(h)
    h = Dense(int(nhidden_n/2))(h)
    h = Activation(activation)(h)
    w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(h)
    w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
    x = Multiply()([w, pxpy])
    outputs = GlobalAveragePooling1D(name='output')(x)

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model
