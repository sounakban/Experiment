

# Single layer with splits with individual activations
## Usage ##
# output = Lambda(ParaNet_layer, arguments={'layer_splits':16, 'nodes_per_split':8})(output)
def ParaNet_layer1(input, layer_splits, nodes_per_split):
    from tensorflow import split, concat
    from keras.layers import Dense, Activation

    para_inputs = split(input, num_or_size_splits=layer_splits, axis=1)

    para_outputs = []
    for layer in range(layer_splits):
        para_outputs.append(Dense(nodes_per_split, activation='relu')(para_inputs[layer]))

    return concat(para_outputs, axis=1)


# Perform matrix operations with one node for each featureset then activate after merging layers
## Usage ##
# output = Lambda(ParaNet_layer2, arguments={'layer_splits':100})(input)
def ParaNet_layer2(input, layer_splits):
    nodes_per_split = 1
    from tensorflow import split, concat
    from keras.layers import Dense, Activation

    para_inputs = split(input, num_or_size_splits=layer_splits, axis=1)

    para_outputs = []
    for layer in range(layer_splits):
        para_outputs.append(Dense(nodes_per_split)(para_inputs[layer]))
    output = concat(para_outputs, axis=1)
    output = Activation('sigmoid')(output)

    return output
