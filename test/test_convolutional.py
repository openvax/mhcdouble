from mhc2.convolutional import ConvolutionalPredictor

def test_convolutional_predictor():
    conv = ConvolutionalPredictor(max_training_epochs=5)
    peptides = []
    labels = []
    # generate a dataset where YSIINFEKL is a 9mer binding core
    # and all negative examples are 9+ stretches of leucine
    n_pos = 0
    for i in range(3):
        for j in range(3):
            peptides.append("Q" * i + "YSIINFEKL" + "Q" * j)
            labels.append(True)
            n_pos += 1
    for i in range(10):
        peptides.append("L" * (i + 9))
        labels.append(False)
    Y = conv.fit_predict(peptides, labels)
    assert Y[:n_pos].mean() > Y[n_pos:].mean(), Y