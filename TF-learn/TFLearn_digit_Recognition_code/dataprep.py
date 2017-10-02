import tflearn.datasets.mnist as mnist



# load minst data
# Retrieve the training and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)