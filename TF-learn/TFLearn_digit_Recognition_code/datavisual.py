import dataprep
import matplotlib.pyplot as plt
#% matplotlib inline


# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = dataprep.trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = dataprep.trainX[index].reshape([28, 28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()


# Display the first (index 0) training image
show_digit(0)