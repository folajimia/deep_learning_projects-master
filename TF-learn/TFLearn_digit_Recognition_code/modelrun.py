import tflearnnetwork
import dataprep
import numpy as np




def tester_accuracy():
    # Compare the labels that our model predicts with the actual labels
    # Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
    predictions = np.array(model.predict(dataprep.testX)).argmax(axis=1)

    # Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
    actual = dataprep.testY.argmax(axis=1)
    test_accuracy = np.mean(predictions == actual, axis=0)

    # Print out the result
    print("Test accuracy: ", test_accuracy)


# Build the model
model = tflearnnetwork.build_model()

# Training
model.fit(dataprep.trainX, dataprep.trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=20)

tester_accuracy()



