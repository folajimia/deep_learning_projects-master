
import sys
import dataprep
import neuralnetwork
import mse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Set the hyperparameters here ###
epochs = 850
learning_rate = 0.2
hidden_nodes = 25
output_nodes = 1

N_i = dataprep.train_features.shape[1]
network = neuralnetwork.NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(dataprep.train_features.index, size=128)
    for record, target in zip(dataprep.train_features.ix[batch].values,
                              dataprep.train_targets.ix[batch]['cnt']):
        network.train(record, target)

    # Printing out the training progress
    train_loss = mse.MSE(network.run(dataprep.train_features), dataprep.train_targets['cnt'].values)
    val_loss = mse.MSE(network.run(dataprep.val_features), dataprep.val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)



#check out Predictions

fig, ax = plt.subplots(figsize=(8,4))

mean, std = dataprep.scaled_features['cnt']
predictions = network.run(dataprep.test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((dataprep.test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(dataprep.rides.ix[dataprep.test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)