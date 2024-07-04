#!/bin/python3
from matplotlib import pyplot
import pickle



metrics_file = open('./metrics/metrics.pkl', 'rb')
history = pickle.load(metrics_file)
metrics_file.close()

pyplot.subplot(211)
# pyplot.title('Loss')
pyplot.plot(history['loss'], label='Train')
pyplot.plot(history['val_loss'], label='Validation')
pyplot.xlabel('', fontsize=16, fontweight='bold')
pyplot.ylabel('Loss', fontsize=16, fontweight='bold')
# pyplot.xticks(x)
x_positions =list(range(0, 10, 1))
x_tick_labels = list(range(1, 11, 1))
pyplot.xticks(x_positions, x_tick_labels, fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.legend(prop={'size':14})

pyplot.subplot(212)
# pyplot.title('Accuracy')
pyplot.plot(history['accuracy'], label='Train')
pyplot.plot(history['val_accuracy'], label='Validation')
pyplot.xlabel('Epoch', fontsize=16, fontweight='bold')
pyplot.ylabel('Accuracy', fontsize=16, fontweight='bold')
pyplot.xticks(x_positions, x_tick_labels, fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.legend(prop={'size':14})
pyplot.savefig('./metrics/metrics.jpg', dpi=600, bbox_inches='tight')
pyplot.ion()
pyplot.pause(1)
pyplot.close()
