import numpy as np
import matplotlib.pyplot as plt
def p(m):
    print(m)
    exit()

def plot_train(train_history, train, validtion):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validtion])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validtion'], loc='upper left')
    plt.show()