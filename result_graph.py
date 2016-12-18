import matplotlib.pyplot as plt
import json


def disp(epochs, a, b):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), a, color='#004358', label='rnn')
    ax.plot(range(epochs), b, color='#ff53d3', label='lstm')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def data_load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    train_loss = data["train_loss"]
    result = [float(loss) for loss in train_loss]
    return result


if __name__ == '__main__':
    epochs = 10

    rnn_train_loss = data_load("./rnn.json")
    lstm_train_loss = data_load("./lstm.json")
    disp(epochs, rnn_train_loss, lstm_train_loss)

