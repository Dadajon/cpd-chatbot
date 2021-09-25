import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()
    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        train = read('results/train_loss.txt')
        test = read('results/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'g', label='validation')
        plt.legend(loc='lower left')


    elif mode == 'bleu':
        bleu = read('results/blue.txt')
        plt.plot(bleu, 'b', label='blue score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig('results/graph.png')


def main():
    draw(mode='loss')
    draw(mode='bleu')


if __name__ == '__main__':
    main()