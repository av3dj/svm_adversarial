import matplotlib.pyplot as plt
import json

with open('config.json') as config_file:
    config = json.load(config_file)

def plot_data(data, index):

    plt.figure(index+1, figsize=[12,8])

    plt.suptitle(data['title'])
    plt.subplot(221)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('Clean Loss')
    plt.plot(range(len(data['clean loss'])), data['clean loss'])
    plt.subplot(222)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.title('Clean Accuracy')
    plt.plot(range(len(data['clean accuracy'])), data['clean accuracy'])
    if len(data['robust loss']) > 0 and len(data['robust accuracy']) > 0:
        plt.subplot(223)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.title('Robust Loss')
        plt.plot(range(len(data['robust loss'])), data['robust loss'])
        plt.subplot(224)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.title('Robust Accuracy')
        plt.plot(range(len(data['robust accuracy'])), data['robust accuracy'])

    plt.show()

history_list = []

with open(config['train_history_dir'] + 'normal.json') as f:
    normal_history = json.load(f)
    history_list.append(normal_history)

with open(config['train_history_dir'] + 'normal_mixed.json') as f:
    normal_mixed_history = json.load(f)
    history_list.append(normal_mixed_history)

with open(config['train_history_dir'] + 'adversarial_normal.json') as f:
    adversarial_normal_history = json.load(f)
    history_list.append(adversarial_normal_history)

with open(config['train_history_dir'] + 'adversarial_mixed.json') as f:
    adversarial_mixed_history = json.load(f)
    history_list.append(adversarial_mixed_history)

for idx, history in enumerate(history_list):
    for key in history.keys():
        if key == 'title':
            continue
        arr = history[key]
        arr_new = [float(i) for i in arr]
        history[key] = arr_new


    plot_data(history, idx)

