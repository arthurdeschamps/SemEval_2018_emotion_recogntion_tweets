from data.dataset_loader import DatasetLoader


def show_train_labels_stats():
    _, train_labels = DatasetLoader.load_training_set()
    show_labels_stats(train_labels)


def show_test_labels_stats():
    _, test_labels = DatasetLoader.load_testing_set()
    show_labels_stats(test_labels)


def show_dev_labels_stats():
    _, dev_labels = DatasetLoader.load_development_set()
    show_labels_stats(dev_labels)


def show_labels_stats(labels):
    stats = {}
    for label in labels:
        k = str(sum(label))
        if k in stats:
            stats[k] += 1
        else:
            stats[k] = 1
    for nb_positive_emotions, nb_occurrences in stats.items():
        print(f"# of positive emotions: {nb_positive_emotions} - # of data samples: {nb_occurrences}")