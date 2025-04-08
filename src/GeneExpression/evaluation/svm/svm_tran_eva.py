import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix
import pyBigWig
from sklearn.model_selection import GridSearchCV
import os
import inspect


def get_features(predicted_file, train_sample_file, include_chr, extend=25600, window_size=128):
    samples = np.loadtxt(train_sample_file, dtype='str')[:, :4]
    predicted_open = pyBigWig.open(predicted_file, 'r')
    max = predicted_open.header()['maxVal']

    features = []
    labels = []

    for sample in samples:
        if sample[0] in include_chr:
            mid = (int(sample[1]) + int(sample[2])) // 2
            start = mid - extend
            end = mid + extend
            pre = (predicted_open.values(sample[0], start, end, numpy=True).astype('float16')).reshape((-1, window_size))
            pre = np.mean(pre, axis=-1) / max
            label = int(sample[-1])

            features.append(pre)
            labels.append(label)

    return np.asarray(features), np.asarray(labels, dtype=int)


def train_svm(features, labels, kernel='rbf', C=10, gamma=1):
    """train svm model"""
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(features, labels)
    return model


def grid_search(features, labels):
    """grid search"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
    grid_search.fit(features, labels)

    print("best parameter:", grid_search.best_params_)

    return grid_search.best_params_


def svm_train_eva(
        train_sample_file,
        test_sample_file,
        predicted_RNA,
        include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
              'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
):
    features, labels = get_features(predicted_RNA, train_sample_file, include_chr)

    # best_params = grid_search(features, labels)

    svm_model = train_svm(features, labels)

    t_features, t_labels = get_features(predicted_RNA, train_sample_file, include_chr)
    y_pred = svm_model.predict(t_features)

    # roc parameters
    fpr, tpr, thresholds = roc_curve(t_labels, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    precision, recall, _ = precision_recall_curve(t_labels, y_pred)
    prc = average_precision_score(t_labels, y_pred, average='macro', sample_weight=None)
    print(prc)


if __name__ == '__main__':
    # Useage:
    current_file_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    AttentionChromeSamples_path = os.path.abspath(os.path.join(current_file_path, '../../../../genome_regions/GeneExpression/AttentionChromeSamples/GM12878/classification'))
    sample_path = os.path.abspath(os.path.join(current_file_path, '../../../../test_samples/rna'))

    include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

    train_sample_file = '%s/train_samples.bed' % AttentionChromeSamples_path
    test_sample_file = '%s/test_samples.sort.bed' % AttentionChromeSamples_path
    predicted_RNA = '%s/GM12878_pred.bw' % sample_path

    svm_train_eva(
        train_sample_file,
        test_sample_file,
        predicted_RNA
        )