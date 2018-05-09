import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, recall_score, roc_auc_score

def plot_roc_auc(all_y, all_pred):
    fpr, tpr, _ = roc_curve(all_y, all_pred)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver operating characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall(all_y, all_pred):
    precision, recall, thresholds = precision_recall_curve(all_y, all_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.title('Precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

def plot_f1(all_y, all_pred):
    precision, recall, thresholds = precision_recall_curve(all_y, all_pred)
    f1 = [2*p*r/(p+r) if p != 0 and r != 0 else 0.0 for p, r in zip(precision, recall)][:len(thresholds)]
    plt.step(thresholds, f1, color='b', where='post')
    plt.title('F1')
    plt.xlabel('Thresholds')
    plt.ylabel('F1')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

def choose_threshold_by_f1(all_y, all_pred):
    precision, recall, thresholds = precision_recall_curve(all_y, all_pred)
    f1 = [2*p*r/(p+r) if p != 0 and r != 0 else 0.0 for p, r in zip(precision, recall)][:len(thresholds)]
    return float(thresholds[np.argmax(f1)])
