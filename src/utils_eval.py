import numpy as np
import pandas as pd
from sklearn import metrics
from .spot import SPOT

import matplotlib.pyplot as plt
using_best = True
def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r

def get_metrics(label, post_s, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    if using_best:
        f1, precision, recall = get_best_f1(label, score)
    else:
        q = 1e-5
        s = SPOT(q)
        s.fit(post_s, score)
        s.initialize()
        ret = s.run()
        pot_th = np.mean(ret['thresholds'])
        predict = adjust_predicts(score, label, pot_th)
        TP = np.sum(predict * label)
        TN = np.sum((1 - predict) * (1 - label))
        FP = np.sum(predict * (1 - label))
        FN = np.sum((1 - predict) * label)
        precision = TP / (TP + FP + 0.00001)
        recall = TP / (TP + FN + 0.00001)
        f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return auroc, ap, f1, precision, recall


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    Accepted from https://github.com/imperial-qore/TranAD/pot.py
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score

def get_event_metrics(df, label, score):
    """
    use the corresponding threshold of the best f1 of adjusted scores
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_threshold = threshold[np.argmax(f1)]
    label_predict = [s >= best_threshold for s in score]
    label_predict = np.array(label_predict, dtype=int)

    # time is previously used as index when reading data frame, rest index to ordered index here
    df = df.reset_index()
    if 'time' in df.columns:
        df_new = df[['time']].copy()
        df_new['time'] = pd.to_datetime(df_new['time']).dt.ceil('S')
        df_new['label'] = label
        df_new['label_predict'] = label_predict

        label_group = count_group('label', df=df_new, delta='12 hour')
        predict_group = count_group('label_predict', df=df_new, delta='12 hour')
        true_group = count_group('label', 'label_predict', df=df_new, delta='12 hour')

        event_precision = true_group / predict_group
        event_recall = true_group / label_group

    else:
        # @TODO event metrics for data frames that are without time column.
        event_precision = -1
        event_recall = -1

    return event_precision, event_recall

def count_group(*args, df, delta):
    if len(args) == 1:
        df_y = df[df[args[0]] == 1]
    if len(args) == 2:
        df_y = df[(df[args[0]] == 1) & (df[args[1]] == 1)]
    df_y_cur1 = df_y.iloc[:-1, :]
    df_y_cur2 = df_y.iloc[1:, :]
    df_y_cur = [df_y_cur2['time'].iloc[i] - df_y_cur1['time'].iloc[i] for i in range(df_y.shape[0] - 1)]
    num_group = 1
    for i in range(len(df_y_cur)):
        if df_y_cur[i] > pd.Timedelta(delta):
            num_group += 1
    return num_group
