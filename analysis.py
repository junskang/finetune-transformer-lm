import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from eval_generated import evaluate

from datasets import _rocstories, _pomo

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))


def pomo(data_dir, pred_path, log_path, meteor_path=None):

    def pomo_result(tag, p, r, f1, bleu, meteor):
        print('ROCStories %s P/R/F1/BLEU/METEOR: %.2f/%.2f/%.2f/%.2f/%.2f)' % (tag, p, r, f1, bleu, meteor))

    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _pomo(os.path.join(data_dir, 'test'))

    # predictions, targets, meteor_path
    test_precision, test_recall, test_f1, test_bleu_score, test_meteor_score = evaluate(preds, labels, meteor_path=meteor_path)
    logs = [json.loads(line) for line in open(log_path)][1:]
    for va_tag in ['valid_f1', 'valid_bleu_score', 'valid_meteor_score']:
        best_validation_index = np.argmax([log[va_tag] for log in logs])

        valid_precision = logs[best_validation_index]['valid_precision']
        valid_recall = logs[best_validation_index]['valid_recall']
        valid_f1 = logs[best_validation_index]['valid_f1']
        valid_bleu_score = logs[best_validation_index]['valid_bleu_score']
        valid_meteor_score = logs[best_validation_index]['valid_meteor_score']

        pomo_result("Valid(%s:%d)"%(va_tag, best_validation_index), valid_precision, valid_recall, valid_f1, valid_bleu_score, valid_meteor_score)

    pomo_result("Test", test_precision, test_recall, test_f1, test_bleu_score, test_meteor_score)


