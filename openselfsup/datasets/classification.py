import torch
import torch.nn.functional as F
from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from sklearn.metrics import classification_report, roc_auc_score, classification_report, confusion_matrix

@DATASETS.register_module
class ClassificationDataset(BaseDataset):
    """Dataset for classification.
    """

    def __init__(self, data_source, pipeline):
        super(ClassificationDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5), AUC=False):
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        print('****************************length:',num)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc
            if logger is not None and logger != 'silent':
                print_log(
                    "{}_top{}: {:.03f}".format(keyword, k, acc),
                    logger=logger)
        if AUC:
            # softmax = nn.Softmax(dim=1)
            print(F.softmax(scores,dim=1))
            auc_score = roc_auc_score(target, F.softmax(scores,dim=1)[:,1]) # y should be 1-D
            if logger is not None and logger != 'silent':
                print_log(
                    "{}_AUC: {:.03f}".format(keyword, auc_score),
                    logger=logger)
        print(target.shape, scores.topk(1, dim=1, largest=True, sorted=True)[1].squeeze(1).shape)
        cls_report = classification_report(target, scores.topk(1, dim=1, largest=True, sorted=True)[1].squeeze(1))
        print(cls_report)
        print(confusion_matrix(target,  scores.topk(1, dim=1, largest=True, sorted=True)[1].squeeze(1)))
            
        return eval_res
