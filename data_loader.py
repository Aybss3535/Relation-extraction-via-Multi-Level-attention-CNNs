import torch
import torch.utils.data as data
import os, random, json, logging
import sklearn.metrics

class MultiAttREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data),
                                                                                            len(self.rel2id)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        e1 = int(item['h']['id'])
        e2 = int(item['t']['id'])
        seq.append(e1)
        seq.append(e2)
        return [self.rel2id[item['relation']]] + seq  # label, seq1, seq2, ...

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:4]
        e1 = torch.LongTensor([list(data[4])]).transpose(0,1)
        e2 = torch.LongTensor([list(data[5])]).transpose(0,1)
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        batch_seqs.append(e1)
        batch_seqs.append(e2)
        return [batch_labels] + batch_seqs

    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        new_pred_result = []
        new_label_result = []
        classes = set()
        for i in range(total):
            if use_name:
                label = self.data[i]['relation']
            else:
                label = self.rel2id[self.data[i]['relation']]
            if label != neg:
                new_pred_result.append(pred_result[i])
                new_label_result.append(label)
            if pred_result[i]==label:
                correct += 1
            classes.add(label)

        acc = float(correct) / float(total)
        classes = list(classes)
        macro_p = sklearn.metrics.precision_score(new_label_result,new_pred_result,labels = classes, average='macro')
        macro_r = sklearn.metrics.recall_score(new_label_result, new_pred_result, labels=classes, average='macro')
        macro_f1 = sklearn.metrics.f1_score(new_label_result, new_pred_result, labels=classes, average='macro')

        result = {'acc': acc, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result


def MultiAttRELoader(path, rel2id, tokenizer, batch_size,
                     shuffle, num_workers=8, collate_fn=MultiAttREDataset.collate_fn, **kwargs):
    dataset = MultiAttREDataset(path=path, rel2id=rel2id, tokenizer=tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

