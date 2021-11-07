import torch.nn as nn
import torch.nn.functional as F
import torch

class DistanceLoss(nn.Module):

    def __init__(self,output_embedding,margin = 1):
        super(DistanceLoss, self).__init__()
        self.output_embedding = output_embedding
        self.margin = margin


    def forward(self,WO,label):
        C = self.output_embedding.weight.data.shape[0] #the number of classes
        WO_norm = F.normalize(WO,dim=-1)
        WO_norm_tile = WO_norm.unsqueeze(1).repeat(1,C-1,1)
        label_dist = torch.norm(WO_norm-self.output_embedding(label),p=2,dim=1)
        all_y = []
        for i in range(label.shape[0]):
            all_y.append(list(range(C)))
            all_y[i].remove(label[i])
        all_y = torch.LongTensor(all_y)
        if torch.cuda.is_available():
            try:
                all_y = all_y.cuda()
            except:
                pass
        all_dist = torch.norm(WO_norm_tile-self.output_embedding(all_y),p=2,dim=-1)
        y_minus_dist = torch.min(all_dist,dim=-1)[0] #this is max in the paper,but i think it's more suitable to use min
        loss = torch.mean(self.margin+label_dist-y_minus_dist)
        return loss