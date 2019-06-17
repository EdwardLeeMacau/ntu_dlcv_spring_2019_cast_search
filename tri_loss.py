import torch
import torch.nn as nn

criterion = nn.TripletMarginLoss(margin=2)

def triplet_loss(inputs, labels, cast_num):
    bs = inputs.size()[0]-1
    candid_num = bs-cast_num
    x_a = inputs[cast_num+1:]
    x_p = inputs[:cast_num]
    loss = 0.0
    for i in range(candid_num):
        
        cand_label = labels[i+1+cast_num]
        num_p = labels.index(cand_label)
        num_n = torch.randint(0, cast_num, (1,))
        if cand_label==labels[num_n]: 
            #if get same label, pick other as x_n
            num_n = cast_num
            
        x_n = inputs[num_n]
#        print('tri_loss a = %d, p = %d, n = %d' 
#              % (cand_label, labels[num_p], labels[num_n]))
        loss += criterion(x_a[i],x_p[num_p],x_n)
    loss /= candid_num
        
    return loss
