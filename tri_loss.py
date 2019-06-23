import torch
import torch.nn as nn

criterion = nn.TripletMarginLoss(margin=0.25)

def triplet_loss(inputs, labels: list, cast_num: int, criterion=criterion):
    """
      Self define triplet loss function.

      Params:
      - inputs: Tensor  _(num_cast + num_cand) x 2048
      - labels: List    _(num_cast + num_cand) x 1
      - cast_num: int   _ number of casts

      Description:
      - x_p: The casts' images
      - x_a: The candidates' images
      - x_n: The candidates' images

      Return:
      - loss
    """

    batchsize = inputs.shape[0]
    candidate_num = batchsize - cast_num

    inputs = inputs.reshape(batchsize, -1)

    x_a, x_p = inputs[cast_num:], inputs[:cast_num]

    labels = torch.LongTensor(labels)

    index_a = labels[cast_num:]
    index_p = labels[cast_num:]
    index_n = torch.randint(0, cast_num, size=(candidate_num, ))

    # If index_n is equal to index_a, random again.
    while (index_n == index_p).nonzero().shape[0] > 0:
        n = (index_n == index_p).nonzero().numel()
        index_n[index_n == index_p] = torch.randint(0, cast_num, size=(n, ))

    # print('A')
    # print(index_a)
    # print('N')
    # print(index_n)
    # print('P')
    # print(index_p)

    # Make the P/N Pairs with index_p and index_n
    # print("index_n :", index_n)
    # print("index_p :", index_p)
    # print("x_p.shape ;", x_p.shape)
    # print()
    x_p, x_n = x_p[index_p], x_p[index_n]

    # print(x_a[0])
    # print(x_p[0])
    # print(x_n[0])
    # print(x_a.shape)
    # print(x_p.shape)
    # print(x_n.shape)

    loss = criterion(x_a, x_p, x_n)

    return loss
