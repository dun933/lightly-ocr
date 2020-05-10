import torch
from torch.autograd import Variable
from MORAN import MORAN

MORAN = MORAN(1,37,256,32,100,bidirectional=True, inputDataType='torch.FloatTensor',CUDA=False)
inputs = Variable(torch.rand(1,1,64,256))
text = Variable(torch.LongTensor([0,0]))
text_rev = text
length = Variable(torch.LongTensor([2]))
output = MORAN(inputs, length, text, text_rev, test=True)
for i in output:
    print(i.size())
