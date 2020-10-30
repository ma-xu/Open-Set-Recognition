import torch

DICT = torch.load("/Users/melody/Downloads/gap.pkl", map_location=torch.device('cpu'))
print(DICT[0]['channel_std'])
print((DICT[0]['channel_std']).shape)

outF = open("std.txt", "w")
for i in torch.arange(0,(DICT[0]['channel_std']).shape[0]):
    i=i.item()
    string = ''
    for j in torch.arange(0,7):
        j = j.item()
        string += str((DICT[j]['channel_std'][i]).item())
        string +='\t'
    # write line to output file
    outF.write(string)
    outF.write("\n")
outF.close()



outF = open("mean.txt", "w")
for i in torch.arange(0,(DICT[0]['channel_mean']).shape[0]):
    i=i.item()
    string = ''
    for j in torch.arange(0,7):
        j = j.item()
        string += str((DICT[j]['channel_mean'][i]).item())
        string +='\t'
    # write line to output file
    outF.write(string)
    outF.write("\n")
outF.close()
