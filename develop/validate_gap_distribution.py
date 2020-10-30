import torch

DICT = torch.load("/Users/melody/Downloads/gap_withdata.pkl", map_location=torch.device('cpu'))
print(DICT[0]['channel_std'])
print((DICT[0]['channel_std']).shape)

selected_channel = [11, 23, 54, 112, 444, 222, 777, 1, 45, 67, 670]

for k in torch.arange(0,7):
    k = k.item()
    print(k)
    outF = open(f"distribution_{k}.txt", "w")
    for i in torch.arange(0, (DICT[0]['gaps']).shape[0]):
        i = i.item()
        string = ''
        for j in selected_channel:
            string += str((DICT[k]['gaps'][i][j]).item())
            string += '\t'
        outF.write(string)
        outF.write("\n")
    outF.close()







for i in torch.arange(0,(DICT[0]['gaps']).shape[0]):
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
