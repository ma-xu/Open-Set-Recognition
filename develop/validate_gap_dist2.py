import torch

DICT = torch.load("/Users/melody/Downloads/gap_withdata.pkl", map_location=torch.device('cpu'))
print(DICT[0]['channel_std'])
print((DICT[0]['channel_std']).shape)
bins=200


hists = []
keys = []
for k in torch.arange(0,7):
    k = k.item()
    selected_channels = torch.empty(3, dtype=int).random_((DICT[0]['channel_std']).shape[0])
    for c in selected_channels:
        c = c.item()
        key = f"Class{k}_Channel{c}"
        keys.append(key)
        data = DICT[k]['gaps'][:,c]
        minValue = data.min()
        maxValue = data.max()
        hist = torch.histc(data, bins=bins, min=minValue, max=maxValue)
        hists.append(hist)

outF = open(f"distribution_gap.txt", "w")
strings=""
for i in range(len(hists)):
    strings += keys[i]
    strings += "\t"
outF.write(strings)
outF.write("\n")

for j in range(bins):
    strings=""
    for i in range(len(hists)):

        strings += str(int((hists[i][j]).item()))
        strings += "\t"
    outF.write(strings)
    outF.write("\n")
outF.close()

# for k in torch.arange(0,7):
#     k = k.item()
#     print(k)
#     selected_channels = torch.empty(3,dtype=int).random_((DICT[0]['channel_std']).shape[0])
#     for c in selected_channels:
#         c = c.item()
#         string = f'Class_{k}_Channel_{c}\n'
#         outF.write(string)
#         gaps = DICT[k]['gaps']
#         string=''
#         for i in range(gaps.shape[0]):
#             string += str((gaps[i][c]).item())[0:6]
#             string += "\t"
#         outF.write(string)
#         outF.write("\n")
# outF.close()





