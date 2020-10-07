import torch
file_path="/Users/melody/Downloads/distance.pkl"
bins = 100
tail_number=20
p_value=0.01


results = torch.load(file_path,map_location=torch.device('cpu'))
for k, v in results.items():
    cls_dist = v['distances']  # distance list for each class
    cls_dist.sort()  # python sort function do not return anything.
    cls_dist = cls_dist[:-tail_number]  # remove the tail examples.
    index = int(len(cls_dist)*(1-p_value))
    threshold = cls_dist[index]
    # cls_dist = cls_dist / (max(cls_dist))  # normalized to 0-1, we consider min as 0.
    # min_distance = min(cls_dist)
    min_distance = min(cls_dist)
    max_distance = max(cls_dist)
    hist = torch.histc(torch.Tensor(cls_dist), bins=bins, min=min_distance, max=max_distance)
    # print(hist)
    print(f"{min_distance}-{max_distance}")



