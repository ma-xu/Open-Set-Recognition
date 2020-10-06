import torch
file_path="/Users/melody/Downloads/distance.pkl"
bins = 100
tail_number=50


results = torch.load(file_path,map_location=torch.device('cpu'))
for k, v in results.items():
    cls_dist = v['distances']  # distance list for each class
    cls_dist.sort()  # python sort function do not return anything.
    cls_dist = cls_dist[:-tail_number]  # remove the tail examples.
    # cls_dist = cls_dist / (max(cls_dist))  # normalized to 0-1, we consider min as 0.
    # min_distance = min(cls_dist)
    min_distance = 0
    max_distance = max(cls_dist)
    hist = torch.histc(torch.Tensor(cls_dist), bins=bins, min=min_distance, max=max_distance)
    print(hist)



