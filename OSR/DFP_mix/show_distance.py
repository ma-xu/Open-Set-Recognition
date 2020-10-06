import torch
file_path="/Users/melody/Downloads/distance_bak.pkl"

results = torch.load(file_path,map_location=torch.device('cpu'))
for k, v in results.items():
    distance_k = v["hist"]
    print(distance_k)
    print(v["max"])




