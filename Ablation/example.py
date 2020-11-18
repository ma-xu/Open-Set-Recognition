import os
import math
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Test distance shifting')
parser.add_argument('-n', default=500, type=int, help='Generated samplers number')
parser.add_argument('--bins', default=50, type=int, help='Bins for histogram')
parser.add_argument('--dims', nargs='+', default=[10, 100,1000,5000,10000], type=int, help='dim list for test')
parser.add_argument('--scaled',  action='store_true', help='Distance scaled by dim')
args = parser.parse_args()

# saving figures(folder) for current folder.
folder = str(args.n)+'-'+str(args.bins)+'-'+str(args.scaled)
args.save_path = os.path.join(os.getcwd(),folder)
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)


class Distance(torch.nn.Module):
    # Distance should return a tensor with shape [n1, n2]
    def __init__(self, scaled=True):
        super(Distance, self).__init__()
        self.scaled = scaled

    def l1(self, features, centroids):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        _, dim_num = features.shape
        centroids = centroids.unsqueeze(0)
        features = features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 1).squeeze(0)
        if self.scaled:
            distance = distance / math.sqrt(dim_num)
        return distance

    def l2(self, features, centroids):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        _, dim_num = features.shape
        centroids = centroids.unsqueeze(0)
        features = features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 2).squeeze(0)
        if self.scaled:
            distance = distance / math.sqrt(dim_num)
        return distance

    def cosine(self, features, centroids):
        sample_num, dim_num = features.shape
        class_num = centroids.shape[0]
        centroids = centroids.unsqueeze(0).expand(sample_num, class_num, dim_num)
        features = features.unsqueeze(1).expand(sample_num, class_num, dim_num)
        distance = F.cosine_similarity(features, centroids, dim=2)

        # re-range the range [-1,1] tp [0,1]
        # after re-ranging, distance 0 means same, 0.5 means orthogonality/decorrelation, 1 means opposite.
        # referring https://en.wikipedia.org/wiki/Cosine_similarity
        # distance = (1.0 - distance) / 2.0
        return distance

    def dotproduct(self, features, centroids):
        centroids = centroids.permute(1, 0)
        distance = torch.matmul(features, centroids)
        if self.scaled:
            scale = math.sqrt(features.size(-1))
            distance = distance / scale
        return distance


def main():
    Dist = Distance(scaled=args.scaled)
    metrics = ['l1', 'l2', 'cosine', 'dotproduct']
    for dim in args.dims:
        samplers = torch.rand(args.n, dim)
        for metric in metrics:
            print(f"calculating dim:{dim} distance metric:{metric}...")
            distance = getattr(Dist, metric)(samplers, samplers)
            distance = distance.reshape(-1)
            distance = distance.tolist()

            # plot
            plt.hist(distance, bins=args.bins,rwidth=0.8, alpha=0.7)
            plt.xlabel("bins")
            plt.ylabel("hist")
            plt.title(f"Metric: {metric} Dim: {dim}")
            file_name = os.path.join(args.save_path, str(dim)+'_'+str(metric) )
            plt.savefig(file_name, bbox_inches='tight', dpi=150)
            plt.close()


if __name__ == '__main__':
    main()
