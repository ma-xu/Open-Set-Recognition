import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    def __init__(self, temperature=1):
        super(DFPLoss, self).__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        sim_classification = net_out["normweight_fea2cen"]  # [n, class_num]; range [-1,1] greater indicates more similar.
        loss_classification = self.ce(sim_classification/self.temperature, targets)

        return {"total": loss_classification}


        # dist_fea2cen = net_out["cosine_fea2cen"]
        # # dist_fea2cen = torch.exp(dist_fea2cen)-1.0
        # # # dist_fea2cen = 0.5 * (dist_fea2cen ** 2)  # 0.5*||d||^2
        # batch_size, num_classes = dist_fea2cen.shape
        # classes = torch.arange(num_classes, device=targets.device).long()
        # labels = targets.unsqueeze(1).expand(batch_size, num_classes)
        # mask = labels.eq(classes.expand(batch_size, num_classes))
        # dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)
        # loss_distance = self.alpha * (dist_within.sum()) / batch_size
        #
        # loss = loss_classification + loss_distance
        # return {
        #     "total": loss,
        #     "classification": loss_classification,
        #     "distance": loss_distance
        # }

class DFPEnergyLoss(nn.Module):
    def __init__(self, mid_known, mid_unknown, temperature=1, alpha=1.0):
        super(DFPEnergyLoss, self).__init__()
        self.mid_known = mid_known
        self.mid_unknown = mid_unknown
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets,net_out_unknown):
        sim_classification = net_out["normweight_fea2cen"]  # [n, class_num];
        loss_classification = self.ce(sim_classification/self.temperature, targets)

        energy_known = net_out["energy"]
        energy_unknown = net_out_unknown["energy"]
        loss_energy_known = (F.relu(self.mid_known-energy_known,inplace=True).pow(2).sum()) / (energy_known.shape[0])
        loss_energy_unknown = (F.relu(energy_unknown-self.mid_unknown, inplace=True).pow(2).sum()) / (energy_unknown.shape[0])
        # print(f"loss_energy_known: {loss_energy_known} | loss_energy_unknown: {loss_energy_unknown}")
        loss_energy = loss_energy_known + loss_energy_unknown
        loss_energy = self.alpha*loss_energy
        total = loss_classification + loss_energy

        return {
            "total": total,
            "loss_classification": loss_classification,
            "loss_energy": loss_energy,
            "loss_energy_known": loss_energy_known,
            "loss_energy_unknown": loss_energy_unknown
        }


def demo():
    n = 3
    c = 5
    sim_fea2cen = torch.rand([n, c])

    label = torch.empty(3, dtype=torch.long).random_(c)
    print(label)
    loss = DFPLoss(1.)
    netout = {
        "dotproduct_fea2cen": sim_fea2cen,
        "cosine_fea2cen": sim_fea2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss)


# demo()
