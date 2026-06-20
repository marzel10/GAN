
import torch_geometric
import numpy as np
import torch

from python_project.graph_dataset import Panel_GraphDataset


def prepare_DI(panels, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = [Panel_GraphDataset(root='graph_data', panel_number=panel, freq=3, big_latent=True) for panel in panels]
    model.eval()
    M = len(panels)
    DI = []
    DI_single = []
    for dataset in datasets:
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out, HI = model(data.x, data.edge_index, data.batch, data.edge_weight)
                DI_single.append(HI.item())
        DI.append(DI_single)

    return DI

def monotonicity_criterion(DI):
    # Check if DI is monotonically increasing across states for each panel
    monotonicity=0
    
    for panel_DI in DI:
        panel_monotonicity  = 0
        for i in range(1, len(panel_DI)):
            point_mono = np.sign(panel_DI[i] - panel_DI[i - 1])/(len(panel_DI)-1)
            panel_monotonicity += point_mono

        panel_monotonicity = np.abs(panel_monotonicity)
        monotonicity += panel_monotonicity
        
    return monotonicity / len(DI)


def trendability_criterion(DI):

    return trendability 

def prognosability_criterion(DI):

    return prognosability

def fitness_criterion(panels, model, a=1, b=1, c=1):
    DI = prepare_DI(panels, model)
    monotonicity = monotonicity_criterion(DI)
    trendability = trendability_criterion(DI)
    prognosability = prognosability_criterion(DI)

    # Combine criteria into a single fitness score (weights can be adjusted)
    fitness_score = a * monotonicity + b * trendability + c * prognosability
    return fitness_score