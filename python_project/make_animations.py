
import os
import torch
from imagining_alghoritm import animate_panel, animate_panel_sidebyside
from GCN_train import build_and_save_ensemble, EnsembleGCN

features = True  
output_dir = r"C:\Users\Maria\Documents\Honours Programme\Networks\GAN\Cross_Validation_Results\2026-07-01-15-03-02"
ensemble_model_path = os.path.join(output_dir, "ensemble_model.pt")
build_and_save_ensemble(output_dir, ensemble_model_path)

ensemble_model = torch.load(ensemble_model_path, weights_only=False)

# animate_panel_sidebyside(panel_number=103, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_103_sidebyside")
# animate_panel_sidebyside(panel_number=104, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_104_sidebyside")
# animate_panel_sidebyside(panel_number=105, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_105_sidebyside")
# animate_panel_sidebyside(panel_number=109, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_109_sidebyside")
animate_panel_sidebyside(panel_number=123, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_123_sidebyside")
