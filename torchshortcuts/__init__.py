import torch


main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
