import torch 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from pathlib import Path
import json
with open(Path(__file__).resolve().parent.parent / Path('cfg') / Path("CarType.json")) as f:
    CLS2ID = json.load(f)
    ID2CLS = {v:k for k,v in CLS2ID.items()}
