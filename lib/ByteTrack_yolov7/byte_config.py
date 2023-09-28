from pathlib import Path
import json

class Parameters(enumerate):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("ByteTrack_settings.json")
    )
    OT_TYPE_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("OT-type.json")
    )
    attrs = {
        "output_pth": None,
        "temp_pth" : None,
        "model_pth": None,
        "conf_thresh": None,
        "iou_thresh" : None,
        "track_thresh": None,
        "track_buffer": None,
        "match_thresh": None,
        "min_box_area": None,
        "imgsz": None,
        "augment": None,
    }
    CLS2ID : dict
    ID2CLS : dict
    output_pth : str
    temp_pth : str
    model_pth :str
    config_thresh : float
    iou_thresh : float
    track_thresh : float
    track_buffer : int 
    match_thresh : float
    min_box_area : int
    imgsz : int
    augment : bool 

    @classmethod
    def initiate(cls) -> None:
        UPDATE = False    
        if Path(cls.OT_TYPE_FILE).exists():
            with open(str(cls.OT_TYPE_FILE), 'r') as f:
                cls.CLS2ID:dict = json.load(f)
            cls.ID2CLS = { v:k for k,v in cls.CLS2ID.items()}
        else:
            print("Can't find out the OT-type.json")
            exit()
        if Path(cls.SETTING_FILE).exists():
            with open(cls.SETTING_FILE, "r") as f:
                settings = json.load(f)
            for attr, v in settings.items():
                setattr(cls, attr, v)
                cls.attrs[attr] = v
                
            for attr in cls.attrs.keys():
                if not hasattr(cls, attr):
                    result = cls.__input_attr(attr)
                    setattr(cls, attr, result)
                    cls.attrs[attr] = result
                    UPDATE = True
        else:
            for attr in cls.attrs.keys():
                result = cls.__input_attr(attr)
                setattr(cls, attr, result)
                cls.attrs[attr] = result
                UPDATE =  True
        if UPDATE:
            with open(cls.SETTING_FILE, "w") as f:
                json.dump(cls.attrs, f, indent=4) 
            print(f"Configure is saved to {cls.SETTING_FILE}")
            exit()

    @classmethod
    def __input_attr(cls, attr):
        while True:
            key = input(f"\rInput {attr} value: ")
            if attr == "output_pth" or attr == "model_pth" or attr == "temp_pth":
                split_list = key.split('/')
                abs_file_pth = Path(__file__).resolve().parent
                if len(split_list) == 1:
                    key = abs_file_pth / Path(split_list[0])
                else:
                    for i, sp in enumerate(split_list):
                        if sp == ".":
                            pass
                        elif sp == "..":
                            abs_file_pth = abs_file_pth.parent
                        else:
                            abs_file_pth = abs_file_pth / Path(sp)
                result = str(abs_file_pth)
                break
            elif attr == "augment":
                key = True if key == "True" else False if key == "False" else None
                if key == None:
                    print("augment should be True or False")
                    continue
                result = key 
                break
                    
            if key.replace(".", "").isnumeric():
                key = (
                    int(key)
                    if attr == "min_box_area"
                    or attr == "track_buffer"
                    or attr == "imgsz"
                    else float(key)
                )
                result = key
                break
            else:
                print("You should input digits.\n")
        return result

Parameters.initiate()