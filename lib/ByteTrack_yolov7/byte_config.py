from pathlib import Path
import json

class Parameters(enumerate):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("ByteTrack_settings.json")
    )
    attrs = {
        "output_folder": None,
        "temp_folder" : None,
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

    @classmethod
    def initiate(cls) -> None:
        UPDATE = False    
        if Path(cls.SETTING_FILE).exists():
            with open(cls.SETTING_FILE, "r") as f:
                settings = json.load(f)
            for k, v in settings.items():
                cls.attrs[k] = v
                setattr(cls, k, v)

            for k, v in cls.attrs.items():
                if not hasattr(cls, k):
                    cls.__input_attr(k)
                    UPDATE = True
        else:
            for attr in cls.attrs.keys():
                cls.__input_attr(attr)
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
            if attr == "output_folder" or attr == "model_pth" or attr == "temp_folder":
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
                abs_file_pth = str(abs_file_pth)
                setattr(cls, attr, abs_file_pth)
                cls.attrs[attr] = abs_file_pth
                break
            elif attr == "augment":
                key = True if key == "True" else False if key == "False" else None
                if key == None:
                    print("augment should be True or False")
                    continue
                setattr(cls, attr, key)
                cls.attrs[attr] = key
                break
                    
            if key.replace(".", "").isnumeric():
                key = (
                    int(key)
                    if attr == "min_box_area"
                    or attr == "track_buffer"
                    or attr == "imgsz"
                    else float(key)
                )
                setattr(cls, attr, key)
                cls.attrs[attr] = key
                break
            else:
                print("You should input digits.\n")


Parameters.initiate()