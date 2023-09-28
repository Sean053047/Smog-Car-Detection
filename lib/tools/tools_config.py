from pathlib import Path
import json

class Parameters(enumerate):
    SETTING_FILE = str(Path(__file__).resolve().parent / Path("cfg") / Path("tools_settings.json"))

    attrs = {
        "FRAME_INTERVAL": None,
        "OT_EXTEND_RATIO" : None,
        "WIDTH_MIN": None,
        "HEIGHT_MIN": None,
        "LicenseModel_pth" : None,
        "SmokeModel_pth" : None,
    }
    FRAME_INTERVAL: int
    OT_EXTEND_RATIO : float
    WIDTH_MIN: int
    HEIGHT_MIN: int
    @classmethod
    def initiate(cls) -> None:
        UPDATE = False    
        if Path(cls.SETTING_FILE).exists():
            print(cls.SETTING_FILE)
            with open(cls.SETTING_FILE, "r") as f:
                settings = json.load(f)
            for k, v in settings.items():
                cls.attrs[k] = v
                setattr(cls, k, v)

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
            if attr.split('_')[-1] == "pth":
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
            elif key.replace(".", "").isnumeric():
                result = int(key) if attr != "OT_EXTEND_RATIO" else float(key)
                break
            else:
                print("You should input digits.\n")
        return result

Parameters.initiate()
if __name__ == "__main__":
    for k in Parameters.attrs.keys():
        print(k," | ", getattr(Parameters, k))