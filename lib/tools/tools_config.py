from pathlib import Path
import json

class Parameters(enumerate):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("tools_settings.json")
    )
    attrs = {
        "FRAME_INTERVAL": None,
        "OT_EXTEND_RATIO" : None,
        "WIDTH_MIN": None,
        "HEIGHT_MIN": None,
    }
    FRAME_INTERVAL: int
    OT_EXTEND_RATIO : float
    WIDTH_MIN: int
    HEIGHT_MIN: int
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
                    
            if key.replace(".", "").isnumeric():
                key = int(key) if attr != "OT_EXTEND_RATIO" else float(key)
                setattr(cls, attr, key)
                cls.attrs[attr] = key
                break
            else:
                print("You should input digits.\n")


Parameters.initiate()
if __name__ == "__main__":
    for k in Parameters.attrs.keys():
        print(k," | ", getattr(Parameters, k))