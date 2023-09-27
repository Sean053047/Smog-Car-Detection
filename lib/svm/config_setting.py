from pathlib import Path
import json


class Parameters(enumerate):
    SETTING_FILE = (
        Path(__file__).resolve().parent / Path("cfg") / Path("svm_settings.json")
    )
    Hogify_pth :str
    Scalify_pth : str
    Svm_pth : str
    traning_info : dict
    attrs = {
        "Hogify_pth": None,
        "Scalify_pth" : None,
        "Svm_pth": None,
        "training_info": {
            "width" : None,
            "height": None,
            "width_min": None,
            "height_min":None,
            "num_label": None,
            "label" : None,
        }
        
    }

    @classmethod
    def initiate(cls) -> None:
        UPDATE = False    
        if Path(cls.SETTING_FILE).exists():
            with open(cls.SETTING_FILE, "r") as f:
                settings = json.load(f)
            # ? assign known value to cls.attrs
            for attr, v in settings.items():
                if isinstance(v, dict):
                    inter_dict = settings[attr]
                    for k,v  in settings[attr].items():
                        if k == "label":
                            cls.attrs[attr][k] = inter_dict["label"]
                            if len(inter_dict["label"]) < inter_dict["num_label"]:
                                for k2, v2 in inter_dict["label"].items():
                                    cls.attrs[attr][k][k2] = v2 
                            
                        else: 
                            cls.attrs[attr][k] = v
                else:
                    cls.attrs[attr] = v
                setattr(cls, attr, cls.attrs[attr])
            
            # ? check if there is any missing values.
            for attr, v in cls.attrs.items():
                if not hasattr(cls, attr):
                    result = cls.__react(attr)
                    setattr(cls, attr, result)
                    cls.attrs[attr] = result
                    UPDATE = True
                elif isinstance(getattr(cls,attr), dict):
                    
                    for k, v in cls.attrs["training_info"].items():
                        if v is None:
                            cls.attrs["training_info"][k] =cls.__react(k)
                            UPDATE = True
                        elif k =="label":
                            if len(cls.attrs["training_info"]["label"]) < cls.attrs["training_info"]["num_label"]:
                                for i in range(cls.attrs["training_info"]["num_label"]):
                                    value = cls.attrs["training_info"]["label"].get(str(i), None)
                                    cls.attrs["training_info"]["label"][str(i)] = input(f"Input the label of {i}: ") if value is None else value
                                UPDATE = True
                    setattr(cls, attr, cls.attrs[attr])
                    

        else:
            # ? when the setting file doesn't exist, create a new one.
            for attr in cls.attrs.keys():
                result = cls.__react(attr)
                setattr(cls, attr, result)
                cls.attrs[attr] = result  
                UPDATE =  True
        if UPDATE:
            with open(cls.SETTING_FILE, "w") as f:
                json.dump(cls.attrs, f, indent=4) 
            print(f"Configure is saved to {cls.SETTING_FILE}")
            exit()


    @staticmethod
    def __input_attr(attr, hint_txt = None):
        hint_txt = f"Input {attr} value: " if hint_txt is None else hint_txt

        while True:
            key = input(f"\r{hint_txt}")
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
            elif key.isnumeric():
                result= int(key)
                break
            else:
                print("You should input digits.\n")
        return result


    @classmethod
    def __react(cls,attr):
        
        if attr == 'training_info':
            result = dict()
            for tr_attr in cls.attrs[attr].keys():
                if tr_attr != "label":
                    result = cls.__input_attr(tr_attr)
                    result[tr_attr] = result
                else:
                    label_dict = dict()
                    for i in range(result["num_label"]):
                        label_dict[i] = input(f"Input the label of {i}: ")
                    result[tr_attr] = label_dict      
        elif attr == 'label':
            result = dict()
            for i in range(cls.attrs['training_info']["num_label"]):
                result[str(i)] = input(f"Input the label of {i}: ")
        else:
            result = cls.__input_attr(attr)
        return result
Parameters.initiate()

if __name__ == "__main__":
    for k in Parameters.attrs.keys():
        print(getattr(Parameters, k))