from pathlib import Path
import json
import inspect
from sys import modules

class BaseAttrs:
    '''This class aims to let you easily load and check the variables in json file.
    Notice that there are two class variables you should be care about...
    Necessary : 
        attrs : list[str]
    Optional : 
        SETTING_FILE : str => (When you define Parameters, it is necessary.)'''
    SETTING_FILE :str = ""
    attrs = list()

    UPDATE : bool
    already_load : set
    file_classes : dict
    
    @classmethod
    def initiate(cls, attrs_dict=None, module_name = __name__) -> None:
        cls.UPDATE = False
        cls.already_load = set()    
        cls.file_classes = {cls_name:cls_obj for cls_name, cls_obj in inspect.getmembers(modules[module_name]) if inspect.isclass(cls_obj)}
        cls.SETTING_FILE = str(cls.SETTING_FILE)
        if cls.SETTING_FILE != "":
            if Path(cls.SETTING_FILE).exists():
                with open(cls.SETTING_FILE, "r") as f:
                    attrs_dict = json.load(f)
            else: 
                attrs_dict = dict()
        elif attrs_dict is None:
            attrs_dict = dict()
        
        
        for attr, v in attrs_dict.items():
            if attr in cls.file_classes:
                new_cls = cls.file_classes[attr]
                setattr(cls, attr, new_cls)
                cls.UPDATE =  new_cls.initiate(v) or cls.UPDATE
            else:
                setattr(cls, attr, v)
            cls.already_load.add(attr)

        diff = [attr for attr in cls.attrs if attr not in cls.already_load]
        if len(diff) != 0 :
            print(f"Start to input {cls.__name__} related parameters...")
        
        for attr in diff:
            if not hasattr(cls, attr):
                if attr in cls.file_classes:
                    new_cls = cls.file_classes[attr]
                    setattr(cls, attr, new_cls)
                    new_cls.initiate()
                    cls.UPDATE = True
                else:
                    result = cls.__input_attr(attr)
                    setattr(cls, attr, result)
                    cls.UPDATE = True
        return cls.UPDATE
   
    @classmethod
    def __input_attr(cls, attr, hint_txt =None):
        hint_txt = f"\rInput {attr} value: " if hint_txt is None else hint_txt
        while True:
            key = input(hint_txt)
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
                result = int(key) if key.find('.') == -1 else float(key)
                break
            else:
                print("You should input digits.\n")
        return result

    @classmethod
    def output_attrs_dict(cls):
        attrs_dict = dict()
        for attr in cls.__dict__:
            if attr in cls.attrs:
                if attr in cls.file_classes:
                    cls_name, sub_attrs_dict = cls.file_classes[attr].output_attrs_dict()
                    attrs_dict[cls_name] = sub_attrs_dict
                else:
                    attrs_dict[attr] = getattr(cls, attr)
        return cls.__name__, attrs_dict
        
    @classmethod
    def save_info2json(cls):
        attrs_dict = dict()
        if cls.UPDATE:
            for attr in cls.attrs:
                if attr in cls.file_classes:
                    cls_name, sub_attrs_dict = cls.file_classes[attr].output_attrs_dict()
                    attrs_dict[cls_name] = sub_attrs_dict
                else:
                    attrs_dict[attr] = getattr(cls, attr)
            with open(cls.SETTING_FILE, "w") as f:
                json.dump(attrs_dict, f, indent=4) 
            print(f"Configure is saved to {cls.SETTING_FILE}")

