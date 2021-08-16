_global_para = {
    "category": {
        "red": 0,
        "blue": 1
    },
    "DEBUG_SETTING": {
        True: True,
        False: False
    }
}

_global_dict = {}


def set_value(name, key, value):
    _global_dict[name] = _global_para[key][value]


def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except:
        raise KeyError('''
        Make sure the key is declared before referencing it.
        You can try launching the program from main.py''')
