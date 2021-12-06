"""Downstream tasks"""


def _get_experts():
    """Module instantiation function.
    
    Dynamically register '{task}.expert.DownstreamExpert' to this module.
    """

    import pathlib
    import importlib

    # `.../downstream`
    _search_root = pathlib.Path(__file__).parent
    # `_subdir`: Downstream task directory
    for _subdir in _search_root.iterdir():
        if _subdir.is_dir() and (_subdir / "expert.py").is_file():
            # `_name`: Task name
            _name = str(_subdir.relative_to(_search_root))
            try:
                _module_name = f".{_name}.expert"
                _module = importlib.import_module(_module_name, package=__package__)

            except ModuleNotFoundError as e:
                full_package = f"{__package__}{_module_name}"
                print(f'[{__name__}] Warning: can not import {full_package}: {str(e)}. Pass.')
                continue
            
            # thisModule.{task_name} = {task}.expert.DownstreamExpert
            globals()[_name] = getattr(_module, "DownstreamExpert")


# Instantiate this module
_get_experts()

# Delete `_get_experts` function from this module
del globals()["_get_experts"]
