def _get_hubconf_entries():
    """Dynamic hub module generation.

    Functions in `hubconf` module in all tasks are exported from this `hub`.
    """

    import os
    import pathlib
    import importlib

    # `_search_root`: 's3prl'
    _search_root = pathlib.Path(__file__).parent
    _hubconfs = list(_search_root.glob("upstream/*/hubconf.py"))
    _hubconfs += list(_search_root.glob("downstream/*/hubconf.py"))

    # `_hubconf`::Path - path of hubconf.py
    for _hubconf in _hubconfs:
        relpath = _hubconf.relative_to(_search_root)
        try:
            # e.g. 'upstream/vq_wav2vec/hubconf.py' => '.upstream.vq_wav2vec.hubconf'
            _module_name = "." + str(relpath).replace(os.path.sep, ".")[:-3]
            # `hubconf` module
            _module = importlib.import_module(_module_name, package=__package__)

        except ModuleNotFoundError as e:
            if "pase" in _module_name:
                # pase is not installed by default. See upstream/pase/README.md
                continue

            full_package = f"{__package__}{_module_name}"
            print(f'[{__name__}] Warning: can not import {full_package}: {str(e)}. Please see {relpath.parent / "README.md"}')
            continue

        # export all 'callable & public' variables
        for variable_name in dir(_module):
            _variable = getattr(_module, variable_name)
            if callable(_variable) and variable_name[0] != '_':
                # Registration
                globals()[variable_name] = _variable

_get_hubconf_entries()
del globals()["_get_hubconf_entries"]
