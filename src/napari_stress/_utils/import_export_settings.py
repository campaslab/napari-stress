import numpy as np
import os
import yaml


def import_settings(parent=None, file_name: str = None) -> dict:
    """Import settings from yaml file.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for dialog, by default None

    Returns
    -------
    dict
        Dictionary of settings
    """
    from qtpy.QtWidgets import QFileDialog

    def ndarray_constructor(loader: yaml.Loader, node: yaml.Node) -> np.ndarray:
        return np.array(loader.construct_sequence(node))

    yaml.add_constructor("tag:yaml.org,2002:ndarray", ndarray_constructor)

    if not file_name:
        file_name, _ = QFileDialog.getOpenFileName(
            parent,
            "Import settings",
            os.path.expanduser("~"),
            "YAML files (*.yaml *.yml)",
        )
    if not file_name:
        return {}
    with open(file_name, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def export_settings(settings: dict, parent=None, file_name: str = None) -> None:
    """Export settings to yaml file.

    Parameters
    ----------
    settings : dict
        Dictionary of settings
    parent : QWidget, optional
        Parent widget for dialog, by default None
    """
    from qtpy.QtWidgets import QFileDialog

    def ndarray_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
        return dumper.represent_sequence("tag:yaml.org,2002:ndarray", data.tolist())

    yaml.add_representer(np.ndarray, ndarray_representer)

    if not file_name:
        file_name, _ = QFileDialog.getSaveFileName(
            parent,
            "Export settings",
            os.path.expanduser("~"),
            "YAML files (*.yaml *.yml)",
        )
    if not file_name:
        return
    with open(file_name, "w") as f:
        yaml.dump(settings, f, default_flow_style=False)
