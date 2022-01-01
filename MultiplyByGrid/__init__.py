# Copyright (c) 2018 Ultimaker B.V.
# Cura is released under the terms of the LGPLv3 or higher.

from . import MultiplyByGrid

from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog("cura")

def getMetaData():
    return {
        "tool": {
            "name": i18n_catalog.i18nc("@label", "Multiply by grid"),
            "description": i18n_catalog.i18nc("@info:tooltip", "Multiply model by grid."),
            "icon": "tool_icon.svg",
            "tool_panel": "MultiplyByGrid.qml",
            "weight": 4
        }
    }

def register(app):
    return { "tool": MultiplyByGrid.MultiplyByGrid() }
