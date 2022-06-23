# Copyright (c) 2018 Ultimaker B.V.
# Cura is released under the terms of the LGPLv3 or higher.

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from UM.Application import Application
from UM.Math.Vector import Vector
from UM.Tool import Tool
from UM.Event import Event, MouseEvent
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Scene.Selection import Selection

from cura.CuraApplication import CuraApplication
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.PickingPass import PickingPass

from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.AddSceneNodeOperation import AddSceneNodeOperation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from cura.Operations.SetParentOperation import SetParentOperation

from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator

from UM.Settings.SettingInstance import SettingInstance

from cura.MultiplyObjectsJob import MultiplyObjectsJob
from UM.Message import Message
from UM.i18n import i18nCatalog
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
import copy
from cura.Arranging.Nest2DArrange import arrange,findNodePlacement
from typing import List, TYPE_CHECKING, Optional, Tuple
from UM.Math.Matrix import Matrix
from UM.Operations.RotateOperation import RotateOperation
from UM.Operations.TranslateOperation import TranslateOperation
from UM.Math.Quaternion import Quaternion

import numpy

class MultiplyByGrid(Tool):
    def __init__(self):
        super().__init__()
        self._shortcut_key = Qt.Key.Key_E
        self._controller = self.getController()

        self._selection_pass = None
        CuraApplication.getInstance().globalContainerStackChanged.connect(self._updateEnabled)

        # Note: if the selection is cleared with this tool active, there is no way to switch to
        # another tool than to reselect an object (by clicking it) because the tool buttons in the
        # toolbar will have been disabled. That is why we need to ignore the first press event
        # after the selection has been cleared.
        Selection.selectionChanged.connect(self._onSelectionChanged)
        self._had_selection = False
        self._skip_press = False

        self._had_selection_timer = QTimer()
        self._had_selection_timer.setInterval(0)
        self._had_selection_timer.setSingleShot(True)
        self._had_selection_timer.timeout.connect(self._selectionChangeDelay)
        
        self._ObjectWidth = 1
        self._ObjectDepth = 1
        self._ScaleX = 50
        self._ScaleY = 50
        self._PcsQty = 1
        self.setExposedProperties(
            "ObjectWidth",
            "ObjectDepth",
            "ScaleX",
            "ScaleY",
            "PcsQty",
        )

    def event(self, event):
        super().event(event)
        modifiers = QApplication.keyboardModifiers()
        ctrl_is_active = modifiers & Qt.KeyboardModifier.ControlModifier

        if event.type == Event.MousePressEvent and MouseEvent.LeftButton in event.buttons and self._controller.getToolsEnabled():
            return
            if ctrl_is_active:
                self._controller.setActiveTool("TranslateTool")
                return

            if self._skip_press:
                # The selection was previously cleared, do not add/remove an anti-support mesh but
                # use this click for selection and reactivating this tool only.
                self._skip_press = False
                return

            if self._selection_pass is None:
                # The selection renderpass is used to identify objects in the current view
                self._selection_pass = Application.getInstance().getRenderer().getRenderPass("selection")
            picked_node = self._controller.getScene().findObject(self._selection_pass.getIdAtPosition(event.x, event.y))
            if not picked_node:
                # There is no slicable object at the picked location
                return

            node_stack = picked_node.callDecoration("getStack")
            if node_stack:
                if node_stack.getProperty("anti_overhang_mesh", "value"):
                    self._removeEraserMesh(picked_node)
                    return

                elif node_stack.getProperty("support_mesh", "value") or node_stack.getProperty("infill_mesh", "value") or node_stack.getProperty("cutting_mesh", "value"):
                    # Only "normal" meshes can have anti_overhang_meshes added to them
                    return

            # Create a pass for picking a world-space location from the mouse location
            active_camera = self._controller.getScene().getActiveCamera()
            picking_pass = PickingPass(active_camera.getViewportWidth(), active_camera.getViewportHeight())
            picking_pass.render()

            picked_position = picking_pass.getPickedPosition(event.x, event.y)

            # Add the anti_overhang_mesh cube at the picked location
            #self._createEraserMesh(picked_node, picked_position)
            #job = MyMultiplyObjectsJob(Selection.getAllSelectedObjects(), 5)
            #job.start()
            
            self._objects = Selection.getAllSelectedObjects()
            self._count = 5
            self._min_offset = 5
            i18n_catalog = i18nCatalog("cura")
            
            status_message = Message(i18n_catalog.i18nc("@info:status", "Multiplying and placing objects"), lifetime = 0,
                                     dismissable = False, progress = 0,
                                     title = i18n_catalog.i18nc("@info:title", "Placing Objects"))
            status_message.show()
            scene = Application.getInstance().getController().getScene()

            global_container_stack = Application.getInstance().getGlobalContainerStack()
            if global_container_stack is None:
                return  # We can't do anything in this case.

            root = scene.getRoot()

            processed_nodes = []  # type: List[SceneNode]
            nodes = []

            fixed_nodes = []
            for node_ in DepthFirstIterator(root):
                # Only count sliceable objects
                if node_.callDecoration("isSliceable"):
                    fixed_nodes.append(node_)

            for node in self._objects:
                # If object is part of a group, multiply group
                current_node = node
                while current_node.getParent() and (current_node.getParent().callDecoration("isGroup") or current_node.getParent().callDecoration("isSliceable")):
                    current_node = current_node.getParent()

                if current_node in processed_nodes:
                    continue
                processed_nodes.append(current_node)

                for _ in range(self._count):
                    new_node = copy.deepcopy(node)

                    # Same build plate
                    build_plate_number = current_node.callDecoration("getBuildPlateNumber")
                    new_node.callDecoration("setBuildPlateNumber", build_plate_number)
                    for child in new_node.getChildren():
                        child.callDecoration("setBuildPlateNumber", build_plate_number)

                    nodes.append(new_node)

            found_solution_for_all = True
            if nodes:
                found_solution_for_all = self._arrange(nodes, Application.getInstance().getBuildVolume(), fixed_nodes,
                                                 factor = 10000, add_new_nodes_in_scene = True)
            status_message.hide()

            if not found_solution_for_all:
                no_full_solution_message = Message(
                    i18n_catalog.i18nc("@info:status", "Unable to find a location within the build volume for all objects"),
                    title = i18n_catalog.i18nc("@info:title", "Placing Object"))
                """no_full_solution_message = Message(
                    i18n_catalog.i18nc("@info:status", "Unable to find a location within the build volume for all objects"),
                    title = i18n_catalog.i18nc("@info:title", "Placing Object"),
                    message_type = Message.MessageType.WARNING"""
                no_full_solution_message.show()

    def _arrange(self, nodes_to_arrange: List["SceneNode"], build_volume: "BuildVolume", fixed_nodes: Optional[List["SceneNode"]] = None, factor = 10000, add_new_nodes_in_scene: bool = False) -> bool:
        """
        Find placement for a set of scene nodes, and move them by using a single grouped operation.
        :param nodes_to_arrange: The list of nodes that need to be moved.
        :param build_volume: The build volume that we want to place the nodes in. It gets size & disallowed areas from this.
        :param fixed_nodes: List of nods that should not be moved, but should be used when deciding where the others nodes
                            are placed.
        :param factor: The library that we use is int based. This factor defines how accuracte we want it to be.
        :param add_new_nodes_in_scene: Whether to create new scene nodes before applying the transformations and rotations
        :return: found_solution_for_all: Whether the algorithm found a place on the buildplate for all the objects
        """
        scene_root = Application.getInstance().getController().getScene().getRoot()
        found_solution_for_all, node_items = findNodePlacement(nodes_to_arrange, build_volume, fixed_nodes, factor)

        not_fit_count = 0
        grouped_operation = GroupedOperation()
        x = 0
        for node, node_item in zip(nodes_to_arrange, node_items):
            if add_new_nodes_in_scene:
                grouped_operation.addOperation(AddSceneNodeOperation(node, scene_root))
            
            if node_item.binId() == 0:
                # We found a spot for it
                #rotation_matrix = Matrix()
                #rotation_matrix.setByRotationAxis(node_item.rotation(), Vector(0, -1, 0))
                #grouped_operation.addOperation(RotateOperation(node, Quaternion.fromMatrix(rotation_matrix)))
                x += 10
                grouped_operation.addOperation(TranslateOperation(node, Vector(x, 0, 0)))
            else:
                # We didn't find a spot
                grouped_operation.addOperation(
                    TranslateOperation(node, Vector(200, node.getWorldPosition().y, -not_fit_count * 20), set_position = True))
                not_fit_count += 1
            
        grouped_operation.push()
        
        return found_solution_for_all

    def doMultiply(self):
        #self.propertyChanged.emit()
        
        build_volume = Application.getInstance().getBuildVolume()
        machine_width = build_volume.getWidth() - (build_volume.getEdgeDisallowedSize() * 2)
        machine_depth = build_volume.getDepth() - (build_volume.getEdgeDisallowedSize() * 2)
       
        scene_root = Application.getInstance().getController().getScene().getRoot()
        #item = scene_root.getItem(1);
        #node = item["node"]
        #self._ObjectWidth = node.getBoundingBox().width
        #return
        maxx = -machine_width;
        miny = machine_depth;
        mynodes = [];
        firstinit = 1
        self._PcsQty = 1
        
        # To the corner
        grouped_operation = GroupedOperation()
        for node in Selection.getAllSelectedObjects():
            node_width = node.getBoundingBox().width                
            node_depth = node.getBoundingBox().depth
            curx = float(node.getBoundingBox().center.x)
            cury = float(node.getBoundingBox().center.z)
            newx = -curx-(machine_width/2)+(node_width/2)
            newy = -cury+(machine_depth/2)-(node_depth/2)
            grouped_operation.addOperation(TranslateOperation(node, Vector(newx, 0, newy)))
        grouped_operation.push()
        
        grouped_operation = GroupedOperation()
        for iy in range(0, self._ScaleY):
            for ix in range(0, self._ScaleX):
                if ix == 0 and iy == 0: continue
                for node in Selection.getAllSelectedObjects():
                    if firstinit:
                        mynodes.append(node);
                    node_width = node.getBoundingBox().width                
                    node_depth = node.getBoundingBox().depth
                    curx = float(node.getBoundingBox().center.x)
                    cury = float(node.getBoundingBox().center.z)
                    newx = ix*(self._ObjectWidth+node_width)
                    newy = -iy*(self._ObjectDepth+node_depth)
                    if newx + curx > machine_width/2-(node_width/2): continue
                    if newy + cury < -machine_depth/2+(node_depth/2): continue
                    maxx = max(maxx, newx)
                    miny = min(miny, newy)
                    
                    new_node = copy.deepcopy(node)
                    self._PcsQty += 1
                    mynodes.append(new_node)
                    # Same build plate
                    build_plate_number = node.callDecoration("getBuildPlateNumber")
                    new_node.callDecoration("setBuildPlateNumber", build_plate_number)
                    for child in new_node.getChildren():
                        child.callDecoration("setBuildPlateNumber", build_plate_number)
                    grouped_operation.addOperation(AddSceneNodeOperation(new_node, scene_root))
                    grouped_operation.addOperation(TranslateOperation(new_node, Vector(newx, 0, newy)))
                firstinit = 0
        grouped_operation.push()

        # Move all nodes to thhe center
        grouped_operation = GroupedOperation()
        newx = -curx - (maxx / 2)
        newy = -cury - (miny / 2)
        for node in mynodes:
            grouped_operation.addOperation(TranslateOperation(node, Vector(newx, 0, newy)))
        grouped_operation.push()

        self.propertyChanged.emit()
        
        # if node.collidesWithAreas(self.getDisallowedAreas())
 
    def _createEraserMesh(self, parent: CuraSceneNode, position: Vector):
        node = CuraSceneNode()

        node.setName("Eraser")
        node.setSelectable(True)
        node.setCalculateBoundingBox(True)
        mesh = self._createCube(20)
        node.setMeshData(mesh.build())
        node.calculateBoundingBoxMesh()

        active_build_plate = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        node.addDecorator(BuildPlateDecorator(active_build_plate))
        node.addDecorator(SliceableObjectDecorator())

        stack = node.callDecoration("getStack") # created by SettingOverrideDecorator that is automatically added to CuraSceneNode
        settings = stack.getTop()

        definition = stack.getSettingDefinition("anti_overhang_mesh")
        new_instance = SettingInstance(definition, settings)
        new_instance.setProperty("value", True)
        new_instance.resetState()  # Ensure that the state is not seen as a user state.
        settings.addInstance(new_instance)

        op = GroupedOperation()
        # First add node to the scene at the correct position/scale, before parenting, so the eraser mesh does not get scaled with the parent
        op.addOperation(AddSceneNodeOperation(node, self._controller.getScene().getRoot()))
        op.addOperation(SetParentOperation(node, parent))
        op.push()
        node.setPosition(position, CuraSceneNode.TransformSpace.World)

        CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)

    def _removeEraserMesh(self, node: CuraSceneNode):
        parent = node.getParent()
        if parent == self._controller.getScene().getRoot():
            parent = None

        op = RemoveSceneNodeOperation(node)
        op.push()

        if parent and not Selection.isSelected(parent):
            Selection.add(parent)

        CuraApplication.getInstance().getController().getScene().sceneChanged.emit(node)

    def _updateEnabled(self):
        plugin_enabled = False

        global_container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if global_container_stack:
            plugin_enabled = global_container_stack.getProperty("anti_overhang_mesh", "enabled")

        CuraApplication.getInstance().getController().toolEnabledChanged.emit(self._plugin_id, plugin_enabled)

    def _onSelectionChanged(self):
        # When selection is passed from one object to another object, first the selection is cleared
        # and then it is set to the new object. We are only interested in the change from no selection
        # to a selection or vice-versa, not in a change from one object to another. A timer is used to
        # "merge" a possible clear/select action in a single frame
        if Selection.hasSelection() != self._had_selection:
            self._had_selection_timer.start()

    def _selectionChangeDelay(self):
        has_selection = Selection.hasSelection()
        if not has_selection and self._had_selection:
            self._skip_press = True
        else:
            self._skip_press = False

        self._had_selection = has_selection

    def _createCube(self, size):
        mesh = MeshBuilder()

        # Can't use MeshBuilder.addCube() because that does not get per-vertex normals
        # Per-vertex normals require duplication of vertices
        s = size / 2
        verts = [ # 6 faces with 4 corners each
            [-s, -s,  s], [-s,  s,  s], [ s,  s,  s], [ s, -s,  s],
            [-s,  s, -s], [-s, -s, -s], [ s, -s, -s], [ s,  s, -s],
            [ s, -s, -s], [-s, -s, -s], [-s, -s,  s], [ s, -s,  s],
            [-s,  s, -s], [ s,  s, -s], [ s,  s,  s], [-s,  s,  s],
            [-s, -s,  s], [-s, -s, -s], [-s,  s, -s], [-s,  s,  s],
            [ s, -s, -s], [ s, -s,  s], [ s,  s,  s], [ s,  s, -s]
        ]
        mesh.setVertices(numpy.asarray(verts, dtype=numpy.float32))

        indices = []
        for i in range(0, 24, 4): # All 6 quads (12 triangles)
            indices.append([i, i+2, i+1])
            indices.append([i, i+3, i+2])
        mesh.setIndices(numpy.asarray(indices, dtype=numpy.int32))

        mesh.calculateNormals()
        return mesh

    def getObjectWidth(self) -> float:
        return self._ObjectWidth

    def getObjectDepth(self) -> float:
        return self._ObjectDepth

    def getScaleX(self):
        return self._ScaleX

    def getScaleY(self):
        return self._ScaleY

    def setObjectWidth(self, width):
        self._ObjectWidth = float(width)
        
    def setObjectDepth(self, height):
        self._ObjectDepth = float(height)

    def setScaleX(self, scale):
        self._ScaleX = round(float(scale))

    def setScaleY(self, scale):
        self._ScaleY = round(float(scale))

    def getPcsQty(self):
        return self._PcsQty

    def setPcsQty(self, va):
        self._PcsQty = int(float(va))
