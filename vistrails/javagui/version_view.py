###############################################################################
##
## Copyright (C) 2006-2011, University of Utah.
## All rights reserved.
## Contact: vistrails@sci.utah.edu
##
## This file is part of VisTrails.
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  - Neither the name of the University of Utah nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
## THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
## OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
## WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
## OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
##
###############################################################################

"""Version view.

Currently non-interactive, only the selection of a different version is
possible.
TODO :
- Tagging versions
- Deleting versions
- Expanding/collapsing branches of the version tree
"""

from java.awt import Color, Font, Point
from java.awt.event import InputEvent
from java.awt.geom import Rectangle2D

from edu.umd.cs.piccolo import PCanvas, PNode, PLayer
from edu.umd.cs.piccolo.event import PInputEventFilter, PBasicInputEventHandler

import core.db.io
from extras.vistrails_tree_layout_lw import VistrailsTreeLayoutLW
from javagui.utils import FontMetricsImpl


class FontMetrics(object):
    def __init__(self, metrics):
        self._metrics = metrics
        self._height = metrics.getHeight()

    def width(self, string):
        return self._metrics.getStringBounds(string, None).getWidth()

    def height(self):
        return self._height


FONT = Font('Dialog', Font.PLAIN, 15)
FONT_METRICS = FontMetricsImpl(FONT)
GRAPH_METRICS = FontMetrics(FONT_METRICS)

MARGIN_X = 60
MARGIN_Y = 35

HORIZONTAL_POSITION = 300
VERTICAL_POSITION = 20


class PVersionNode(PNode):
    def __init__(self, x, y, version_id, label, view, selected=False, current=False):
        super(PVersionNode, self).__init__()

        self.version_id = version_id

        self.translate(x, y)

        self._fontRect = FONT_METRICS.getStringBounds(label, None)
        self.node_width = int(self._fontRect.getWidth()) + MARGIN_X
        self.node_height = int(self._fontRect.getHeight()) + MARGIN_Y

        self.setBounds(-self.node_width/2, -self.node_height/2,
                       self.node_width, self.node_height)

        self._label = label
        self._selected = selected
        self._current = current

        self._view = view

    def _get_selected(self):
        return self._selected
    def _set_selected(self, s):
        self._selected = s
        self.invalidatePaint()
    selected = property(_get_selected, _set_selected)

    # @Override
    def paint(self, paintContext):
        graphics = paintContext.getGraphics()
        graphics.setFont(FONT);

        graphics.setColor(Color.white)
        oval = (-self.node_width/2,
                -self.node_height/2,
                self.node_width,
                self.node_height)
        graphics.fillOval(*oval)
        if self._selected:
            graphics.setColor(Color.blue)
        else:
            graphics.setColor(Color.black)
        graphics.drawOval(*oval)
        if self.version_id == self._view.current_version:
            graphics.setColor(Color.red)
        else:
            graphics.setColor(Color.black)
        graphics.drawString(
                self._label,
                int(-self._fontRect.getWidth()/2),
                int(-self._fontRect.getY() - self._fontRect.getHeight()/2))


class PConnection(PNode):
    def __init__(self, top, bottom):
        super(PConnection, self).__init__()
        self._top = top
        self._bottom = bottom

        self.computeBounds()

    def computeBounds(self):
        x1 = self._top.getXOffset()
        #y1 = self._top.getYOffset() + self._top.node_height/2
        y1 = self._top.getYOffset()
        x2 = self._bottom.getXOffset()
        #y2 = self._bottom.getYOffset() + self._bottom.node_height/2
        y2 = self._bottom.getYOffset()

        b = Rectangle2D.Double(int(x1), int(y1), 1, 1)
        b.add(Point(int(x2), int(y2)))
        self.setBounds(b)

    # @Override
    def paint(self, paintContext):
        graphics = paintContext.getGraphics()

        graphics.setColor(Color.black)
        graphics.drawLine(
                int(self._top.getXOffset()),
                int(self._top.getYOffset()),
                int(self._bottom.getXOffset()),
                int(self._bottom.getYOffset()))


class VersionSelectingEventHandler(PBasicInputEventHandler):
    def __init__(self, view):
        self._view = view

    # @Override
    def mouseClicked(self, event):
        #eventX = event.getPosition().getX()
        #eventY = event.getPosition().getY()
        #for nodeID, node in self._view._nodes.iteritems():
        #    x = float(eventX - node.x)*2/node.node_width
        #    y = float(eventY - node.y)*2/node.node_height
        #    if x*x + y*y <= 1.0:
        #        self._view.node_clicked(nodeID)
        #        break

        node = event.getPickedNode()
        if node is not None:
            self._view.node_clicked(node.version_id)


class JVersionView(PCanvas):
    def __init__(self, controller, builder_frame):
        super(JVersionView, self).__init__()

        # Use the middle mouse button for panning instead of the left, as we'll
        # use the later to select versions
        self.getPanEventHandler().setEventFilter(PInputEventFilter(
                InputEvent.BUTTON2_MASK))

        self.full_tree = True
        self.refine = False
        self._controller = controller
        self.builder_frame = builder_frame
        self.idScope = self._controller.id_scope

        # Setup the layers
        node_layer = self.getLayer()
        # We override fullPick() for edge_layer to ensure that nodes are picked
        # first
        class CustomPickingLayer(PLayer):
            # @Override
            def fullPick(self, pickPath):
                return (node_layer.fullPick(pickPath) or
                        PLayer.fullPick(self, pickPath))
        connection_layer = CustomPickingLayer()
        self.getCamera().addLayer(0, connection_layer)

        node_layer.addInputEventListener(VersionSelectingEventHandler(self))

        self.set_graph()

        # This method is called back by the controller when an action is
        # performed that creates a new version.
        controller.register_version_callback(self.set_graph)

    def _get_current_version(self):
        return self._controller.current_version
    current_version = property(_get_current_version)

    def set_graph(self):
        self._controller.recompute_terse_graph()

        self._current_terse_graph = self._controller._current_terse_graph
        self._current_full_graph = self._controller._current_full_graph
        self._current_graph_layout = VistrailsTreeLayoutLW()
        self._current_graph_layout.layout_from(
                self._controller.vistrail, self._current_terse_graph,
                GRAPH_METRICS,
                MARGIN_X,
                MARGIN_Y)

        self._controller.current_pipeline = core.db.io.get_workflow(
                self._controller.vistrail, self._controller.current_version)
        self.clicked_version_id = None

        node_layer = self.getCamera().getLayer(1)
        node_layer.removeAllChildren()
        connection_layer = self.getCamera().getLayer(0)
        connection_layer.removeAllChildren()

        self.getCamera().setViewOffset(HORIZONTAL_POSITION,
                                       VERTICAL_POSITION)

        self._nodes = dict()
        self._connections = dict()

        vistrail = self._controller.vistrail
        tm = vistrail.get_tagMap()

        # Create the nodes
        for node in self._current_graph_layout.nodes.itervalues():
            v = node.id
            tag = tm.get(v, None)
            description = vistrail.get_description(v)

            if tag is not None:
                label = tag
            elif description is not None:
                label = description
            else:
                label = ''

            pnode = PVersionNode(
                    int(node.p.x),
                    int(node.p.y),
                    node.id,
                    label,
                    self,
                    selected=(v == self.clicked_version_id),
                    current=(v == self._controller.current_version))
            self._nodes[node.id] = pnode
            node_layer.addChild(pnode)

        # Create the edges
        alreadyVisitedNode = set()
        for node1 in self._current_graph_layout.nodes.itervalues():
            v1 = node1.id
            for node2 in self._current_graph_layout.nodes.itervalues():
                v2 = node2.id
                if node2 in alreadyVisitedNode:
                    pass
                else:
                    if (self._current_terse_graph.has_edge(v1, v2) or
                            self._current_terse_graph.has_edge(v2, v1)):
                        # Detecting the top node1 in order to correctly draw edges
                        if node1.p.y < node2.p.y:
                            topNode = node1
                            bottomNode = node2
                        else:
                            topNode = node2
                            bottomNode = node1
                        c = PConnection(self._nodes[topNode.id],
                                        self._nodes[bottomNode.id])
                        connection_layer.addChild(c)
                        self._connections[(topNode.id, bottomNode.id)] = c
            alreadyVisitedNode.add(v1)

    def node_clicked(self, nodeID):
        if self.clicked_version_id is not None:
            self._nodes[self.clicked_version_id].selected = False
        self.clicked_version_id = nodeID
        self._nodes[nodeID].selected = True

        # Switch version
        self.builder_frame.current_version = nodeID

        self.invalidate()
        self.revalidate()
        self.repaint()
