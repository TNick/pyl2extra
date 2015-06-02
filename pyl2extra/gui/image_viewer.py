#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code was copied from github_ repository
for MDIImageViewer. According to license_ page it is released under
GNU General Public License version 3.

 _tpgit: http://tpgit.github.io/MDIImageViewer/
 _github: https://github.com/tpgit/MDIImageViewer
 _license: http://www.riverbankcomputing.com/commercial/pyqt

"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__authors__ = "TPWorks (?)"
__copyright__ = "(?)"
__credits__ = []
__license__ = "GNU General Public License version 3"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"


# This is only needed for Python v2 but is harmless for Python v3.
import sip
try:
    sip.setapi('QDate', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QUrl', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QVariant', 2)
    sip.setapi('QString', 2)
except ValueError:
    pass

import Image, ImageQt
import os
import sys
from PyQt4 import (QtCore, QtGui)

from pyl2extra.gui.guihelpers import get_icon

class SynchableGraphicsView(QtGui.QGraphicsView):
    """|QGraphicsView| that can synchronize panning & zooming of multiple
    instances.
    Also adds support for various scrolling operations and mouse wheel
    zooming."""

    def __init__(self, scene=None, parent=None):
        """:param scene: initial |QGraphicsScene|
        :type scene: QGraphicsScene or None
        :param QWidget: parent widget
        :type QWidget: QWidget or None"""
        if scene:
            super(SynchableGraphicsView, self).__init__(scene, parent)
        else:
            super(SynchableGraphicsView, self).__init__(parent)

        self._handDrag = False #disable panning view by dragging
        self.clearTransformChanges()
        self.connectSbarSignals(self.scrollChanged)

    # ------------------------------------------------------------------

    #Signals

    transformChanged = QtCore.pyqtSignal()
    """Transformed Changed **Signal**.
    Emitted whenever the |QGraphicsView| Transform matrix has been
    changed."""

    scrollChanged = QtCore.pyqtSignal()
    """Scroll Changed **Signal**.
    Emitted whenever the scrollbar position or range has changed."""

    wheelNotches = QtCore.pyqtSignal(float)
    """Wheel Notches **Signal** (*float*).
    Emitted whenever the mouse wheel has been rolled. A wheelnotch is
    equal to wheel delta / 240"""

    def connectSbarSignals(self, slot):
        """Connect to scrollbar changed signals to synchronize panning.
        :param slot: slot to connect scrollbar signals to."""
        sbar = self.horizontalScrollBar()
        sbar.valueChanged.connect(slot, type=QtCore.Qt.UniqueConnection)
        #sbar.sliderMoved.connect(slot, type=QtCore.Qt.UniqueConnection)
        sbar.rangeChanged.connect(slot, type=QtCore.Qt.UniqueConnection)

        sbar = self.verticalScrollBar()
        sbar.valueChanged.connect(slot, type=QtCore.Qt.UniqueConnection)
        #sbar.sliderMoved.connect(slot, type=QtCore.Qt.UniqueConnection)
        sbar.rangeChanged.connect(slot, type=QtCore.Qt.UniqueConnection)

        #self.scrollChanged.connect(slot, type=QtCore.Qt.UniqueConnection)

    def disconnectSbarSignals(self):
        """Disconnect from scrollbar changed signals."""
        sbar = self.horizontalScrollBar()
        sbar.valueChanged.disconnect()
        #sbar.sliderMoved.disconnect()
        sbar.rangeChanged.disconnect()

        sbar = self.verticalScrollBar()
        sbar.valueChanged.disconnect()
        #sbar.sliderMoved.disconnect()
        sbar.rangeChanged.disconnect()

    # ------------------------------------------------------------------

    @property
    def handDragging(self):
        """Hand dragging state (*bool*)"""
        return self._handDrag

    @property
    def scrollState(self):
        """Tuple of percentage of scene extents
        *(sceneWidthPercent, sceneHeightPercent)*"""
        centerPoint = self.mapToScene(self.viewport().width()/2,
                                      self.viewport().height()/2)
        sceneRect = self.sceneRect()
        centerWidth = centerPoint.x() - sceneRect.left()
        centerHeight = centerPoint.y() - sceneRect.top()
        sceneWidth = sceneRect.width()
        sceneHeight = sceneRect.height()

        sceneWidthPercent = centerWidth / sceneWidth if sceneWidth != 0 else 0
        sceneHeightPercent = centerHeight / sceneHeight if sceneHeight != 0 else 0
        return (sceneWidthPercent, sceneHeightPercent)

    @scrollState.setter
    def scrollState(self, state):
        sceneWidthPercent, sceneHeightPercent = state
        x = (sceneWidthPercent * self.sceneRect().width() +
             self.sceneRect().left())
        y = (sceneHeightPercent * self.sceneRect().height() +
             self.sceneRect().top())
        self.centerOn(x, y)

    @property
    def zoomFactor(self):
        """Zoom scale factor (*float*)."""
        return self.transform().m11()

    @zoomFactor.setter
    def zoomFactor(self, newZoomFactor):
        newZoomFactor = newZoomFactor / self.zoomFactor
        self.scale(newZoomFactor, newZoomFactor)

    # ------------------------------------------------------------------

    def wheelEvent(self, wheelEvent):
        """Overrides the wheelEvent to handle zooming.
        :param QWheelEvent wheelEvent: instance of |QWheelEvent|"""
        assert isinstance(wheelEvent, QtGui.QWheelEvent)
        if wheelEvent.modifiers() & QtCore.Qt.ControlModifier:
            self.wheelNotches.emit(wheelEvent.delta() / 240.0)
            wheelEvent.accept()
        else:
            super(SynchableGraphicsView, self).wheelEvent(wheelEvent)

    def keyReleaseEvent(self, keyEvent):
        """Overrides to make sure key release passed on to other classes.
        :param QKeyEvent keyEvent: instance of |QKeyEvent|"""
        assert isinstance(keyEvent, QtGui.QKeyEvent)
        #print("graphicsView keyRelease count=%d, autoRepeat=%s" %
              #(keyEvent.count(), keyEvent.isAutoRepeat()))
        keyEvent.ignore()
        #super(SynchableGraphicsView, self).keyReleaseEvent(keyEvent)

    # ------------------------------------------------------------------

    def checkTransformChanged(self):
        """Return True if view transform has changed.
        Overkill. For current implementation really only need to check
        if ``m11()`` has changed.
        :rtype: bool"""
        delta = 0.001
        result = False

        def different(t, u):
            if u == 0.0:
                d = abs(t - u)
            else:
                d = abs((t - u) / u)
            return d > delta

        t = self.transform()
        u = self._transform

        if False:
            print("t = ")
            self.dumpTransform(t, "    ")
            print("u = ")
            self.dumpTransform(u, "    ")
            print("")

        if (different(t.m11(), u.m11()) or
            different(t.m22(), u.m22()) or
            different(t.m12(), u.m12()) or
            different(t.m21(), u.m21()) or
            different(t.m31(), u.m31()) or
            different(t.m32(), u.m32())):
                
            self._transform = t
            self.transformChanged.emit()
            result = True
        return result

    def clearTransformChanges(self):
        """Reset view transform changed info."""
        self._transform = self.transform()

    def scrollToTop(self):
        """Scroll view to top."""
        sbar = self.verticalScrollBar()
        sbar.setValue(sbar.minimum())

    def scrollToBottom(self):
        """Scroll view to bottom."""
        sbar = self.verticalScrollBar()
        sbar.setValue(sbar.maximum())

    def scrollToBegin(self):
        """Scroll view to left edge."""
        sbar = self.horizontalScrollBar()
        sbar.setValue(sbar.minimum())

    def scrollToEnd(self):
        """Scroll view to right edge."""
        sbar = self.horizontalScrollBar()
        sbar.setValue(sbar.maximum())

    def centerView(self):
        """Center view."""
        sbar = self.verticalScrollBar()
        sbar.setValue((sbar.maximum() + sbar.minimum())/2)
        sbar = self.horizontalScrollBar()
        sbar.setValue((sbar.maximum() + sbar.minimum())/2)

    def enableScrollBars(self, enable):
        """Set visiblility of the view's scrollbars.
        :param bool enable: True to enable the scrollbars """
        if enable:
            self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        else:
            self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def enableHandDrag(self, enable):
        """Set whether dragging the view with the hand cursor is allowed.
        :param bool enable: True to enable hand dragging """
        if enable:
            if not self._handDrag:
                self._handDrag = True
                self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        else:
            if self._handDrag:
                self._handDrag = False
                self.setDragMode(QtGui.QGraphicsView.NoDrag)

    # ------------------------------------------------------------------

    def dumpTransform(self, t, padding=""):
        """Dump the transform t to stdout.
        :param t: the transform to dump
        :param str padding: each line is preceded by padding"""
        print("%s%5.3f %5.3f %5.3f" % (padding, t.m11(), t.m12(), t.m13()))
        print("%s%5.3f %5.3f %5.3f" % (padding, t.m21(), t.m22(), t.m23()))
        print("%s%5.3f %5.3f %5.3f" % (padding, t.m31(), t.m32(), t.m33()))


class ImageViewer(QtGui.QFrame):
    """
    Image Viewer than can pan & zoom images).

    Parameters
    ----------
    pixmap : QPixmap, str or None
        The image to display. If this is a string the file is loaded into a
        QPixmap internally.
    name : str or None
        name associated with this ImageViewer.
    """

    def __init__(self, pixmap=None, name=None):
        super(ImageViewer, self).__init__()
        #self.setFrameStyle(QtGui.QFrame.Sunken | QtGui.QFrame.StyledPanel)
        self.setFrameStyle(QtGui.QFrame.NoFrame)

        self._relativeScale = 1.0 #scale relative to other ImageViewer instances
        self._zoomFactorDelta = 1.25

        self._scene = QtGui.QGraphicsScene()
        self._view = SynchableGraphicsView(self._scene)

        self._view.setInteractive(False)
        #self._view.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self._view.setViewportUpdateMode(QtGui.QGraphicsView.MinimalViewportUpdate)
        #self._view.setViewportUpdateMode(QtGui.QGraphicsView.SmartViewportUpdate)
        #self._view.setTransformationAnchor(QtGui.QGraphicsView.NoAnchor)
        self._view.setTransformationAnchor(QtGui.QGraphicsView.AnchorViewCenter)

        #pass along underlying signals
        self._scene.changed.connect(self.sceneChanged)
        self._view.transformChanged.connect(self.transformChanged)
        self._view.scrollChanged.connect(self.scrollChanged)
        self._view.wheelNotches.connect(self.handleWheelNotches)

        gridSize = 10
        backgroundPixmap = QtGui.QPixmap(gridSize*2, gridSize*2)
        backgroundPixmap.fill(QtGui.QColor("powderblue"))
        painter = QtGui.QPainter(backgroundPixmap)
        backgroundColor = QtGui.QColor("palegoldenrod")
        painter.fillRect(0, 0, gridSize, gridSize, backgroundColor)
        painter.fillRect(gridSize, gridSize, gridSize, gridSize, backgroundColor)
        painter.end()

        self._scene.setBackgroundBrush(QtGui.QBrush(backgroundPixmap))
        self._view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        self._pixmapItem = QtGui.QGraphicsPixmapItem(scene=self._scene)
        if pixmap:
            if isinstance(pixmap, basestring):
                ldd = Image.open(pixmap)
                ldd = ImageQt.ImageQt(ldd)
                self.pixmap = QtGui.QPixmap.fromImage(ldd)
            else:
                self.pixmap = pixmap

        #        rect = self._scene.addRect(QtCore.QRectF(0, 0, 100, 100),
        #                                   QtGui.QPen(QtGui.QColor("red")))
        #        rect.setZValue(1.0)

        layout = QtGui.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        #layout.setSpacing(0)

        self._label = QtGui.QLabel()
        #self._label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self._label.setFrameStyle(QtGui.QFrame.Panel)
        self._label.setAutoFillBackground(True)
        self._label.setBackgroundRole(QtGui.QPalette.ToolTipBase)
        self.viewName = name

        layout.addWidget(self._view, 0, 0)
        layout.addWidget(self._label, 0, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setLayout(layout)

        self.enableScrollBars(True)
        self._label.hide()
        self._view.show()

    # ------------------------------------------------------------------

    sceneChanged = QtCore.pyqtSignal('QList<QRectF>')
    """Scene Changed **Signal**.
    Emitted whenever the |QGraphicsScene| content changes."""

    transformChanged = QtCore.pyqtSignal()
    """Transformed Changed **Signal**.
    Emitted whenever the |QGraphicsView| Transform matrix has been changed."""

    scrollChanged = QtCore.pyqtSignal()
    """Scroll Changed **Signal**.
    Emitted whenever the scrollbar position or range has changed."""

    def connectSbarSignals(self, slot):
        """Connect to scrollbar changed signals.
        :param slot: slot to connect scrollbar signals to."""
        self._view.connectSbarSignals(slot)

    def disconnectSbarSignals(self):
        self._view.disconnectSbarSignals()

    # ------------------------------------------------------------------

    @property
    def pixmap(self):
        """The currently viewed |QPixmap| (*QPixmap*)."""
        return self._pixmapItem.pixmap()

    @pixmap.setter
    def pixmap(self, pixmap):
        assert isinstance(pixmap, QtGui.QPixmap)
        self._pixmapItem.setPixmap(pixmap)
        self._pixmapItem.setOffset(-pixmap.width()/2.0, -pixmap.height()/2.0)
        self._pixmapItem.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self.fitToWindow()

    @property
    def labelled(self):
        """Tell if the label is visible or not."""
        return self._label.isVisible()

    @labelled.setter
    def labelled(self, visible):
        if visible:
            self._label.show()
        else:
            self._label.hide()

    @property
    def viewName(self):
        """The name associated with ImageViewer (*str*)."""
        return self._name

    @viewName.setter
    def viewName(self, name):
        if self._label:
            if name:
                self._label.setText("<b>%s</b>" % name)
                self._label.show()
            else:
                self._label.setText("")
                self._label.hide()
        self._name = name

    @property
    def handDragging(self):
        """Hand dragging state (*bool*)"""
        return self._view.handDragging

    @property
    def scrollState(self):
        """Tuple of percentage of scene extents
        *(sceneWidthPercent, sceneHeightPercent)*"""
        return self._view.scrollState

    @scrollState.setter
    def scrollState(self, state):
        self._view.scrollState = state

    @property
    def zoomFactor(self):
        """Zoom scale factor (*float*)."""
        return self._view.zoomFactor

    @zoomFactor.setter
    def zoomFactor(self, newZoomFactor):
        if newZoomFactor < 1.0:
            self._pixmapItem.setTransformationMode(QtCore.Qt.SmoothTransformation)
        else:
            self._pixmapItem.setTransformationMode(QtCore.Qt.FastTransformation)
        self._view.zoomFactor = newZoomFactor

    @property
    def _horizontalScrollBar(self):
        """Get the ImageViewer horizontal scrollbar widget (*QScrollBar*).
        (Only used for debugging purposes)"""
        return self._view.horizontalScrollBar()

    @property
    def _verticalScrollBar(self):
        """Get the ImageViewer vertical scrollbar widget (*QScrollBar*).
        (Only used for debugging purposes)"""
        return self._view.verticalScrollBar()

    @property
    def _sceneRect(self):
        """Get the ImageViewer sceneRect (*QRectF*).
        (Only used for debugging purposes)"""
        return self._view.sceneRect()

    # ------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def scrollToTop(self):
        """Scroll to top of image."""
        self._view.scrollToTop()

    @QtCore.pyqtSlot()
    def scrollToBottom(self):
        """Scroll to bottom of image."""
        self._view.scrollToBottom()

    @QtCore.pyqtSlot()
    def scrollToBegin(self):
        """Scroll to left side of image."""
        self._view.scrollToBegin()

    @QtCore.pyqtSlot()
    def scrollToEnd(self):
        """Scroll to right side of image."""
        self._view.scrollToEnd()

    @QtCore.pyqtSlot()
    def centerView(self):
        """Center image in view."""
        self._view.centerView()

    @QtCore.pyqtSlot(bool)
    def enableScrollBars(self, enable):
        """Set visiblility of the view's scrollbars.
        :param bool enable: True to enable the scrollbars """
        self._view.enableScrollBars(enable)

    @QtCore.pyqtSlot(bool)
    def enableHandDrag(self, enable):
        """Set whether dragging the view with the hand cursor is allowed.
        :param bool enable: True to enable hand dragging """
        self._view.enableHandDrag(enable)

    @QtCore.pyqtSlot()
    def zoomIn(self):
        """Zoom in on image."""
        self.scaleImage(self._zoomFactorDelta)

    @QtCore.pyqtSlot()
    def zoomOut(self):
        """Zoom out on image."""
        self.scaleImage(1 / self._zoomFactorDelta)

    @QtCore.pyqtSlot()
    def actualSize(self):
        """Change zoom to show image at actual size.
        (image pixel is equal to screen pixel)"""
        self.scaleImage(1.0, combine=False)

    @QtCore.pyqtSlot()
    def fitToWindow(self):
        """Fit image within view."""
        if not self._pixmapItem.pixmap():
            return
        self._pixmapItem.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self._view.fitInView(self._pixmapItem, QtCore.Qt.KeepAspectRatio)
        self._view.checkTransformChanged()

    @QtCore.pyqtSlot()
    def fitWidth(self):
        """Fit image width to view width."""
        if not self._pixmapItem.pixmap():
            return
        margin = 2
        viewRect = self._view.viewport().rect().adjusted(margin, margin,
                                                         -margin, -margin)
        factor = viewRect.width() / self._pixmapItem.pixmap().width()
        self.scaleImage(factor, combine=False)

    @QtCore.pyqtSlot()
    def fitHeight(self):
        """Fit image height to view height."""
        if not self._pixmapItem.pixmap():
            return
        margin = 2
        viewRect = self._view.viewport().rect().adjusted(margin, margin,
                                                         -margin, -margin)
        factor = viewRect.height() / self._pixmapItem.pixmap().height()
        self.scaleImage(factor, combine=False)

    # ------------------------------------------------------------------

    def handleWheelNotches(self, notches):
        """Handle wheel notch event from underlying |QGraphicsView|.
        :param float notches: Mouse wheel notches"""
        self.scaleImage(self._zoomFactorDelta ** notches)

    def closeEvent(self, event):
        """Overriden in order to disconnect scrollbar signals before
        closing.
        :param QEvent event: instance of a |QEvent|

        If this isn't done Python crashes!"""
        #self.scrollChanged.disconnect() #doesn't prevent crash
        self.disconnectSbarSignals()
        super(ImageViewer, self).closeEvent(event)

    # ------------------------------------------------------------------

    def scaleImage(self, factor, combine=True):
        """Scale image by factor.
        :param float factor: either new :attr:`zoomFactor` or amount to scale
                             current :attr:`zoomFactor`
        :param bool combine: if ``True`` scales the current
                             :attr:`zoomFactor` by factor.  Otherwise
                             just sets :attr:`zoomFactor` to factor"""
        if not self._pixmapItem.pixmap():
            return

        if combine:
            self.zoomFactor = self.zoomFactor * factor
        else:
            self.zoomFactor = factor
        self._view.checkTransformChanged()

    def dumpTransform(self):
        """Dump view transform to stdout."""
        self._view.dumpTransform(self._view.transform(), " "*4)

    @staticmethod
    def pixmap_from_file(fname):
        """Loads a pixmap from a file"""
        pixmap = QtGui.QPixmap(fname)
        if (not pixmap or
            pixmap.width()==0 or pixmap.height==0):
                
            ldd = Image.open(fname)
            ldd = ImageQt.ImageQt(ldd)
            pixmap = QtGui.QPixmap.fromImage(ldd)
        return pixmap

    @staticmethod
    def pixmap_from_array(array):
        """Loads a pixmap from an array"""
        ldd = Image.fromarray(array)
        ldd = ImageQt.ImageQt(ldd)
        pixmap = QtGui.QPixmap.fromImage(ldd)
        return pixmap, ldd


class NavigMixin(object):
    """
    Mixin used by implamentations to navigate a set of images.

    Parameters
    ----------
    img_set : list
        Members can be strings (file paths) or QPixmap instance.
    """
    def __init__(self, img_set=None):
        super(NavigMixin, self).__init__()
        self.img_set = img_set
        self.crt = 0

    def createNavigActions(self, menu=None, toolbar=None):
        """Create common actions for the menus."""
        self.navig_first_act = QtGui.QAction(
            get_icon('zoom_height.png'),
            "First", self,
            shortcut="Ctrl+Home",
            triggered=self._imageViewer.navig_first)

        self.navig_last_act = QtGui.QAction(
            get_icon('zoom_height.png'),
            "First", self,
            shortcut="Ctrl+End",
            triggered=self._imageViewer.navig_last)

        self.navig_prev_act = QtGui.QAction(
            get_icon('zoom_height.png'),
            "First", self,
            shortcut="LeftArrow",
            triggered=self._imageViewer.navig_prev)

        self.navig_next_act = QtGui.QAction(
            get_icon('zoom_height.png'),
            "First", self,
            shortcut="Alt+LeftArrow",
            triggered=self._imageViewer.navig_next)

        if menu:
            menu.addAction(self.navig_first_act)
            menu.addAction(self.navig_next_act)
            menu.addAction(self.navig_prev_act)
            menu.addAction(self.navig_last_act)

        if toolbar:
            toolbar.addAction(self.navig_first_act)
            toolbar.addAction(self.navig_next_act)
            toolbar.addAction(self.navig_prev_act)
            toolbar.addAction(self.navig_last_act)

    @QtCore.pyqtSlot()
    def navig_first(self):
        """Go to first image in the set."""
        self.crt = 0
        self.show_crt_nav_img()

    @QtCore.pyqtSlot()
    def navig_last(self):
        """Go to last image in the set."""
        self.crt = len(self.img_set) - 1
        self.show_crt_nav_img()

    @QtCore.pyqtSlot()
    def navig_prev(self):
        """Go to previous image in the set."""
        self.crt = self.crt - 1
        if self.crt < 0:
            self.crt = len(self.img_set) - 1
        self.show_crt_nav_img()

    @QtCore.pyqtSlot()
    def navig_next(self):
        """Go to next image in the set."""
        self.crt = self.crt + 1
        if self.crt >= len(self.img_set):
            self.crt = 0
        self.show_crt_nav_img()

    def show_crt_nav_img(self):
        """
        Present current image.
        """
        assert len(self.img_set) > 0
        assert self.crt >= 0 and self.crt < len(self.img_set)
        pixmap = self.img_set[self.crt]
        if isinstance(pixmap, basestring):
            pixmap = ImageViewer.pixmap_from_file(pixmap)
        self.pixmap = pixmap


class ActionsMixin(object):
    """
    Mixin used by implamentations to create actions and menus.
    """
    def __init__(self):
        super(ActionsMixin, self).__init__()

    def createActions(self):
        """Create common actions for the menus."""

        #File Actions
        self.exitAct = QtGui.QAction(
            get_icon('door_in.png'),
            "E&xit", self,
            shortcut=QtGui.QKeySequence.Quit,
            statusTip="Exit the application",
            triggered=QtGui.qApp.closeAllWindows)

        #view actions
        self.scrollToTopAct = QtGui.QAction(
            get_icon('wrapping_in_front_top.png'),
            "&Top", self,
            shortcut=QtGui.QKeySequence.MoveToStartOfDocument,
            triggered=self._imageViewer.scrollToTop)

        self.scrollToBottomAct = QtGui.QAction(
            get_icon('wrapping_in_front_bottom.png'),
            "&Bottom", self,
            shortcut=QtGui.QKeySequence.MoveToEndOfDocument,
            triggered=self._imageViewer.scrollToBottom)

        self.scrollToBeginAct = QtGui.QAction(
            get_icon('wrapping_in_front_left.png'),
            "&Left Edge", self,
            shortcut=QtGui.QKeySequence.MoveToStartOfLine,
            triggered=self._imageViewer.scrollToBegin)

        self.scrollToEndAct = QtGui.QAction(
            get_icon('wrapping_in_front_right.png'),
            "&Right Edge", self,
            shortcut=QtGui.QKeySequence.MoveToEndOfLine,
            triggered=self._imageViewer.scrollToEnd)

        self.centerView = QtGui.QAction(
            get_icon('wrapping_in_front.png'),
            "&Center", self,
            shortcut="5",
            triggered=self._imageViewer.centerView)

        #zoom actions
        self.zoomInAct = QtGui.QAction(
            get_icon('zoom_in.png'),
            "Zoo&m In (25%)", self,
            shortcut=QtGui.QKeySequence.ZoomIn,
            triggered=self._imageViewer.zoomIn)

        self.zoomOutAct = QtGui.QAction(
            get_icon('zoom_out.png'),
            "Zoom &Out (25%)", self,
            shortcut=QtGui.QKeySequence.ZoomOut,
            triggered=self._imageViewer.zoomOut)

        self.actualSizeAct = QtGui.QAction(
            get_icon('zoom_actual_equal.png'),
            "Actual &Size", self,
            shortcut="/",
            triggered=self._imageViewer.actualSize)

        self.fitToWindowAct = QtGui.QAction(
            get_icon('zoom_fit.png'),
            "Fit &Image", self,
            shortcut="*",
            triggered=self._imageViewer.fitToWindow)

        self.fitWidthAct = QtGui.QAction(
            get_icon('zoom_width.png'),
            "Fit &Width", self,
            shortcut="Alt+Right",
            triggered=self._imageViewer.fitWidth)

        self.fitHeightAct = QtGui.QAction(
            get_icon('zoom_height.png'),
            "Fit &Height", self,
            shortcut="Alt+Down",
            triggered=self._imageViewer.fitHeight)

        self.zoomToAct = QtGui.QAction(
            get_icon('zoom.png'),
            "&Zoom To...", self,
            shortcut="Z")

    def createMenus(self):
        """Create the menus."""

        #Create File Menu
        self.fileMenu = QtGui.QMenu("&File")
        self.fileMenu.addAction(self.exitAct)

        #Create Scroll Menu
        self.scrollMenu = QtGui.QMenu("&Scroll", self)
        self.scrollMenu.addAction(self.scrollToTopAct)
        self.scrollMenu.addAction(self.scrollToBottomAct)
        self.scrollMenu.addAction(self.scrollToBeginAct)
        self.scrollMenu.addAction(self.scrollToEndAct)
        self.scrollMenu.addAction(self.centerView)

        #Create Zoom Menu
        self.zoomMenu = QtGui.QMenu("&Zoom", self)
        self.zoomMenu.addAction(self.zoomInAct)
        self.zoomMenu.addAction(self.zoomOutAct)
        self.zoomMenu.addSeparator()
        self.zoomMenu.addAction(self.actualSizeAct)
        self.zoomMenu.addAction(self.fitToWindowAct)
        self.zoomMenu.addAction(self.fitWidthAct)
        self.zoomMenu.addAction(self.fitHeightAct)
        #self.zoomMenu.addSeparator()
        #self.zoomMenu.addAction(self.zoomToAct)

        #Add menus to menubar
        menubar = self.menuBar()
        menubar.addMenu(self.fileMenu)
        menubar.addMenu(self.scrollMenu)
        menubar.addMenu(self.zoomMenu)

    def createToolbars(self):
        """Create the toolbars."""
        self.tbar_view = self.addToolBar('View')

        self.tbar_view.addAction(self.scrollToTopAct)
        self.tbar_view.addAction(self.scrollToBottomAct)
        self.tbar_view.addAction(self.scrollToBeginAct)
        self.tbar_view.addAction(self.scrollToEndAct)
        self.tbar_view.addAction(self.centerView)
        self.tbar_view.addSeparator()
        self.tbar_view.addAction(self.zoomInAct)
        self.tbar_view.addAction(self.zoomOutAct)
        self.tbar_view.addSeparator()
        self.tbar_view.addAction(self.actualSizeAct)
        self.tbar_view.addAction(self.fitToWindowAct)
        self.tbar_view.addAction(self.fitWidthAct)
        self.tbar_view.addAction(self.fitHeightAct)


class SettingsMixin(object):
    """
    Mixin used by implamentations to manage settings.
    """
    def __init__(self):
        self.settings = QtCore.QSettings()
        self.settings.beginGroup(str(self.__class__))
        super(SettingsMixin, self).__init__()

    def writeSettings(self):
        """Write application settings."""

        self.settings.setValue('pos', self.pos())
        self.settings.setValue('size', self.size())
        self.settings.setValue('windowgeometry', self.saveGeometry())
        self.settings.setValue('windowstate', self.saveState())

    def readSettings(self):
        """Read application settings."""
        pos = self.settings.value('pos', QtCore.QPoint(200, 200))
        size = self.settings.value('size', QtCore.QSize(400, 400))
        self.move(pos)
        self.resize(size)
        if self.settings.contains('windowgeometry'):
            self.restoreGeometry(self.settings.value('windowgeometry'))
        if self.settings.contains('windowstate'):
            self.restoreState(self.settings.value('windowstate'))

    @staticmethod
    def appSettings(appname=None, company=None, domain=None):
        """
        To be called from main to prepare settings.
        """
        if appname is None:
            appname = os.path.splitext(sys.argv[0])[0]
        if company is None:
            company = appname
        if domain is None:
            domain = '%s.%s.org' % (company, appname)

        app = QtCore.QCoreApplication.instance()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
        app.setOrganizationName(company)
        app.setOrganizationDomain(domain)
        app.setApplicationName(appname)


class MainWindow(QtGui.QMainWindow, ActionsMixin, SettingsMixin, NavigMixin):
    """Main window for the application"""

    def __init__(self, pixmap=None, nav_list=None):
        """:param QPixmap pixmap: |QPixmap| to display"""
        #super(MainWindow, self).__init__()
        QtGui.QMainWindow.__init__(self)
        ActionsMixin.__init__(self)
        SettingsMixin.__init__(self)
        NavigMixin.__init__(self)

        self._imageViewer = ImageViewer(pixmap, "View 1")
        self.setCentralWidget(self._imageViewer)

        self._imageViewer.sceneChanged.connect(self.sceneChanged)
        self._imageViewer.transformChanged.connect(self.transformChanged)
        self._imageViewer.scrollChanged.connect(self.scrollChanged)

        #self._imageViewer.enableScrollBars(True)
        self._imageViewer.enableHandDrag(True)

        self.createActions()
        self.createMenus()
        self.createToolbars()

        if nav_list:
            self.tbar_nav = self.addToolBar('Navigate')
            self.img_set = nav_list
            self.createNavigActions(toolbar=self.tbar_nav)
            self.show_crt_nav_img()

        self.eventCounter = 0

    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(list)
    def sceneChanged(self, rects):
        """Triggered when the underlying graphics scene has changed.
        :param list rects: scene rectangles that indicate the area that
                           has been changed."""
        # r = self._imageViewer._sceneRect
        #print("%3d Scene changed = (%.2f,%.2f,%.2f,%.2f %.2fx%.2f)" %
        #      (self.eventCounter, r.left(), r.top(), r.right(), r.bottom(),
        #       r.width(), r.height()))
        self.eventCounter += 1

    @QtCore.pyqtSlot()
    def transformChanged(self):
        """Triggered when the underlying view has been scaled, translated,
        or rotated.
        In practice, only scaling occurs."""
        #print("%3d transform changed = " % self.eventCounter)
        #self._imageViewer.dumpTransform()
        self.eventCounter += 1

    @QtCore.pyqtSlot()
    def scrollChanged(self):
        """Triggered when the views scrollbars have changed."""
        #        hbar = self._imageViewer._horizontalScrollBar
        #        hpos = hbar.value()
        #        hmin = hbar.minimum()
        #        hmax = hbar.maximum()
        #
        #        vbar = self._imageViewer._verticalScrollBar
        #        vpos = vbar.value()
        #        vmin = vbar.minimum()
        #        vmax = vbar.maximum()

        #print("%3d scroll changed h=(%d,%d,%d) v=(%d,%d,%d)" %
        #      (self.eventCounter, hpos,hmin,hmax, vpos,vmin,vmax))
        self.eventCounter += 1

    # ------------------------------------------------------------------

    #overriden events

    def keyPressEvent(self, keyEvent):
        """Overrides to enable panning while dragging.
        :param QKeyEvent keyEvent: instance of |QKeyEvent|"""
        assert isinstance(keyEvent, QtGui.QKeyEvent)
        #        if keyEvent.key() == QtCore.Qt.Key_Space:
        #            if (not keyEvent.isAutoRepeat() and
        #                not self._imageViewer.handDragging):
        #                self._imageViewer.enableHandDrag(True)
        #            keyEvent.accept()
        #        else:
        keyEvent.ignore()
        super(MainWindow, self).keyPressEvent(keyEvent)

    def keyReleaseEvent(self, keyEvent):
        """Overrides to disable panning while dragging.
        :param QKeyEvent keyEvent: instance of |QKeyEvent|"""
        assert isinstance(keyEvent, QtGui.QKeyEvent)
        #        if keyEvent.key() == QtCore.Qt.Key_Space:
        #            if not keyEvent.isAutoRepeat() and self._imageViewer.handDragging:
        #                self._imageViewer.enableHandDrag(False)
        #            keyEvent.accept()
        #        else:
        keyEvent.ignore()
        super(MainWindow, self).keyReleaseEvent(keyEvent)

    def closeEvent(self, event):
        """Overrides close event to save application settings.
        :param QEvent event: instance of |QEvent|"""
        self.writeSettings()
        event.accept()


def main():
    """
    Test app to run from command line.
    **Usage**::
      python26 imageviewer.py imagefilename
    """

    COMPANY = "pyl2extra"
    DOMAIN = "pyl2extra.org"
    APPNAME = "Image Viewer"

    app = QtGui.QApplication(sys.argv)
    fname = app.arguments()[-1]
    if not os.path.exists(fname):
        print('File "%s" does not exist.' % fname)
        pixmap = None
    else:
        pixmap = ImageViewer.pixmap_from_file(fname)

    SettingsMixin.appSettings(DOMAIN, COMPANY, APPNAME)
    #app.setWindowIcon(QtGui.QIcon(":/icon.png"))

    mainWin = MainWindow(pixmap)
    mainWin.setWindowTitle(APPNAME)
    mainWin.readSettings()
    mainWin.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
