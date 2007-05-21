############################################################################
##
## Copyright (C) 2006-2007 University of Utah. All rights reserved.
##
## This file is part of VisTrails.
##
## This file may be used under the terms of the GNU General Public
## License version 2.0 as published by the Free Software Foundation
## and appearing in the file LICENSE.GPL included in the packaging of
## this file.  Please review the following to ensure GNU General Public
## Licensing requirements will be met:
## http://www.opensource.org/licenses/gpl-license.php
##
## If you are unsure which license is appropriate for your use (for
## instance, you are interested in developing a commercial derivative
## of VisTrails), please contact us at vistrails@sci.utah.edu.
##
## This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
## WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
##
############################################################################
""" This module is responsible for dealing with macros inside a vistrail.
These are the classes defined here:
 - Macro
 - ExternalModuleNotSet
 - InvalidActionFound

"""
from db.domain import DBMacro
import copy
from core.utils import unimplemented, abstract, VistrailsInternalError, enum
from core.vistrail.connection import Connection
from core.data_structures.point import Point
from core.vistrail.action import Action

###############################################################################

class Macro(DBMacro):
    """ Class that represents a Vistrail Macro.
    
    Member Description
    ------------------
    
        self.modules is a dictionary of modules referenced by the macro
        and it is in the form:
                macroModuleId -> externalModuleId
                
             Notice that when the macro contains an action addModule, 
             macroModuleId and externalModuleId will be the same

        self.connections is a dictionary of
                macroConnectionId -> externalConnectionId

        self.extModules is a list of ModuleIds that point to external 
        pre-existent modules in the pipeline (outside the macro).

        self.extConnections is a list of ConnectionIds that point to external
        pre-existent connections in the pipeline (outside the macro).
        
    """
    __fields__ = ['id','startTime', 'name', 'description', 'actions',
                  'modules', 'extModules', 'connections',
                  'extConnections','pipeline', 'vistrail', 'actionList',
                  'startPos']
    def __init__(self, vis):
        """ Constructor.

        Keyword Arguments:

        - vis : 'Vistrail'
              Vistrail object
          
        """
	DBMacro.__init__(self)
        self.vistrail = vis
        self.name = ''
        self.description = ''
        self.id = -1
        self.actionList = []
        self.clear()
        self.pipeline = None

    def _set_endTime(self,time):
        """ _set_endTime(time:int) -> None
        Set macro end time. Time is a vistrail timestep.
        You shouldn't use this method. Use endTime property instead.

        """
        self.__endTime = time
        if time != 0:
            self.loadActions()

    def _get_endTime(self):
        """ _get_endTime() -> int
        Return macro end time. Time is a vistrail timestep.
        You shouldn't use this method. Use endTime property instead.

        """
        return self.__endTime

    endTime = property(_get_endTime, _set_endTime)

    def _get_id(self):
	return _Macro._get_id(self)
    def _set_id(self, id):
	_Macro._set_id(self, id)
    id = property(_get_id, _set_id)

    def _get_name(self):
	return _Macro._get_name(self)
    def _set_name(self, name):
	_Macro._set_name(self, name)
    name = property(_get_name, _set_name)

    def _get_description(self):
	return _Macro._get_descrptn(self)
    def _set_description(self, description):
	return _Macro._set_descrptn(self, description)
    description = property(_get_description, _set_description)

    @staticmethod
    def convert(_macro):
	_macro.__class__ = Macro

    @staticmethod
    def convertToDB(macro):
	macro.__class__ = _Macro

    def clear(self):
        """ clear() -> None 
        Clear all macro internal lists and dictionaries.

        """
        self.actions = {}
        self.modules = {}
        self.extModules = []
        self.connections = {}
        self.extConnections = []
        
    def loadActions(self):
        """ loadActions() -> None 
        it will look through all the action ids in self.actionList and build
        the macro action chain.

        """
        if self.vistrail:
            self.clear()
            if len(self.actionList) > 0:
                #the last action on actionList will contain the endTime
                endTime = self.actionList[-1]
                self.pipeline = self.getPipeline(endTime)
                self.buildActionChain()
        else:
            raise VistrailsInternalError("vistrail object is null")

    def getPipeline(self, timestep):
        """ getPipeline(timestep:int) -> core.vistrail.Pipeline 
        Query the current self.vistrail for the pipeline with id timestep.

        """
        return self.vistrail.getPipeline(timestep)
        
    def addAction(self, timestep):
        """ addAction(timestep:int) -> None 
        Adds an action id to the macro during recording time.
        
        """
        self.actionList.append(timestep)
           
    def buildActionChain(self):
        """ buildActionChain() -> None 
        Builds the action chain composing the macro according to 
        self.actionList.
       
        """
        st = self.actionList[0]
        et = self.actionList[-1]
        list = []
        self.actions = {}
        for a in self.actionList:
            self.actions[a] = MacroAction.create(self.vistrail.actionMap[a],
                                                 self)

    def applyMacro(self, viscontrol, pipeline, force=True):
        """ applyMacro(viscontrol, pipeline,force=True) -> None
        Applies this macro to pipeline. When applying a macro, the
        external modules must have been mapped or we get an error. 
        Every action will have its dependencies checked. 
        If force == True, if the dependencies of an action
        are not fulfilled the macro will still be applied but not that action.
        The actions to be performed will also be added to the vistrail history,
        through the vistrail controller object.

        Keyword Arguments
        ----------

        - viscontrol : 'VistrailController'
        
        - pipeline : 'VisPipeline'
        
        - force : 'boolean'
          
        """
        if self.checkExternals():
            if self.checkDependencies(force):
                if len(self.extModules) > 0:
                    for i in range(len(self.extModules)):
                        m_id = self.modules[self.extModules[i]]
                        if m_id:
                            m = pipeline.getModuleById(m_id)
                            self.startPos = Point(m.center.x, m.center.y)
                            break
                for a in self.actionList:
                    action = self.actions[a]
                    if action.enabled: 
                        action.generate(pipeline)
                        if action.endAction.type == "DeleteModule":
                            #the connections to that module will be deleted by
                            #VisController when this action is performed
                            for id in action.endAction.ids:
                                viscontrol.deleteModule(id)
                        else:
                            mycopy = copy.copy
                            viscontrol.performAction(mycopy(action.endAction))
            else:
                raise InvalidActionFound() 
        else:
            raise ExternalModuleNotSet()

    def checkExternals(self):
        """ checkExternals() -> boolean 
        Return True if all external modules in the macro are set. 
        
        """
        result = True
        for ext in self.extModules:
            if self.modules[ext] == None:
                for a in self.actions.values():
                    if a.moduleReferenced(ext) and a.enabled:
                        result = False
                        break
        for ext in self.extConnections:
            if self.connections[ext] == None:
                for a in self.actions.values():
                    if a.connReferenced(ext) and a.enabled:
                        if not self.moduleToBeDeletedIn(ext):
                            result = False
                            break
        return result
    
    def checkExternalModule(self,id):
        """ checkExternalModule(id) -> boolean 
        Return True if external module id is set.
        
        """
        if self.modules[id] == None:
            return False
        else:
            return True

    def checkExternalConnection(self,id):
        """ checkExternalConnection(id) -> boolean
        Return True if external module of that connection is valid

        """
        result = True
        if self.connections[id] == None:
            if not self.moduleToBeDeletedIn(id):
                result = False
        return result

    def getAction(self, type, id, source):
        """ getAction(type, id, source) -> int 
        Returns the first enabled action id (timestep) of type 'type' 
        starting from source and following the action chain that 
        references 'id'.
        This is used when source wants to know if its dependencies are enabled
        
        Keyword Arguments
        
        - type : 'str'
        type of action to be found
        - id : 'list' or 'int'
        - source : MacroAction
    
        """
        a = source.baseAction.parent
        while a in self.actionList:
            action = self.actions[a]
            if action.enabled:
                if action.baseAction.type == type:
                    ids = action.getId()
                    for i in ids:
                        if i == id:
                            return a
            a = action.baseAction.parent
        return -1 #failure

    def checkDependencies(self, force=True):
        """  checkDependencies(force=True) -> boolean
        Checks for all enabled actions if all required dependent actions
        are also enabled.
        If force is True, it will disable invalid actions. 
        
        Dependencies are considered in the following way: 
        For each action a, if a is a:
            * AddModule -> no dependecies required
            * ChangeParameter -> AddModule
            * DeleteModule -> AddModule
            * DeleteFunction -> ChangeParameter -> AddModule
            * MoveModule -> AddModule
            * AddConnection -> AddModule, AddModule
            * DeleteConnection -> AddConnection -> AddModule, AddModule 
        """
        verified = {}
        for t in self.actionList:
            action = self.actions[t]
            if action.enabled:
                if not verified.has_key(t):
                    res = action.checkDependencies(verified,force)
                    if not res:
                        return False
        return True

    def moduleToBeDeletedIn(self,id):
        """ moduleToBeDeletedIn(id) -> boolean  
        Returns True if the connection identified by id connects a module that
        will be deleted by the macro

        Keyword Argumennts
        ----------

        - id : int 
          connection id
          
        """
        result = False

        a = self.actions[self.startTime]
        t = a.baseAction.parent
        pipeline = self.getPipeline(t)

        if pipeline:
            c = pipeline.getConnectionById(id)
            m1 = c.sourceId
            m2 = c.destinationId
            for a in self.actions.values():
                if a.endAction.type == 'DeleteModule' and a.enabled:
                    if a.sourceId in [m1,m2]:
                        if self.checkExternalModule(a.sourceId):
                            result = True
                            break
        return result
            

    def serialize(self, dom, element):
        """ serialize(dom, element) -> None
        Adds the serialized version of itself to an XML parent element

        Keyword Arguments: 

        - dom : 'XML Document'

        - element : 'XML Element'
        
        """
        m = dom.createElement('macro')
        m.setAttribute('id',str(self.id))
        m.setAttribute('name',str(self.name))
        m.setAttribute('desc', str(self.description))
        for a in self.actionList:
            child = dom.createElement('action')
            child.setAttribute('time',str(a))
            m.appendChild(child)
    
        element.appendChild(m)

    @staticmethod
    def createFromXML(vis, element):
        """ createFromXML(vis, element) -> Macro 
        Static method that creates a Macro object given the corresponding 
        XML element. 
        
        Keyword Arguments:

        - vis : 'Vistrail'
        - element : 'XML Element'

        """
        newMacro = Macro(vis)
        newMacro.id = int(element.getAttribute('id'))
        newMacro.name = str(element.getAttribute('name'))
        newMacro.description = str(element.getAttribute('desc'))

        newMacro.actionList = [int(m.getAttribute('time'))
                               for m in named_elements(element, 'action')]
        newMacro.loadActions()
        return newMacro
    
    def __str__(self):
        """ Returns a string representation of a Macro object.
    
        Returns
        -------

        - 'str'
        
        """
        s = '<macro name=%s desc=%s id=%s>\n' % \
               (self.name, self.description, self.id)
        tmp = ''
        for a in self.actionList:
            tmp += tmp + '<action time=%s/>\n' % a
        s += s + tmp
        s += '</macro>'
            
###############################################################################
### Exception Classes
        
class ExternalModuleNotSet(Exception):
    def __str__(self):
        return "You need to set all the external references inside the macro " \
            "before applying it."
    pass

class InvalidActionFound(Exception):
    def __str__(self):
        return "This macro has actions that depend on actions that won't be " \
            " played. \
            Please disable these actions or include the other required actions."
    pass
        
#############################################################################

import unittest

class TestMacro(unittest.TestCase):
    
    def test1(self):
        # FIXME: This test is testing nothing. Put some macros in dummy.xml
        import core.vistrail
        import core.xml_parser
        import core.system
        parser = core.xml_parser.XMLParser()
        parser.openVistrail(core.system.vistrails_root_directory() +
                            'tests/resources/dummy.xml')
        v = parser.getVistrail()
        parser.closeVistrail()
        import tempfile
        import os
        (fd, name) = tempfile.mkstemp()
        os.close(fd)
        v.serialize(name)
        os.unlink(name)

    def testDependencies(self):
        # FIXME: This test is testing nothing. Put some macros in dummy.xml
        import core.vistrail
        import core.xml_parser
        import core.system
        parser = core.xml_parser.XMLParser()
        parser.openVistrail(core.system.vistrails_root_directory() +
                            'tests/resources/dummy.xml')
        v = parser.getVistrail()
        parser.closeVistrail()

if __name__ == '__main__':
    unittest.main()
