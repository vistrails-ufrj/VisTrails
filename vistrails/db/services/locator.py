###############################################################################
##
## Copyright (C) 2014-2015, New York University.
## Copyright (C) 2011-2014, NYU-Poly.
## Copyright (C) 2006-2011, University of Utah.
## All rights reserved.
## Contact: contact@vistrails.org
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
##  - Neither the name of the New York University nor the names of its
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
from __future__ import division

import cgi
from datetime import datetime, date
import hashlib
import locale
import os.path
import re
import sys
import urllib
import urlparse
import uuid

import vistrails.core.system
from vistrails.db.services import io
from vistrails.db.services.bundle import Bundle, VistrailBundle, BundleObj
from vistrails.db.domain import DBVistrail, DBWorkflow
from vistrails.db import VistrailsDBException
from vistrails.core import debug
from vistrails.core.system import get_elementtree_library, systemType, \
    time_strptime

ElementTree = get_elementtree_library()

def process_workflow_tag(kwargs, value):
    try:
        kwargs["version_node"] = int(value)
    except ValueError:
        kwargs["version_tag"] = value

def add_properties(cls):
    """class decorator to programmatically add properties"""
    for name in cls.KWARG_PROPS:
        cls.add_property(name)
    return cls

_drive_regex = re.compile(r"/*([a-zA-Z]:/.+)$")
def pathname2url(path):
    """ Takes an absolute filename and turns it into a file:// URL.

    While urllib.pathname2url seems like a good idea, it doesn't appear
    to do anything sensible in practice on Windows.
    """
    if path.startswith('file://'):
        path = urllib.unquote(path[7:])
    if systemType in ('Windows', 'Microsoft'):
        path = path.replace('\\', '/')
        match = _drive_regex.match(path)
        if match is not None:
            path = '/%s' % match.group(1)
    path = urllib.quote(path, safe='/:')
    return path


def url2pathname(urlpath):
    """ Takes a file:// URL and turns it into a filename.
    """
    path = urllib.url2pathname(urlpath)
    if systemType in ('Windows', 'Microsoft'):
        path = path.replace('/', '\\')
        path = path.lstrip('\\')
    return path

@add_properties
class BaseLocator(object):
    """KWARG_PROPS defines 2-tuples (<python property>, <url tag>).  From
    this, properties are automatically created and the parse_args and
    generate_args methods use this info to do the necessary
    conversions.  If you want to have a new property for a locator,
    define it here.

    """
    
    KWARG_PROPS = dict([("obj_type", "type"),
                        ("obj_id", "id"),
                        ("version_node", "workflow"),
                        ("version_tag", "workflow"),
                        ("workflow_exec", "workflow_exec"),
                        ("parameterExploration", "parameterExploration"),
                        ("mashuptrail", "mashuptrail"),
                        ("mashupVersion", "mashupVersion"),
                        ("mashup", "mashupVersion")])

    SPECIAL_TAGS = {"workflow": process_workflow_tag}

    @classmethod
    def add_property(cls, attr):
        def setter(self, v):
            self.kwargs[attr] = v
        def getter(self):
            return self.kwargs.get(attr, None)
        setattr(cls, attr, property(getter, setter))

    @classmethod
    def get_kwarg_props(cls):
        return cls.KWARG_PROPS

    @classmethod
    def get_special_tags(cls):
        return cls.SPECIAL_TAGS
    
    @classmethod
    def parse_args(cls, arg_str):
        kwargs = {}
        parsed_dict = cgi.parse_qs(arg_str)
        special_tags = cls.get_special_tags()
        for (prop, url_tag) in cls.get_kwarg_props().iteritems():
            if url_tag in parsed_dict:
                if url_tag in special_tags:
                    # special handling
                    special_tags[url_tag](kwargs, parsed_dict[url_tag][0])
                elif prop not in kwargs:
                    # don't overwrite if we already set the prop
                    kwargs[prop] = parsed_dict[url_tag][0]
        return kwargs

    @classmethod
    def generate_args(cls, kwargs):
        generate_dict = {}        
        for (prop, url_tag) in cls.get_kwarg_props().iteritems():
            if prop in kwargs and kwargs[prop]:
                generate_dict[url_tag] = kwargs[prop]
        return urllib.urlencode(generate_dict)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self):
        raise NotImplementedError("load is not implemented")

    def save(self, obj, do_copy=True, version=None):
        """Saves an object in the given place.
        
        """
        raise NotImplementedError("save is not implemented")

    def save_as(self, obj, version=None):
        return self.save(obj, True, version) # calls save by default

    def close(self):
        """Closes locator.
        
        """
        pass

    def is_valid(self):
        """Returns true if locator refers to a valid object.
        
        """
        raise NotImplementedError("is_valid is not implemented")
        
    def get_temporary(self):
        return None

    def has_temporaries(self):
        return self.get_temporary() is not None

    def to_xml(self, node=None):
        """Serialize locator, optionally into existing ElementTree node.

        """
        raise NotImplementedError("to_xml is not implemented")
    
    @staticmethod
    def from_xml(node):
        """Unserialize locator given ElementTree node.

        """
        raise NotImplementedError("from_xml is not implemented")

    @staticmethod
    def convert_filename_to_url(filename):
        """ Converts a local filename to a file:// URL.

        All file:// URLs are absolute, so abspath() will be used on the
        argument.
        """
        exts = ["vt", "xml"]
        q_mark = False
        query_str_idx = None
        for match in re.finditer("\.(%s)(\??)" % "|".join(exts), filename):
            if match.group(2):
                if q_mark:
                    raise VistrailsDBException('Ambiguous URI with '
                                               'multiple question '
                                               'marks: "%s"' % filename)
                else:
                    q_mark = True
                    query_str_idx = match.end()
        if q_mark:
            args_str = filename[query_str_idx-1:]
            filename = filename[:query_str_idx-1]
        else:
            args_str = ""

        return 'file://%s%s' % (pathname2url(os.path.abspath(filename)),
                                urllib.quote(args_str, safe='/?=&'))

    @staticmethod
    def from_url(url):
        """Assumes a valid URL if the scheme is specified.  For example,
        'file:///C:/My%20Documents/test.vt'.  If only a filename is
        specified, it converts the filename to a URL.

        """
        if '://' in url:
            scheme = url.split('://', 1)[0]
        elif url.startswith('untitled:'):
            scheme = 'untitled'
        else:
            scheme = 'file'
            url = BaseLocator.convert_filename_to_url(url)
        if scheme == 'untitled':
            return UntitledLocator.from_url(url)
        elif scheme == 'db':
            return DBLocator.from_url(url)
        elif scheme == 'file':
            old_uses_query = urlparse.uses_query
            urlparse.uses_query = urlparse.uses_query + ['file']
            scheme, host, path, query, fragment = urlparse.urlsplit(str(url))
            urlparse.uses_query = old_uses_query
            path = url2pathname(path)
            if path.endswith(".vt"):
                return ZIPFileLocator.from_url(url)
            elif path.endswith(".xml"):
                return XMLFileLocator.from_url(url)
            else:
                return DirectoryLocator.from_url(url)
        return None

    def _get_name(self):
        return None # Returns a name that will be displayed for the object
    name = property(_get_name)

    def _get_short_filename(self):
        """ Returns a short name that can be used to derive other filenames
        """
        return None
    short_filename = property(_get_short_filename)

    def _get_short_name(self):
        """ Returns a short name that can be used for display
        """
        return None
    short_name = property(_get_short_name)
      
    def _get_version(self):
        if self.version_tag is not None:
            return self.version_tag
        return self.version_node
    version = property(_get_version)
  
    ###########################################################################
    # Operators

    def __eq__(self, other):
        pass # Implement equality

    def __ne__(self, other):
        pass # Implement nonequality

    def __hash__(self):
        return hash(self.name)
    
    def is_untitled(self):
        return False

class SaveTemporariesMixin(object):
    """A mixin class that saves temporary copies of a file.  It requires
    self.name to exist for proper functioning.

    """

    @staticmethod
    def get_autosave_dir():
        dot_vistrails = vistrails.core.system.current_dot_vistrails()
        auto_save_dir = os.path.join(dot_vistrails, "autosave")
        if not os.path.exists(auto_save_dir):
            # !!! we assume dot_vistrails exists !!!
            os.mkdir(auto_save_dir)
        if not os.path.isdir(auto_save_dir):
            raise VistrailsDBException('Auto-save path "%s" is not a '
                                       'directory' % auto_save_dir)
        return auto_save_dir

    def get_temp_basename(self):
        return self.name

    def save_temporary(self, obj):
        fname = self._find_latest_temporary()
        new_temp_fname = self._next_temporary(fname)
        io.save_to_xml(obj, new_temp_fname)

    def clean_temporaries(self):
        """_remove_temporaries() -> None

        Erases all temporary files.

        """
        def remove_it(fname):
            os.unlink(fname)
        self._iter_temporaries(remove_it)

    def encode_name(self, filename):
        """encode_name(filename) -> str
        Encodes a file path using urllib.quoteplus

        """
        name = urllib.quote_plus(filename) + '_tmp_'
        return os.path.join(self.get_autosave_dir(), name)

    def _iter_temporaries(self, f):
        """_iter_temporaries(f): calls f with each temporary file name, in
        sequence.

        """
        latest = None
        current = 0
        while True:
            fname = self.encode_name(self.get_temp_basename()) + str(current)
            if os.path.isfile(fname):
                f(fname)
                current += 1
            else:
                break

    def _find_latest_temporary(self):
        """_find_latest_temporary(): String or None.

        Returns the latest temporary file saved, if it exists. Returns
        None otherwise.
        
        """
        latest = [None]
        def set_it(fname):
            latest[0] = fname
        self._iter_temporaries(set_it)
        return latest[0]
        
    def _next_temporary(self, temporary):
        """_find_latest_temporary(string or None): String

        Returns the next suitable temporary file given the current
        latest one.

        """
        if temporary is None:
            return self.encode_name(self.get_temp_basename()) + '0'
        else:
            split = temporary.rfind('_')+1
            base = temporary[:split]
            number = int(temporary[split:])
            return base + str(number+1)

class UntitledLocator(BaseLocator, SaveTemporariesMixin):
    UNTITLED_NAME = "Untitled"
    UNTITLED_PREFIX = UNTITLED_NAME + "_"

    def __init__(self, my_uuid=None, **kwargs):
        if my_uuid is not None:
            self._uuid = my_uuid
        else:
            self._uuid = uuid.uuid4()
        self.kwargs = kwargs

    def load(self, type):
        fname = self.get_temporary()
        if fname:
            obj = io.open_from_xml(fname, type)
        else:
            obj = DBVistrail()
        obj.locator = self
        return obj

    def is_valid(self):
        return False

    def get_temp_basename(self):
        return UntitledLocator.UNTITLED_PREFIX + self._uuid.hex

    def get_temporary(self):
        return self._find_latest_temporary()

    def _get_name(self):
        return UntitledLocator.UNTITLED_NAME
    name = property(_get_name)

    def _get_short_filename(self):
        return self._get_name()
    short_filename = property(_get_short_filename)

    def _get_short_name(self):
        return self._get_name().decode('ascii')
    short_name = property(_get_short_name)

    @classmethod
    def from_url(cls, url):
        if not url.startswith('untitled:'):
            raise VistrailsDBException("URL does not start with untitled:")

        rest = url[9:]
        my_uuid = None
        if len(rest) >= 32:
            try:
                my_uuid = uuid.UUID(rest[:32])
                rest = rest[32:]
            except ValueError:
                pass

        if not rest:
            kwargs = dict()
        elif rest[0] == '?':
            kwargs = cls.parse_args(rest[1:])
        else:
            raise ValueError
        return cls(my_uuid, **kwargs)

    def to_url(self):
        args_str = self.generate_args(self.kwargs)
        url_tuple = ('untitled', '', self._uuid.hex, args_str, '')
        return urlparse.urlunsplit(url_tuple)

    def __hash__(self):
        return self._uuid.int

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return (self._uuid == other._uuid)

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_untitled(self):
        return True
    
    @classmethod
    def all_untitled_temporaries(cls):
        autosave_dir = SaveTemporariesMixin.get_autosave_dir()
        fnames = []
        for fname in os.listdir(autosave_dir):
            if fname.startswith(cls.UNTITLED_PREFIX) and \
               os.path.isfile(os.path.join(autosave_dir, fname)):
                fnames.append(fname)
        locators = {}
        for fname in fnames:
            uuid_start = len(cls.UNTITLED_PREFIX)
            my_uuid = uuid.UUID(fname[uuid_start:uuid_start+32])
            if my_uuid not in locators:
                locators[my_uuid] = cls(my_uuid)
        return locators.values()

class DirectoryLocator(BaseLocator, SaveTemporariesMixin):
    def __init__(self, dirname, **kwargs):
        self._name = dirname
        self.kwargs = kwargs
    
    def load(self, type):
        raise Exception("Need to implement!")

    def save(self, obj, do_copy=True, version=None):
        raise Exception("Need to implement!")

    def is_valid(self):
        return os.path.isdir(self._name)

    def get_temporary(self):
        return self._find_latest_temporary()

    def _get_name(self):
        return str(self._name)
    name = property(_get_name)

    def _get_short_filename(self):
        return os.path.basename(self._name)
    short_filename = property(_get_short_filename)

    def _get_short_name(self):
        name = self._get_short_filename()
        enc = sys.getfilesystemencoding() or locale.getpreferredencoding()
        return name.decode(enc)
    short_name = property(_get_short_name)

    @classmethod
    def from_url(cls, url):
        if '://' in url:
            scheme, path = url.split('://', 1)
            if scheme != 'file':
                raise ValueError
        else:
            url = BaseLocator.convert_filename_to_url(url)

        old_uses_query = urlparse.uses_query
        urlparse.uses_query = urlparse.uses_query + ['file']
        scheme, host, path, args_str, fragment = urlparse.urlsplit(url)
        urlparse.uses_query = old_uses_query
        # De-urlencode pathname
        path = url2pathname(str(path))
        kwargs = cls.parse_args(args_str)
        return cls(os.path.abspath(path), **kwargs)

    def to_url(self):
        args_str = self.generate_args(self.kwargs)
        url_tuple = ('file', '',
                     pathname2url(os.path.abspath(self._name)),
                     args_str, '')
        return urlparse.urlunsplit(url_tuple)

class XMLFileLocator(BaseLocator, SaveTemporariesMixin):
    def __init__(self, filename, **kwargs):
        self._name = filename
        self.kwargs = kwargs

    def load(self, type):
        fname = self.get_temporary()
        if fname:
            obj = io.open_from_xml(fname, type)
        else:
            obj = io.open_from_xml(self._name, type)
        obj.locator = self
        return obj

    def save(self, obj, do_copy=True, version=None):
        is_bundle = False
        if isinstance(obj, Bundle):
            is_bundle = True
            bundle = obj
            bundleobj = bundle.get_primary_obj()
            obj = bundleobj.obj

        obj = io.save_to_xml(obj, self._name, version)
        obj.locator = self
        # Only remove the temporaries if save succeeded!
        self.clean_temporaries()
        if is_bundle:
            bundleobj.obj = obj
            return bundle
        return obj

    def is_valid(self):
        return os.path.isfile(self._name)

    def get_temporary(self):
        return self._find_latest_temporary()

    def _get_name(self):
        return str(self._name)
    name = property(_get_name)

    def _get_short_filename(self):
        return os.path.splitext(os.path.basename(self._name))[0]
    short_filename = property(_get_short_filename)

    def _get_short_name(self):
        name = self._get_short_filename()
        enc = sys.getfilesystemencoding() or locale.getpreferredencoding()
        return name.decode(enc)
    short_name = property(_get_short_name)

    @classmethod
    def from_url(cls, url):
        if '://' in url:
            scheme, path = url.split('://', 1)
            if scheme != 'file':
                raise ValueError
        else:
            url = BaseLocator.convert_filename_to_url(url)

        old_uses_query = urlparse.uses_query
        urlparse.uses_query = urlparse.uses_query + ['file']
        scheme, host, path, args_str, fragment = urlparse.urlsplit(url)
        urlparse.uses_query = old_uses_query
        # De-urlencode pathname
        path = url2pathname(str(path))
        kwargs = cls.parse_args(args_str)

        return cls(os.path.abspath(path), **kwargs)

    def to_url(self):
        args_str = BaseLocator.generate_args(self.kwargs)
        url_tuple = ('file', '',
                     pathname2url(os.path.abspath(self._name)),
                     args_str, '')
        return urlparse.urlunsplit(url_tuple)

    #ElementTree port
    def to_xml(self, node=None):
        """to_xml(node: ElementTree.Element) -> ElementTree.Element
        Convert this object to an XML representation.
        """
        if node is None:
            node = ElementTree.Element('locator')

        node.set('type', 'file')
        childnode = ElementTree.SubElement(node,'name')
        childnode.text = self._name.decode('latin-1')
        return node

    @staticmethod
    def from_xml(node):
        """from_xml(node:ElementTree.Element) -> XMLFileLocator or None
        Parse an XML object representing a locator and returns a
        XMLFileLocator object."""
        if node.tag != 'locator':
            return None

        #read attributes
        data = node.get('type', '')
        type = str(data)
        if type == 'file':
            for child in node.getchildren():
                if child.tag == 'name':
                    filename = child.text.encode('latin-1').strip()
                    return XMLFileLocator(filename)
        return None

    def __str__(self):
        return '<%s vistrail_name="%s" />' % (self.__class__.__name__, self._name)

    ###########################################################################
    # Operators

    def __eq__(self, other):
        if not isinstance(other, XMLFileLocator):
            return False
        return self._name == other._name

    def __ne__(self, other):
        return not self.__eq__(other)

class ZIPFileLocator(XMLFileLocator):
    """Files are compressed in zip format. The temporaries are
    still in xml"""
    def __init__(self, filename, **kwargs):
        XMLFileLocator.__init__(self, filename, **kwargs)
        self.tmp_dir = None

    def load(self, type):
        fname = self.get_temporary()
        if fname:
            obj = io.open_from_xml(fname, type)
            bundle = VistrailBundle()
            bundle.add_object(BundleObj(obj))
            return bundle
        else:
            (bundle, tmp_dir) = io.open_bundle_from_zip_xml(type, self._name)
            self.tmp_dir = tmp_dir
            for obj in bundle.get_db_objs():
                obj.obj.locator = self
            return bundle

    def save(self, bundle, do_copy=True, version=None):
        if do_copy:
            # make sure we create a fresh temporary directory if we're
            # duplicating the vistrail
            tmp_dir = None
        else:
            # otherwise, use the existing temp directory if one is set
            tmp_dir = self.tmp_dir
        (bundle, tmp_dir) = io.save_bundle_to_zip_xml(bundle, self._name, tmp_dir, version)
        self.tmp_dir = tmp_dir
        for obj in bundle.get_db_objs():
            obj.obj.locator = self
        # Only remove the temporaries if save succeeded!
        self.clean_temporaries()
        return bundle

    def close(self):
        if self.tmp_dir is not None:
            io.close_zip_xml(self.tmp_dir)
            self.tmp_dir = None

    ###########################################################################
    # Operators

    def __eq__(self, other):
        if not isinstance(other, ZIPFileLocator):
            return False
        return self._name == other._name

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def parse(element):
        """ parse(element) -> ZIPFileLocator or None
        Parse an XML object representing a locator and returns a
        ZIPFileLocator object.

        """
        if str(element.getAttribute('type')) == 'file':
            for n in element.childNodes:
                if n.localName == "name":
                    filename = str(n.firstChild.nodeValue).strip(" \n\t")
                    return ZIPFileLocator(filename)
            return None
        else:
            return None
        
    #ElementTree port    
    @staticmethod
    def from_xml(node):
        """from_xml(node:ElementTree.Element) -> ZIPFileLocator or None
        Parse an XML object representing a locator and returns a
        ZIPFileLocator object."""
        if node.tag != 'locator':
            return None

        #read attributes
        data = node.get('type', '')
        type = str(data)
        if type == 'file':
            for child in node.getchildren():
                if child.tag == 'name':
                    filename = child.text.encode('latin-1').strip()
                    return ZIPFileLocator(filename)
            return None
        return None

# class URLLocator(ZIPFileLocator):
#     def load(self, type):
        
class DBLocator(BaseLocator, SaveTemporariesMixin):
    cache = {}
    cache_timestamps = {}
    connections = {}
    cache_connections = {}
        
    def __init__(self, host, port, database, user, passwd, name=None,
                 **kwargs):
        self._host = host
        self._port = int(port)
        self._db = database
        self._user = user
        self._passwd = passwd
        self._name = name
        self._hash = ''
        self.kwargs = kwargs
        self._obj_id = self.kwargs.get('obj_id', None)
        if self._obj_id is not None:
            self._obj_id = long(self._obj_id)
        self._obj_type = self.kwargs.get('obj_type', None)
        self._conn_id = self.kwargs.get('connection_id', None)
        
    def _get_host(self):
        return self._host
    host = property(_get_host)

    def _get_port(self):
        return self._port
    port = property(_get_port)

    def _get_db(self):
        return self._db
    db = property(_get_db)
    
    def _get_obj_id(self):
        return self._obj_id
    obj_id = property(_get_obj_id)

    def _get_obj_type(self):
        return self._obj_type
    obj_type = property(_get_obj_type)

    def _get_connection_id(self):
        return self._conn_id
    connection_id = property(_get_connection_id)
    
    def _get_name(self):
        return self._host + ':' + str(self._port) + ':' + self._db + ':' + \
            str(self._name)
    name = property(_get_name)

    def _get_short_filename(self):
        return str(self._name)
    short_filename = property(_get_short_filename)

    def _get_short_name(self):
        name = self._name
        if not isinstance(name, unicode):
            name = name.decode('ascii')
        return name
    short_name = property(_get_short_name)

    def hash(self):
        node = self.to_xml()
        xml_string = ElementTree.tostring(node)
        #print "hash", xml_string
        return hashlib.sha224(xml_string).hexdigest()
    
    def is_valid(self):
        if self._conn_id is not None \
                and self._conn_id in DBLocator.connections:
            return True
        try:
            self.get_connection()
        except Exception:
            return False
        return True
        
    def get_connection(self):
        if self._conn_id is not None \
                and DBLocator.connections.has_key(self._conn_id):
            connection = DBLocator.connections[self._conn_id]
            if io.ping_db_connection(connection):
                return connection
        else:
            if self._conn_id is None:
                if DBLocator.cache_connections.has_key(self._hash):
                    connection = DBLocator.cache_connections[self._hash]
                    if io.ping_db_connection(connection):
                        debug.log("Reusing cached connection")
                        return connection

                if len(DBLocator.connections.keys()) == 0:
                    self._conn_id = 1
                else:
                    self._conn_id = max(DBLocator.connections.keys()) + 1
        config = {'host': self._host,
                  'port': self._port,
                  'db': self._db,
                  'user': self._user,
                  'passwd': self._passwd}
        #print "config:", config
        connection = io.open_db_connection(config)
            
        DBLocator.connections[self._conn_id] = connection
        DBLocator.cache_connections[self._hash] = connection
        return connection

    def load(self, type, tmp_dir=None):
        self._hash = self.hash()
        #print "LLoad Big|type", type
        if DBLocator.cache.has_key(self._hash):
            save_bundle = DBLocator.cache[self._hash]
            obj = save_bundle.get_primary_obj()

            ts = self.get_db_modification_time(obj.vtType)
            #debug.log("cached time: %s, db time: %s"%(DBLocator.cache_timestamps[self._hash],ts))
            if DBLocator.cache_timestamps[self._hash] == ts:
                #debug.log("using cached vistrail")
                self._name = obj.db_name
                # If thumbnail cache was cleared, get thumbs from db
                if tmp_dir is not None:
                    for absfname in save_bundle.thumbnails:
                        if not os.path.isfile(absfname):
                            save_bundle.thumbnails = io.open_thumbnails_from_db(self.get_connection(), type, self.obj_id, tmp_dir)
                            break
                return save_bundle
        #debug.log("loading vistrail from db")
        connection = self.get_connection()
        if type == DBWorkflow.vtType:
            return io.open_from_db(connection, type, self.obj_id)
        save_bundle = io.open_bundle_from_db(type, connection, self.obj_id, tmp_dir)
        primary_obj = save_bundle.get_primary_obj()
        self._name = primary_obj.db_name
        #print "locator db name:", self._name
        for obj in save_bundle.get_db_objs():
            obj.obj.locator = self
        
        _hash = self.hash()
        DBLocator.cache[self._hash] = save_bundle.do_copy()
        DBLocator.cache_timestamps[self._hash] = primary_obj.db_last_modified
        return save_bundle

    def save(self, save_bundle, do_copy=False, version=None):
        connection = self.get_connection()
        for obj in save_bundle.get_db_objs():
            obj.obj.db_name = self._name
        save_bundle = io.save_bundle_to_db(save_bundle, connection, do_copy, version)
        primary_obj = save_bundle.get_primary_obj()
        self._obj_id = primary_obj.db_id
        if self._obj_id is not None:
            self._obj_id = long(self._obj_id)
        self._obj_type = primary_obj.vtType
        for obj in save_bundle.get_db_objs():
            obj.obj.locator = self
        #update the cache with a copy of the new bundle
        self._hash = self.hash()
        DBLocator.cache[self._hash] = save_bundle.do_copy()
        DBLocator.cache_timestamps[self._hash] = primary_obj.db_last_modified
        return save_bundle

    def get_db_modification_time(self, obj_type=None):
        if obj_type is None:
            if self.obj_type is None:
                obj_type = DBVistrail.vtType 
            else:
                obj_type = self.obj_type

        ts = io.get_db_object_modification_time(self.get_connection(),
                                                self.obj_id,
                                                obj_type)
        ts = datetime(*time_strptime(str(ts).strip(), '%Y-%m-%d %H:%M:%S')[0:6])
        return ts

    @classmethod
    def from_url(cls, url):
        format = re.compile(
                r"^"
                "([a-zA-Z0-9_-]+)://"   # scheme
                "(?:"
                    "([^:@]+)"          # user name
                    "(?:([^:@]+))?"     # password
                "@)?"
                "([^/]+)"               # net location
                "/([^?]+)"              # database name
                "(?:\?(.+))?"           # query arguments
                "$")
        match = format.match(url)
        if match is None:
            return ValueError
        else:
            scheme, user, passwd, net_loc, db_name, args_str = match.groups('')
            if ':' in net_loc:
                host, port = net_loc.rsplit(':', 1)
            else:
                host, port = net_loc, None
            db_name = urllib.unquote(str(db_name))
            kwargs = cls.parse_args(args_str)
            return cls(host, port, db_name, user, passwd, **kwargs)
    
    def to_url(self):
        # FIXME may also want to allow database type to be encoded in 
        # scheme (ie mysql://host/db, sqlite3://path/to)
        net_loc = '%s:%s' % (self._host, self._port)
        args_str = self.generate_args(self.kwargs)
        # query_str = '%s=%s' % (self._obj_type, self._obj_id)
        url_tuple = ('db', net_loc, urllib.quote(self._db, ''), args_str, '')
        return urlparse.urlunsplit(url_tuple)

    #ElementTree port
    def to_xml(self, node=None, include_name = False):
        """to_xml(node: ElementTree.Element) -> ElementTree.Element
        Convert this object to an XML representation.
        """
        if node is None:
            node = ElementTree.Element('locator')

        node.set('type', 'db')
        node.set('host', str(self._host))
        node.set('port', str(self._port))
        node.set('db', str(self._db))
        node.set('vt_id', str(self._obj_id))
        node.set('user', str(self._user))
        if include_name:
            childnode = ElementTree.SubElement(node,'name')
            childnode.text = str(self._name)
        return node

    @staticmethod
    def from_xml(node, include_name=False):
        """from_xml(node:ElementTree.Element) -> DBLocator or None
        Parse an XML object representing a locator and returns a
        DBLocator object."""
        
        def convert_from_str(value,type):
            def bool_conv(x):
                s = str(x).upper()
                if s == 'TRUE':
                    return True
                if s == 'FALSE':
                    return False

            if value is not None:
                if type == 'str':
                    return str(value)
                elif value.strip() != '':
                    if type == 'long':
                        return long(value)
                    elif type == 'float':
                        return float(value)
                    elif type == 'int':
                        return int(value)
                    elif type == 'bool':
                        return bool_conv(value)
                    elif type == 'date':
                        return date(*time_strptime(value, '%Y-%m-%d')[0:3])
                    elif type == 'datetime':
                        return datetime(*time_strptime(value, '%Y-%m-%d %H:%M:%S')[0:6])
            return None
    
        if node.tag != 'locator':
            return None

        #read attributes
        data = node.get('type', '')
        type = convert_from_str(data, 'str')
        
        if type == 'db':
            data = node.get('host', None)
            host = convert_from_str(data, 'str')
            data = node.get('port', None)
            port = convert_from_str(data,'int')
            data = node.get('db', None)
            database = convert_from_str(data,'str')
            data = node.get('vt_id')
            vt_id = convert_from_str(data, 'str')
            data = node.get('user')
            user = convert_from_str(data, 'str')
            passwd = ""
            name = None
            if include_name:
                for child in node.getchildren():
                    if child.tag == 'name':
                        name = str(child.text).strip(" \n\t")
            return DBLocator(host, port, database,
                             user, passwd, name, obj_id=vt_id, obj_type='vistrail')
        else:
            return None

    def __str__(self):
        return '<DBLocator host="%s" port="%s" database="%s" vistrail_id="%s" \
vistrail_name="%s"/>' % ( self._host, self._port, self._db,
                          self._obj_id, self._name)

    ###########################################################################
    # Operators

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return (self._host == other._host and
                self._port == other._port and
                self._db == other._db and
                self._user == other._user and
                #self._name == other._name and
                self._obj_id == other._obj_id and
                self._obj_type == other._obj_type)

    def __ne__(self, other):
        return not self.__eq__(other)


import unittest

class TestLocators(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertIsInstance'):
        def assertIsInstance(self, obj, cls, msg=None):
            assert(isinstance(obj, cls))
        def assertIsNone(self, obj):
            self.assertEqual(obj, None)

    @staticmethod
    def path2url(fname):
        path = os.path.abspath(fname)
        path = path.replace(os.sep, '/')
        if path.startswith('/'):
            path = path[1:]
        return "file:///%s" % urllib.quote(path, '/:')

    def test_convert_filename(self):
        # Test both systemTypes
        global systemType
        old_systemType = systemType
        # Don't use abspath, it would cause Linux tests to fail on Windows
        # we are using abspaths anyway
        old_abspath = os.path.abspath
        os.path.abspath = lambda x: x
        try:
            systemType = 'Linux'
            self.assertEqual(
                    BaseLocator.convert_filename_to_url(
                            '/a dir/test.vt?v=a\xE9&b'),
                    'file:///a%20dir/test.vt?v=a%E9&b')
            systemType = 'Windows'
            self.assertEqual(
                    BaseLocator.convert_filename_to_url(
                            'C:\\a dir\\test.vt?v=a\xE9&b'),
                    'file:///C:/a%20dir/test.vt?v=a%E9&b')
        finally:
            systemType = old_systemType
            os.path.abspath = old_abspath

    def test_parse_untitled(self):
        loc_str = "untitled:e78394a73b87429e952b71b858e03242?workflow=42"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, UntitledLocator)
        self.assertEqual(loc.kwargs['version_node'], 42)
        self.assertEqual(loc._uuid, 
                         uuid.UUID('e78394a73b87429e952b71b858e03242'))
        self.assertEqual(loc.to_url(), loc_str)

    def test_untitled_no_uuid(self):
        loc_str = "untitled:"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, UntitledLocator)
        # make sure it adds a uuid
        self.assertEqual(len(loc.to_url()), 41)

    def test_parse_zip_file(self):
        loc_str = self.path2url(
                "/vistrails/tmp/test_parse_zip_file \xE9 \xEA.vt")
        loc_str += "?workflow=abc"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, ZIPFileLocator)
        self.assertEqual(loc.kwargs['version_tag'], "abc")
        self.assertEqual(loc.short_filename, "test_parse_zip_file \xE9 \xEA")
        self.assertEqual(loc.to_url(), loc_str)

    def test_parse_zip_file_no_scheme(self):
        loc_str = os.path.abspath(
                "../tmp/test_parse_zip_file_no_scheme \xE9 \xEA.vt")
        loc_str += "?workflow=abc"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, ZIPFileLocator)
        self.assertEqual(loc.kwargs['version_tag'], "abc")
        self.assertEqual(loc.short_filename,
                         "test_parse_zip_file_no_scheme \xE9 \xEA")
        loc_str = loc_str.replace(os.sep, '/')
        if loc_str[0] == '/':
            loc_str = loc_str[1:]
        loc_str = "file:///%s" % urllib.quote(loc_str, '/:?=')
        self.assertEqual(loc.to_url(), loc_str)

    def test_parse_xml_file(self):
        loc_str = self.path2url(
                "/vistrails/tmp/test_parse_xml_file \xE9 \xEA.xml")
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, XMLFileLocator)
        self.assertEqual(loc.short_filename, "test_parse_xml_file \xE9 \xEA")
        self.assertEqual(loc.to_url(), loc_str)

    def test_short_names(self):
        enc = sys.getfilesystemencoding() or locale.getpreferredencoding()
        if (enc.lower() not in ('mbcs', 'utf-8', 'utf8',
                                'latin-1', 'iso-8859-1', 'iso-8859-15')):
            self.skipTest("unusual encoding on this system: %r" % enc)
        if enc.lower() in ('mbcs', 'latin-1', 'iso-8859-1', 'iso-8859-15'):
            fname = "test_short_names \xE9 \xEA"
        elif enc.lower() in ('utf8', 'utf-8'):
            fname = "test_short_names \xC3\xA9 \xC3\xAA"
        else:
            self.skipTest("unusual encoding on this system: %r" % enc)
        loc = BaseLocator.from_url("../%s.xml" % fname)
        self.assertEqual(loc.short_filename, fname)
        self.assertEqual(loc.short_name, u"test_short_names \xE9 \xEA")

    def test_win_xml_file(self):
        try:
            import ntpath
            import nturl2path
        except ImportError:
            return self.skipTest("Do not have ntpath or nturl2path installed.")
            
        global systemType
        old_sys_type = systemType
        old_path = os.path
        old_pathname2url = urllib.pathname2url
        old_url2pathname = urllib.url2pathname
        systemType = 'Windows'
        os.path = ntpath
        urllib.pathname2url = nturl2path.pathname2url
        urllib.url2pathname = nturl2path.url2pathname
        try:
            loc_str = "C:\\vt?dir\\tmp\\test_win_xml_file.xml?workflow=3"
            loc = BaseLocator.from_url(loc_str)
            self.assertIsInstance(loc, XMLFileLocator)
            self.assertEqual(loc.short_filename, "test_win_xml_file")
            self.assertEqual(loc.kwargs['version_node'], 3)
            self.assertEqual(
                    loc.to_url(),
                    "file:///C:/vt%3Fdir/tmp/test_win_xml_file.xml?workflow=3")
        finally:
            systemType = old_sys_type
            os.path = old_path
            urllib.pathname2url = old_pathname2url
            urllib.url2pathname = old_url2pathname

    def test_parse_db(self):
        loc_str = "db://localhost:3306/vistrails?workflow=42"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsInstance(loc, DBLocator)
        self.assertEqual(loc.kwargs['version_node'], 42)
        self.assertEqual(loc._host, "localhost")
        self.assertEqual(loc._port, 3306)
        self.assertEqual(loc._db, "vistrails")
        self.assertEqual(loc.to_url(), loc_str)

    def test_parse_bad_url(self):
        loc_str = "http://blah.com/"
        loc = BaseLocator.from_url(loc_str)
        self.assertIsNone(loc)

if __name__ == '__main__':
    unittest.main()
