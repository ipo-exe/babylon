import glob
import os, copy, shutil, datetime
import pandas as pd
import matplotlib.pyplot as plt

class MbaE:
    """
    **Mba'e** in Guarani means **Thing**.

    .. important::

        **Mba'e is the origin**. The the very-basic almost-zero level object.
        Deeper than here is only the Python builtin ``object`` class.


    """

    def __init__(self, name="MyMbaE", alias=None):
        """Initialize the ``MbaE`` object.

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from ``name``
        :type alias: str

        """
        # ------------ pseudo-static ----------- #

        #
        self.object_name = self.__class__.__name__
        self.object_alias = "mbae"

        # name
        self.name = name

        # alias
        self.alias = alias

        # handle None alias
        if self.alias is None:
            self._create_alias()

        # fields
        self._set_fields()

        # ------------ set mutables ----------- #
        self.bootfile = None
        self.folder_bootfile = "./"  # start in the local place

        # ... continues in downstream objects ... #

    def __str__(self):
        """The ``MbaE`` string"""
        str_type = str(type(self))
        str_df_metadata = self.get_metadata_df().to_string(index=False)
        str_out = "[{} ({})]\n{} ({}):\t{}\n{}".format(
            self.name,
            self.alias,
            self.object_name,
            self.object_alias,
            str_type,
            str_df_metadata,
        )
        return str_out

    def _create_alias(self):
        """If ``alias`` is ``None``, it takes the first and last characters from ``name``"""
        if len(self.name) >= 2:
            self.alias = self.name[0] + self.name[len(self.name) - 1]
        else:
            self.alias = self.name[:]

    def _set_fields(self):
        """Set fields names"""

        # Attribute fields
        self.name_field = "Name"
        self.alias_field = "Alias"

        # Metadata fields
        self.mdata_attr_field = "Attribute"
        self.mdata_val_field = "Value"
        # ... continues in downstream objects ... #

    def get_metadata(self):
        """Get a dictionary with object metadata.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        dict_meta = {
            self.name_field: self.name,
            self.alias_field: self.alias,
        }
        return dict_meta

    def get_metadata_df(self):
        """Get a :class:`pandas.DataFrame` created from the metadata dictionary

        :return: :class:`pandas.DataFrame` with ``Attribute`` and ``Value``
        :rtype: :class:`pandas.DataFrame`
        """
        dict_metadata = self.get_metadata()
        df_metadata = pd.DataFrame(
            {
                self.mdata_attr_field: [k for k in dict_metadata],
                self.mdata_val_field: [dict_metadata[k] for k in dict_metadata],
            }
        )
        return df_metadata

    def set(self, dict_setter):
        """Set selected attributes based on an incoming dictionary

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict
        """
        # ---------- set basic attributes --------- #
        self.name = dict_setter[self.name_field]
        self.alias = dict_setter[self.alias_field]

        # ... continues in downstream objects ... #

    def boot(self, bootfile):
        """Boot fundamental attributes from a ``csv`` table.

        :param bootfile: file path to ``csv`` table with metadata information.
            Expected format:

            .. code-block:: text

                Attribute;Value
                Name;ResTia
                Alias;Ra

        :type bootfile: str

        :return:
        :rtype: str
        """
        # ---------- update file attributes ---------- #
        self.bootfile = bootfile[:]
        self.folder_bootfile = os.path.dirname(bootfile)

        # get expected fields
        list_columns = [self.mdata_attr_field, self.mdata_val_field]

        # read info table from ``csv`` file. metadata keys are the expected fields
        df_info_table = pd.read_csv(bootfile, sep=";", usecols=list_columns)

        # setter loop
        dict_setter = {}
        for i in range(len(df_info_table)):
            # build setter from row
            dict_setter[df_info_table[self.mdata_attr_field].values[i]] = df_info_table[
                self.mdata_val_field
            ].values[i]

        # pass setter to set() method
        self.set(dict_setter=dict_setter)

        return None


class Collection(MbaE):
    """
    A collection of primitive ``MbaE`` objects with associated metadata.
    Useful for large scale manipulations in ``MbaE``-based objects.
    Expected to have custom methods and attributes downstream.

    Attributes:

    - ``catalog`` (:class:`pandas.DataFrame`): A catalog containing metadata of the objects in the test_collection.
    - ``collection`` (dict): A dictionary containing the objects in the ``Collection``.
    - name (str): The name of the ``Collection``.
    - alias (str): The name of the ``Collection``.
    - baseobject: The class of the base object used to initialize the ``Collection``.

    Methods:

    - __init__(self, base_object, name="myCatalog"): Initializes a new ``Collection`` with a base object.
    - update(self, details=False): Updates the ``Collection`` catalog.
    - append(self, new_object): Appends a new object to the ``Collection``.
    - remove(self, name): Removes an object from the ``Collection``.


    **Examples:**
    Here's how to use the Collection class:

    1. Initializing a Collection

    >>> base_obj = YourBaseObject()
    >>> test_collection = Collection(base_object=base_obj, name="myCatalog")

    2. Appending a New Object

    >>> new_obj = YourNewObject()
    >>> test_collection.append(new_object=new_obj)

    3. Removing an Object

    >>> test_collection.remove(name="ObjectToRemove")

    4. Updating the Catalog

    >>> test_collection.update(details=True)

    """

    def __init__(self, base_object, name="MyCollection", alias="Col0"):
        """Initialize the ``Collection`` object.

        :param base_object: ``MbaE``-based object for collection
        :type base_object: :class:`MbaE`

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # ------------ set pseudo-static ----------- #
        self.object_alias = "COL"
        # Set the name and baseobject attributes
        self.baseobject = base_object
        self.baseobject_name = base_object.__name__

        # Initialize the catalog with an empty DataFrame
        dict_metadata = self.baseobject().get_metadata()

        self.catalog = pd.DataFrame(columns=dict_metadata.keys())

        # Initialize the ``Collection`` as an empty dictionary
        self.collection = dict()

        # ------------ set mutables ----------- #
        self.size = 0

        self._set_fields()
        # ... continues in downstream objects ... #

    def __str__(self):
        """
        The ``Collection`` string.
        Expected to overwrite superior methods.
        """
        str_type = str(type(self))
        str_df_metadata = self.get_metadata_df().to_string(index=False)
        str_df_data = self.catalog.to_string(index=False)
        str_out = "{}:\t{}\nMetadata:\n{}\nCatalog:\n{}\n".format(
            self.name, str_type, str_df_metadata, str_df_data
        )
        return str_out

    def _set_fields(self):
        """
        Set fields names.
        Expected to increment superior methods.
        """
        # ------------ call super ----------- #
        super()._set_fields()

        # Attribute fields
        self.size_field = "Size"
        self.baseobject_field = "Base_Object"  # self.baseobject().__name__

        # ... continues in downstream objects ... #

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {
            self.size_field: self.size,
            self.baseobject_field: self.baseobject_name,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def update(self, details=False):
        """Update the ``Collection`` catalog.

        :param details: Option to update catalog details, defaults to False.
        :type details: bool
        :return: None
        :rtype: None
        """

        # Update details if specified
        if details:
            # Create a new empty catalog
            df_new_catalog = pd.DataFrame(columns=self.catalog.columns)

            # retrieve details from collection
            for name in self.collection:
                # retrieve updated metadata from base object
                dct_meta = self.collection[name].get_metadata()

                # set up a single-row helper dataframe
                lst_keys = dct_meta.keys()
                _dct = dict()
                for k in lst_keys:
                    _dct[k] = [dct_meta[k]]

                # Set new information
                df_aux = pd.DataFrame(_dct)

                # Append to the new catalog
                df_new_catalog = pd.concat([df_new_catalog, df_aux], ignore_index=True)

            # consider if the name itself has changed in the
            old_key_names = list(self.collection.keys())[:]
            new_key_names = list(df_new_catalog[self.catalog.columns[0]].values)

            # loop for checking consistency in collection keys
            for i in range(len(old_key_names)):
                old_key = old_key_names[i]
                new_key = new_key_names[i]
                # name change condition
                if old_key != new_key:
                    # rename key in the collection dictionary
                    self.collection[new_key] = self.collection.pop(old_key)

            # Update the catalog with the new details
            self.catalog = df_new_catalog.copy()
            # clear
            del df_new_catalog

        # Basic updates
        # --- the first row is expected to be the Unique name
        str_unique_name = self.catalog.columns[0]
        self.catalog = self.catalog.drop_duplicates(subset=str_unique_name, keep="last")
        self.catalog = self.catalog.sort_values(by=str_unique_name).reset_index(
            drop=True
        )
        self.size = len(self.catalog)
        return None

    # review ok
    def append(self, new_object):
        """Append a new object to the ``Collection``.

        The object is expected to have a ``.get_metadata()`` method that
        returns a dictionary with metadata keys and values

        :param new_object: Object to append.
        :type new_object: object

        :return: None
        :rtype: None
        """
        # Append a copy of the object to the ``Collection``
        copied_object = copy.deepcopy(new_object)
        self.collection[new_object.name] = copied_object

        # Update the catalog with the new object's metadata
        dct_meta = new_object.get_metadata()
        dct_meta_df = dict()
        for k in dct_meta:
            dct_meta_df[k] = [dct_meta[k]]
        df_aux = pd.DataFrame(dct_meta_df)
        self.catalog = pd.concat([self.catalog, df_aux], ignore_index=True)

        self.update()
        return None

    def remove(self, name):
        """Remove an object from the ``Collection`` by the name.

        :param name: Name attribute of the object to remove.
        :type name: str

        :return: None
        :rtype: None
        """
        # Delete the object from the ``Collection``
        del self.collection[name]
        # Delete the object's entry from the catalog
        str_unique_name = self.catalog.columns[
            0
        ]  # assuming the first column is the unique name
        self.catalog = self.catalog.drop(
            self.catalog[self.catalog[str_unique_name] == name].index
        ).reset_index(drop=True)
        self.update()
        return None


class DataSet(MbaE):
    """
    The core ``DataSet`` base/demo object.
    Expected to hold one :class:`pandas.DataFrame`.
    This is a Base and Dummy object. Expected to be implemented downstream for
    custom applications.

    """

    def __init__(self, name="MyDataSet", alias="DS0"):
        """Initialize the ``DataSet`` object.
        Expected to increment superior methods.

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "DS"

        # ------------ set mutables ----------- #
        self.file_data = None
        self.folder_data = None
        self.data = None
        self.size = None

        # descriptors
        self.source_data = None
        self.descri_data = None

        # ------------ set defaults ----------- #
        self.color = "blue"
        self.file_data_sep = ";"

        # UPDATE
        self.update()

        # ... continues in downstream objects ... #

    def __str__(self):
        """
        The ``DataSet`` string.
        Expected to overwrite superior methods.

        """
        str_super = super().__str__()
        if self.data is None:
            str_df_data = "None"
            str_out = "{}\nData:\n{}\n".format(str_super, str_df_data)
        else:
            # first 5 rows
            str_df_data_head = self.data.head().to_string(index=False)
            str_df_data_tail = self.data.tail().to_string(index=False)
            str_out = "{}\nData:\n{}\n ... \n{}\n".format(
                str_super, str_df_data_head, str_df_data_tail
            )
        return str_out

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # Attribute fields
        self.filedata_field = "File_Data"
        self.size_field = "Size"
        self.color_field = "Color"
        self.source_data_field = "Source"
        self.descri_data_field = "Description"

        # ... continues in downstream objects ... #

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_data,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": self.name,
            "width": 5 * 1.618,
            "height": 5 * 1.618,
            "xvar": "RM",
            "yvar": "TempDB",
            "xlabel": "RM",
            "ylabel": "TempDB",
            "color": self.color,
            "xmin": None,
            "xmax": None,
            "ymin": None,
            "ymax": None,
        }
        return None

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {
            self.size_field: self.size,
            self.color_field: self.color,
            self.source_data_field: self.source_data,
            self.descri_data_field: self.descri_data,
            self.filedata_field: self.file_data,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def update(self):
        """Refresh all mutable attributes based on data (includins paths).
        Base method. Expected to be incremented downstrem.

        :return: None
        :rtype: None
        """
        # refresh all mutable attributes

        # set fields
        self._set_fields()

        if self.data is not None:
            # data size (rows)
            self.size = len(self.data)

        # update data folder
        if self.file_data is not None:
            # set folder
            self.folder_data = os.path.abspath(os.path.dirname(self.file_data))
        else:
            self.folder_data = None

        # view specs at the end
        self._set_view_specs()

        # ... continues in downstream objects ... #
        return None

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        super().set(dict_setter=dict_setter)

        # ---------- settable attributes --------- #

        # COLOR
        self.color = dict_setter[self.color_field]

        # DATA: FILE AND FOLDER
        # handle if only filename is provided
        if os.path.isfile(dict_setter[self.filedata_field]):
            file_data = dict_setter[self.filedata_field][:]
        else:
            # assumes file is in the same folder as the boot-file
            file_data = os.path.join(
                self.folder_bootfile, dict_setter[self.filedata_field][:]
            )
        self.file_data = os.path.abspath(file_data)

        # -------------- set data logic here -------------- #
        if load_data:
            self.load_data(file_data=self.file_data)

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

    def load_data(self, file_data):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """

        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #
        default_columns = {
            #'DateTime': 'datetime64[1s]',
            "P": float,
            "RM": float,
            "TempDB": float,
        }
        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            file_data,
            sep=self.file_data_sep,
            dtype=default_columns,
            usecols=list(default_columns.keys()),
        )

        # -------------- post-loading logic -------------- #
        self.data.dropna(inplace=True)

        return None

    def view(self, show=True):
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view_specs["xlabel"] = "The X variable"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """
        # get specs
        specs = self.view_specs.copy()

        # --------------------- figure setup --------------------- #
        fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # --------------------- plotting --------------------- #
        plt.scatter(
            self.data[specs["xvar"]],
            self.data[specs["yvar"]],
            marker=".",
            color=specs["color"],
        )

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff
        plt.title(specs["title"])
        plt.ylabel(specs["ylabel"])
        plt.xlabel(specs["xlabel"])

        # handle min max
        if specs["xmin"] is None:
            specs["xmin"] = self.data[specs["xvar"]].min()
        if specs["ymin"] is None:
            specs["ymin"] = self.data[specs["yvar"]].min()
        if specs["xmax"] is None:
            specs["xmax"] = self.data[specs["xvar"]].max()
        if specs["ymax"] is None:
            specs["ymax"] = self.data[specs["yvar"]].max()

        plt.xlim(specs["xmin"], specs["xmax"])
        plt.ylim(specs["ymin"], 1.2 * specs["ymax"])

        # Adjust layout to prevent cutoff
        plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path


class RecordTable(DataSet):
    """
    The core object for Record Tables. A Record is expected to keep adding stamped records
    in order to keep track of large inventories, catalogs, etc.
    All records are expected to have a unique Id. It is considered to be a relational table.

    """

    def __init__(self, name="MyRecordTable", alias="RcT"):
        # prior attributes

        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)
        # overwriters
        self.object_alias = "FS"

        # --------- defaults --------- #
        self.id_size = 4 # for zfill

        # --------- customizations --------- #
        self._set_base_columns()
        self._set_data_columns()
        self._set_operator()

        # UPDATE
        self.update()


    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # base columns fields
        self.recid_field = "RecId"
        self.rectable_field = "RecTable"
        self.rectimest_field = "RecTimestamp"
        self.recstatus_field = "RecStatus"
        # ... continues in downstream objects ... #

    def _set_base_columns(self):
        """Set base columns names.
        Base Method. Expected to be incremented in superior methods.

        """
        self.columns_base = [
            self.recid_field,
            self.rectable_field,
            self.rectimest_field,
            self.recstatus_field
        ]
        # ... continues in downstream objects ... #

    def _set_data_columns(self):
        """Set specifics data columns names.
        Base Dummy Method. Expected to be incremented in superior methods.

        """
        # Main data columns
        self.columns_data_main = [
            "Kind",
            "Value",
        ]
        # Extra data columns
        self.columns_data_extra = [
            "Category",
        ]
        # File-related columns
        self.columns_data_files = [
            "File_NF",
            "File_Invoice"
        ]
        # concat all lists
        self.columns_data = self.columns_data_main + self.columns_data_extra + self.columns_data_files
        # ... continues in downstream objects ... #


    def _set_operator(self):
        """Set the builtin operator for automatic column calculations.
        This is a Base and Dummy method. It is expected to be overwrited and implemented downstream.

        :return: None
        :rtype:None
        """

        # ------------- define sub routines here ------------- #

        def func_file_status():
            return FileSys.check_file_status(files=self.data["File"].values)

        def func_sum():
            return None

        def func_age():
            return RecordTable.running_time(
                start_datetimes=self.data["Date_Birth"],
                kind="human"
            )

        # ---------------- the operator ---------------- #
        self.operator = {
            "Sum": func_sum,
            "Age": func_age,
            "File_Status": func_file_status
        }
        # remove here
        self.operator = None
        return None


    def _get_organized_columns(self):
        """Return the organized columns (base + data columns)

        :return: organized columns (base + data columns)
        :rtype:list
        """
        return self.columns_base + self.columns_data

    def _get_timestamp(self):
        """Return a string timestamp

        :return: full timestamp text %Y-%m-%d %H:%M:%S
        :rtype:str
        """
        # compute timestamp
        _now = datetime.datetime.now()
        return str(_now.strftime("%Y-%m-%d %H:%M:%S"))

    def _last_id_int(self):
        """Compute the last ID integer in the record data table.

        :return: last Id integer from the record data table.
        :rtype: int
        """
        if self.data is None:
            return 0
        else:
            df = self.data.sort_values(by=self.recid_field, ascending=True)
            return int(df[self.recid_field].values[-1].replace("Rec", ""))

    def _next_recid(self):
        """Get the next record id string based on the existing ids.

        :return: next record id
        :rtype: str
        """
        last_id_int = self._last_id_int()
        next_id = "Rec" + str(last_id_int + 1).zfill(self.id_size)
        return next_id

    def _filter_dict_rec(self, input_dict):
        """Filter input record dictionary based on the expected table data columns.

        :param input_dict: input record dictionary
        :type input_dict: dict
        :return: filtered record dictionary
        :rtype: dict
        """
        # ------ parse expected fields ------- #
        # filter expected columns
        dict_rec_filter = {}
        for k in self.columns_data:
            if k in input_dict:
                dict_rec_filter[k] = input_dict[k]
        return dict_rec_filter

    def update(self):
        super().update()
        # ... continues in downstream objects ... #
        return None

    def save(self):
        """Save the data to the sourced file data.

        .. danger::

            This method **overwrites** the sourced data file.


        :return: integer denoting succesfull save (0) or file not found (1)
        :rtype: int
        """
        if self.file_data is not None:
            # handle filename
            filename = os.path.basename(self.file_data).split(".")[0]
            # handle folder
            self.export(
                folder_export=os.path.dirname(self.file_data),
                filename=filename
            )
            return 0
        else:
            return 1

    def export(self, folder_export=None, filename=None, filter_archive=False):
        """Export the ``RecordTable`` data.

        :param folder_export: folder to export
        :type folder_export: str
        :param filename: file name (name alone, without file extension)
        :type filename: str
        :param filter_archive: option for exporting only records with ``RecStatus`` = ``On``
        :type filter_archive: bool
        :return: file path is export is successfull (1 otherwise)
        :rtype: str or int
        """
        if filename is None:
            filename = self.name
        # append extension
        filename = filename + ".csv"
        if self.data is not None:
            # handle folders
            if folder_export is not None:
                filepath = os.path.join(folder_export, filename)
            else:
                filepath = os.path.join(self.folder_data, filename)
            # handle archived records
            if filter_archive:
                df = self.data.query("RecStatus == 'On'")
            else:
                df = self.data.copy()
            # filter default columns:
            df = df[self._get_organized_columns()]
            df.to_csv(
                filepath, sep=self.file_data_sep, index=False
            )
            return filepath
        else:
            return 1

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        # ignore color
        dict_setter[self.color_field] = None
        super().set(dict_setter=dict_setter, load_data=False)

        # ---------- set basic attributes --------- #

        # -------------- set data logic here -------------- #
        if load_data:
            self.load_data(file_data=self.file_data)
            self.refresh_data()

        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

    def refresh_data(self):  # todo docstring
        if self.operator is not None:
            for c in self.operator:
                self.data[c] = self.operator[c]()
        # update object
        self.update()

    def load_data(self, file_data):
        """Load data from file.
        Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """
        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)
        # -------------- implement loading logic -------------- #

        # -------------- call loading function -------------- #
        df = pd.read_csv(
            file_data,
            sep=self.file_data_sep
        )

        # -------------- post-loading logic -------------- #
        self.set_data(input_df=df)

        return None

    def set_data(self, input_df, append=True, inplace=True):
        """Set RecordTable data from incoming dataframe.
        Base Method. Expected to be incremented downstream.

        :param input_df: incoming dataframe
        :type input_df: dataframe

        :param append: option for appending the dataframe to existing data. Default True
        :type append: bool

        :param inplace: option for overwrite data. Else return dataframe. Default True
        :type inplace: bool

        :return: None
        :rtype: None
        """
        list_input_cols = list(input_df.columns)

        # overwrite RecTable column
        input_df[self.rectable_field] = self.name

        # handle RecId
        if self.recid_field not in list_input_cols:
            # enforce Id based on index
            n_last_id = self._last_id_int()
            n_incr = n_last_id + 1
            input_df[self.recid_field] = ["Rec" + str(_ + n_incr).zfill(self.id_size) for _ in input_df.index]
        else:
            # remove incoming duplicates
            input_df.drop_duplicates(subset=self.recid_field, inplace=True)

        # handle timestamp
        if self.rectimest_field not in list_input_cols:
            input_df[self.rectimest_field] = self._get_timestamp()

        # handle timestamp
        if self.recstatus_field not in list_input_cols:
            input_df[self.recstatus_field] = "On"

        # Add missing columns with default values
        for column in self._get_organized_columns():
            if column not in input_df.columns:
                input_df[column] = ""
        df_merged = input_df[self._get_organized_columns()]

        # concatenate dataframes
        if append:
            if self.data is not None:
                df_merged = pd.concat([self.data, df_merged], ignore_index=True)

        if inplace:
            # pass copy
            self.data = df_merged.copy()
            return None
        else:
            return df_merged


    def insert_record(self, dict_rec):
        """Insert a record in the RT

        :param dict_rec: input record dictionary
        :type dict_rec: dict
        :return: None
        :rtype: None
        """

        # ------ parse expected fields ------- #
        # filter expected columns
        dict_rec_filter = self._filter_dict_rec(input_dict=dict_rec)
        # ------ set default fields ------- #
        # set table field
        dict_rec_filter[self.rectable_field] = self.name
        # create index
        dict_rec_filter[self.recid_field] = self._next_recid()
        # compute timestamp
        dict_rec_filter[self.rectimest_field] = self._get_timestamp()
        # set active
        dict_rec_filter[self.recstatus_field] = "On"

        # ------ merge ------- #
        # create single-row dataframe
        df = pd.DataFrame({k: [dict_rec_filter[k]] for k in dict_rec_filter})
        # concat to data
        self.data = pd.concat([self.data, df]).reset_index(drop=True)

        self.update()
        return None

    def edit_record(self, rec_id, dict_rec, filter_dict=True):
        """Edit RT record

        :param rec_id: record id
        :type rec_id: str
        :param dict_rec: incoming record dictionary
        :type dict_rec: dict
        :return: None
        :rtype: None
        """
        # input dict rec data
        if filter_dict:
            dict_rec_filter = self._filter_dict_rec(input_dict=dict_rec)
        else:
            dict_rec_filter = dict_rec
        # include timestamp for edit operation
        dict_rec_filter[self.rectimest_field] = self._get_timestamp()

        # get data
        df = self.data.copy()
        # set index
        df = df.set_index(self.recid_field)
        # get filter series by rec id
        sr = df.loc[rec_id].copy()

        # update edits
        for k in dict_rec_filter:
            sr[k] = dict_rec_filter[k]

        # set in row
        df.loc[rec_id] = sr
        # restore index
        df.reset_index(inplace=True)
        self.data = df.copy()

        return None

    def archive_record(self, rec_id):
        """Archive a record in the RT, that is ``RecStatus`` = ``Off``

        :param rec_id: record id
        :type rec_id: str
        :return: None
        :rtype: None
        """
        dict_rec = {
            self.recstatus_field: "Off"
        }
        self.edit_record(rec_id=rec_id, dict_rec=dict_rec, filter_dict=False)
        return None


    def get_record(self, rec_id):
        """Get a record dictionary

        :param rec_id: record id
        :type rec_id: str
        :return: record dictionary
        :rtype: dict
        """
        # set index
        df = self.data.set_index(self.recid_field)

        # locate series by index and convert to dict
        dict_rec = {self.recid_field: rec_id}
        dict_rec.update(dict(df.loc[rec_id].copy()))
        return dict_rec

    def get_record_df(self, rec_id):
        """Get a record dataframe

        :param rec_id: record id
        :type rec_id: str
        :return: record dictionary
        :rtype: dict
        """
        # get dict
        dict_rec = self.get_record(rec_id=rec_id)
        # convert in vertical dataframe
        dict_df = {
            "Field": [k for k in dict_rec],
            "Value": [dict_rec[k] for k in dict_rec],
        }
        return pd.DataFrame(dict_df)

    def load_record_data(self, file_record_data, input_field="Field", input_value="Value"):
        """Load record data from a ``csv`` file.

        .. note::

            This method **does not insert** the record data to the ``RecordTable``.


        :param file_record_data: file path to ``csv`` file.
        :type file_record_data: str
        :param input_field: Name of ``Field`` column in the file.
        :type input_field:
        :param input_value: Name of ``Value`` column in the file.
        :type input_value:
        :return: record dictionary
        :rtype: dict
        """
        # load record from file
        df = pd.read_csv(
            file_record_data,
            sep=self.file_data_sep,
            usecols=[input_field, input_value]
        )
        # convert into a dict
        dict_rec_raw = {df[input_field].values[i]: df[input_value].values[i] for i in range(len(df))}

        # filter for expected data columns
        dict_rec = {}
        for c in self.columns_data:
            if c in dict_rec_raw:
                dict_rec[c] = dict_rec_raw[c]

        return dict_rec

    def export_record(self, rec_id, filename=None, folder_export=None):
        """Export a record to a csv file.

        :param rec_id: record id
        :type rec_id: str
        :param filename: file name (name alone, without file extension)
        :type filename: str
        :param folder_export: folder to export
        :type folder_export: str
        :return: path to exported file
        :rtype: str
        """
        # retrieve dataframe
        df = self.get_record_df(rec_id=rec_id)
        # handle filename and folder
        if filename is None:
            filename = self.name + "_" + rec_id
        if folder_export is None:
            folder_export = self.folder_data
        filepath = os.path.join(folder_export, filename + ".csv")
        # save
        df.to_csv(filepath, sep=self.file_data_sep, index=False)
        return filepath

    # ----------------- STATIC METHODS ----------------- #
    @staticmethod
    def timedelta_disagg(timedelta):
        """Util static method for dissaggregation of time delta

        :param timedelta: TimeDelta object from pandas
        :type timedelta: :class:`pandas.TimeDelta`
        :return: dictionary of time delta
        :rtype: dict
        """
        days = timedelta.days
        years, days = divmod(days, 365)
        months, days = divmod(days, 30)
        hours, remainder = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return {
            "Years": years,
            "Months": months,
            "Days": days,
            "Hours": hours,
            "Minutes": minutes,
            "Seconds": seconds
        }

    @staticmethod
    def timedelta_to_str(timedelta, dct_struct):
        """Util static method for string conversion of timedelta

        :param timedelta: TimeDelta object from pandas
        :type timedelta: :class:`pandas.TimeDelta`
        :param dct_struct: Dictionary of string strucuture. Ex: {'Expected days': 'Days'}
        :type dct_struct: dict
        :return: text of time delta
        :rtype: str
        """
        dct_td = RecordTable.timedelta_disagg(timedelta=timedelta)
        parts = []
        for k in dct_struct:
            parts.append("{}: {}".format(dct_struct[k], dct_td[k]))
        return ", ".join(parts)

    @staticmethod
    def running_time(start_datetimes, kind="raw"):
        """Util static method for computing the runnning time for a list of starting dates

        :param start_datetimes: List of starting dates
        :type start_datetimes: list
        :param kind: mode for output format ('raw', 'human' or 'age')
        :type kind: str
        :return: list of running time
        :rtype: list
        """
        # Convert 'start_datetimes' to datetime format
        start_datetimes = pd.to_datetime(start_datetimes)
        # Calculate the running time as a timedelta
        current_datetime = pd.to_datetime('now')
        running_time = current_datetime - start_datetimes
        # Apply the custom function to create a new column
        if kind == "raw":
            running_time = running_time.tolist()
        elif kind == "human":
            dct_str = {
                "Years": "yr",
                "Months": "mth"
            }
            running_time = running_time.apply(RecordTable.timedelta_to_str, args=(dct_str,))
        elif kind == "age":
            running_time = [int(e.days / 365) for e in running_time]

        return running_time


class FileSys(DataSet):
    """
    The core ``FileSys`` base/demo object. File System object.
    Useful for complex folder structure setups and controlling the status
    of expected file.
    This is a Base and Dummy object. Expected to be implemented downstream for
    custom applications.

    """

    def __init__(self, folder_base, name="MyFS", alias="FS0"):
        """Initialize the ``FileSys`` object.
        Expected to increment superior methods.

        :param folder_base: path to File System folder location
        :type folder_base: str

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # prior attributes
        self.folder_base = folder_base

        # ------------ call super ----------- #
        super().__init__(name=name, alias=alias)

        # overwriters
        self.object_alias = "FS"

        # ------------ set mutables ----------- #
        self.folder_main = os.path.join(self.folder_base, self.name)
        self._set_view_specs()

        # ... continues in downstream objects ... #

    def _set_fields(self):
        """
        Set fields names. Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()

        # Attribute fields
        self.folder_base_field = "Folder_Base"

        # ... continues in downstream objects ... #

    def _set_view_specs(self):
        """Set view specifications.
        Expected to overwrite superior methods.

        :return: None
        :rtype: None
        """
        self.view_specs = {
            "folder": self.folder_data,
            "filename": self.name,
            "fig_format": "jpg",
            "dpi": 300,
            "title": self.name,
            "width": 5 * 1.618,
            "height": 5 * 1.618,
            # todo include more view specs
        }
        return None

    #
    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {self.folder_base_field: self.folder_base}

        # update
        dict_meta.update(dict_meta_local)

        # removals

        # remove color
        dict_meta.pop(self.color_field)
        # remove source
        dict_meta.pop(self.source_data_field)

        return dict_meta

    def get_structure(self):
        """Get FileSys structure dictionary. Expected to overwrite superior methods

        :return: structure dictionary
        :rtype: dict
        """
        # get pandas DataFrame
        df = self.data.copy()

        # Initialize the nested dictionary
        dict_structure = {}

        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            current_dict = dict_structure

            # Split the folder path into individual folder names
            folders = row["Folder"].split("/")

            # Iterate over the folders to create the nested structure
            for folder in folders:
                current_dict = current_dict.setdefault(folder, {})

            # If a file is present, add it to the nested structure
            if pd.notna(row["File"]):
                current_dict[row["File"]] = [
                    row["Format"],
                    row["File_Source"] if pd.notna(row["File_Source"]) else "",
                    row["Folder_Source"] if pd.notna(row["Folder_Source"]) else "",
                ]
        return dict_structure

    def get_status(self, folder_name): # todo dosctring
        dict_status = {}
        # get full folder path
        folder = self.folder_main + "/" + folder_name
        # filter expected files
        df = self.data.copy()
        df = df.query("Folder == '{}'".format(folder_name))
        df = df.dropna(subset=["File"])
        if len(df) == 0:
            return None
        else:
            dict_status["Folder"] = df.copy()
            dict_status["Files"] = {}
            dict_files = {}
            for i in range(len(df)):
                # get file name
                lcl_file_name = df["File"].values[i]
                dict_files[lcl_file_name] = {}
                # get file format
                lcl_file_format = df["Format"].values[i]
                # get extensions:
                dict_extensions = self.get_extensions()
                #
                list_lcl_extensions = dict_extensions[lcl_file_format]
                #print(list_lcl_extensions)
                for ext in list_lcl_extensions:
                    lcl_path = os.path.join(folder, lcl_file_name + "." + ext)
                    list_files = glob.glob(lcl_path)
                    lst_filenames_found = [os.path.basename(f) for f in list_files]
                    dict_files[lcl_file_name][ext] = lst_filenames_found
            for k in dict_files:
                # Convert each list in the dictionary to a pandas Series and then create a DataFrame
                _df = pd.DataFrame({key: pd.Series(value) for key, value in dict_files[k].items()})
                if len(_df) == 0:
                    for c in _df.columns:
                        _df[c] = [None]
                _df = _df.fillna("missing")
                dict_status["Files"][k] = _df.copy()

            return dict_status

    def update(self):
        super().update()
        # set main folder
        self.folder_main = os.path.join(self.folder_base, self.name)
        # ... continues in downstream objects ... #

    def set(self, dict_setter, load_data=True):
        """Set selected attributes based on an incoming dictionary.
        Expected to increment superior methods.

        :param dict_setter: incoming dictionary with attribute values
        :type dict_setter: dict

        :param load_data: option for loading data from incoming file. Default is True.
        :type load_data: bool

        """
        # ignore color
        dict_setter[self.color_field] = None

        # -------------- super -------------- #
        super().set(dict_setter=dict_setter, load_data=False)

        # ---------- set basic attributes --------- #
        # set base folder
        self.folder_base = dict_setter[self.folder_base_field]


        # -------------- set data logic here -------------- #
        if load_data:
            self.load_data(file_data=self.file_data)


        # -------------- update other mutables -------------- #
        self.update()

        # ... continues in downstream objects ... #

    def load_data(self, file_data):
        """Load data from file. Expected to overwrite superior methods.

        :param file_data: file path to data.
        :type file_data: str
        :return: None
        :rtype: None
        """
        # -------------- overwrite relative path input -------------- #
        file_data = os.path.abspath(file_data)

        # -------------- implement loading logic -------------- #

        # -------------- call loading function -------------- #
        self.data = pd.read_csv(
            file_data,
            sep=self.file_data_sep
        )

        # -------------- post-loading logic -------------- #

        return None

    def setup(self):
        """
        This method sets up all the FileSys structure (default folders and files)

        .. danger::

            This method overwrites all existing default files.

        """
        # update structure
        self.structure = self.get_structure()

        # make main dir
        self.make_dir(str_path=self.folder_main)

        # fill structure
        FileSys.fill(dict_struct=self.structure, folder=self.folder_main)

    def backup(self, location_dir,  version_id="v-0-0",):  # todo docstring
        dst_dir = os.path.join(location_dir, self.name + "_" + version_id)
        FileSys.archive(src_dir=self.folder_main, dst_dir=dst_dir)
        return None

    def view(self, show=True): # todo implement
        """Get a basic visualization.
        Expected to overwrite superior methods.

        :param show: option for showing instead of saving.
        :type show: bool

        :return: None or file path to figure
        :rtype: None or str

        **Notes:**

        - Uses values in the ``view_specs()`` attribute for plotting

        **Examples:**

        Simple visualization:

        >>> ds.view(show=True)

        Customize view specs:

        >>> ds.view_specs["title"] = "My Custom Title"
        >>> ds.view(show=True)

        Save the figure:

        >>> ds.view_specs["folder"] = "path/to/folder"
        >>> ds.view_specs["filename"] = "my_visual"
        >>> ds.view_specs["fig_format"] = "png"
        >>> ds.view(show=False)

        """

        # get specs
        specs = self.view_specs.copy()

        # --------------------- figure setup --------------------- #

        # fig = plt.figure(figsize=(specs["width"], specs["height"]))  # Width, Height

        # --------------------- plotting --------------------- #

        # --------------------- post-plotting --------------------- #
        # set basic plotting stuff

        # Adjust layout to prevent cutoff
        # plt.tight_layout()

        # --------------------- end --------------------- #
        # show or save
        if show:
            # plt.show()
            return None
        else:
            file_path = "{}/{}.{}".format(
                specs["folder"], specs["filename"], specs["fig_format"]
            )
            plt.savefig(file_path, dpi=specs["dpi"])
            plt.close(fig)
            return file_path

    # ----------------- STATIC METHODS ----------------- #
    @staticmethod
    def archive(src_dir, dst_dir):
        # Create a zip archive from the directory
        shutil.make_archive(dst_dir, 'zip', src_dir)
        return None

    @staticmethod
    def get_extensions():
        list_basics = [
            "pdf",
            "docx",
            "xlsx",
            "bib",
            "tex",
            "svg",
            "png",
            "jpg",
            "txt",
            "csv",
            "qml",
            "tif",
            "gpkg",
        ]
        dict_extensions = {e: [e] for e in list_basics}
        dict_aliases = {
            "table": ["csv"],
            "raster": ["asc", "prj", "qml"],
            "qraster": ["asc", "prj", "csv", "qml"],
            "fig": ["jpg"],
            "vfig": ["svg"],
            "receipt": ["pdf", "jpg"],
        }
        dict_extensions.update(dict_aliases)
        return dict_extensions

    @staticmethod
    def check_file_status(files):
        """Static method for file existing checkup

        :param files: iterable with file paths
        :type files: list
        :return: list status ('ok' or 'missing')
        :rtype: list
        """
        list_status = []
        for f in files:
            str_status = "missing"
            if os.path.isfile(f):
                str_status = "ok"
            list_status.append(str_status)
        return list_status

    @staticmethod
    def make_dir(str_path):
        """Util function for making a diretory

        :param str_path: path to dir
        :type str_path: str
        :return: None
        :rtype: None
        """
        if os.path.isdir(str_path):
            pass
        else:
            os.mkdir(str_path)
        return None

    @staticmethod
    def copy_batch(dst_pattern, src_pattern):
        """Util static method for batch-copying pattern-based files.

        .. note::

            Pattern is expected to be a prefix prior to ``*`` suffix.

        :param dst_pattern: destination path with file pattern. Example: path/to/dst_file_*.csv
        :type dst_pattern: str
        :param src_pattern: source path with file pattern. Example: path/to/src_file_*.csv
        :type src_pattern: str
        :return: None
        :rtype: None
        """
        # handle destination variables
        dst_basename = os.path.basename(dst_pattern).split(".")[0].replace("*", "")  # k
        dst_folder = os.path.dirname(dst_pattern)  # folder

        # handle sourced variables
        src_extension = os.path.basename(src_pattern).split(".")[1]
        src_prefix = os.path.basename(src_pattern).split(".")[0].replace("*", "")

        # get the list of sourced files
        list_files = glob.glob(src_pattern)
        # copy loop
        if len(list_files) != 0:
            for _f in list_files:
                _full = os.path.basename(_f).split(".")[0]
                _suffix = _full[len(src_prefix):]
                _dst = os.path.join(
                    dst_folder, dst_basename + _suffix + "." + src_extension
                )
                shutil.copy(_f, _dst)
        return None

    @staticmethod
    def fill(dict_struct, folder, handle_files=True):
        """Recursive function for filling the ``FileSys`` structure

        :param dict_struct: dicitonary of directory structure
        :type dict_struct: dict

        :param folder: path to local folder
        :type folder: str

        :return: None
        :rtype: None
        """

        def handle_file(dst_name, lst_specs, dst_folder):
            """Sub routine for handling expected files in the FileSys structure.

            :param dst_name: destination filename
            :type dst_name: str
            :param lst_specs: list for expected file specifications
            :type lst_specs: list
            :param dst_folder: destination folder
            :type dst_folder: str
            :return: None
            :rtype: None
            """
            dict_exts = FileSys.get_extensions()
            lst_exts = dict_exts[lst_specs[0]]
            src_name = lst_specs[1]
            src_dir = lst_specs[2]

            # there is a sourcing directory
            if os.path.isdir(src_dir):
                # extension loop:
                for extension in lst_exts:
                    # source
                    src_file = src_name + "." + extension
                    src_filepath = os.path.join(src_dir, src_file)
                    # destination
                    dst_file = dst_name + "." + extension
                    dst_filepath = os.path.join(dst_folder, dst_file)
                    #
                    # there might be a sourced file
                    if os.path.isfile(src_filepath):
                        shutil.copy(src=src_filepath, dst=dst_filepath)
                    elif "*" in src_name:
                        # is a pattern file
                        FileSys.copy_batch(
                            src_pattern=src_filepath, dst_pattern=dst_filepath
                        )
                    else:
                        pass

            return None

        # structure loop:
        for k in dict_struct:
            # get current folder or file
            _d = folder + "/" + k

            # [case 1] bottom is a folder
            if isinstance(dict_struct[k], dict):
                # make a dir
                FileSys.make_dir(str_path=_d)
                # now move down:
                FileSys.fill(dict_struct=dict_struct[k], folder=_d)

            # bottom is an expected file
            else:
                if handle_files:
                    handle_file(dst_name=k, lst_specs=dict_struct[k], dst_folder=folder)

        return None


if __name__ == "__main__":
    fs = FileSys(folder_base="C:/data", name="MyPlans", alias="MPlans")
    fs.load_data(file_data="./iofiles.csv")
    print(fs.folder_main)
    fs.data  = fs.data[["Folder", "File", "Format", "Description", "File_Source", "Folder_Source"]].copy()
    #fs.setup()

    d = fs.get_status(folder_name=r"datasets\\topo")
    if d is None:
        print("Not found expected files")
    else:
        print(d["Folder"][["Folder", "File", "Format", "Description"]].to_string(index=False))
        for f in d["Files"]:
            print("*"*40)
            print(f)
            print(d["Files"][f].to_string(index=False))
