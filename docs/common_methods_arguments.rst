.. _common_methods_arguments:

.. currentmodule:: common_methods_arguments

Common methods arguments
------------------------

.. argument:: overwrite   

    Argument used by (or passed to) the methods that save files.

    - ``overwrite=True``: if a file with the same name already exists, it gets overwritten. 
    - ``overwrite=False``: if a file with the same name already exists, it gets renamed with the :func:`utils.check_rename_file <NF4HEP.utils.check_rename_file>` function.
    - ``overwrite="dump"``: a dump of the file is saved through the :func:`utils.generate_dump_file_name <NF4HEP.utils.generate_dump_file_name>` function.
        
        - **type**: ``bool`` or ``str``
        - **allowed str**: ``"dump"``
        - **default**: ``False`` (if not specified otherwise)

.. argument:: timestamp   
            
    A "timestamp" string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
    If it is not passed, then it is generated, when needed. 
    In methods that save files it is passed to the :func:`utils.generate_dump_file_name <NF4HEP.utils.generate_dump_file_name>` 
    function to generate the dump file name.
    In the ``log`` dictionary it is used to label actions, i.e. calls to the class methods (as key in the dictionary).
    In the ``predictions`` dictionary it is used to label predictions (as key in the dictionary).
        
        - **type**: ``str``
        - **default**: ``None``

.. argument:: verbose   
            
    Verbosity mode. 
    See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
        
        - **type**: ``bool`` or ``int``
        - **default**: ``None``

.. argument:: show_plot   
            
    If ``True`` plots are shown on the 
    interactive console.
        
        - **type**: ``bool``
        - **default**: ``False``