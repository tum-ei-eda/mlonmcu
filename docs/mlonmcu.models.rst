mlonmcu.models package
======================

Submodules
----------

mlonmcu.models.convert\_data module
-----------------------------------

.. automodule:: mlonmcu.models.convert_data
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.frontend module
------------------------------

Model checksums in ``definition.yml`` (``network.hash.algorithm`` and
``network.hash.value``) are verified by default before processing metadata.
Models without a checksum are loaded without verification. To skip verification,
set the frontend's ``check_integrity`` option to false, for example::

   python3 -m mlonmcu.cli.main flow run resnet -b tvmaotplus -t host_x86 -c tflite.check_integrity=False

.. automodule:: mlonmcu.models.frontend
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.group module
---------------------------

.. automodule:: mlonmcu.models.group
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.lookup module
----------------------------

.. automodule:: mlonmcu.models.lookup
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.metadata module
------------------------------

.. automodule:: mlonmcu.models.metadata
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.model module
---------------------------

.. automodule:: mlonmcu.models.model
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.options module
-----------------------------

.. automodule:: mlonmcu.models.options
   :members:
   :undoc-members:
   :show-inheritance:

mlonmcu.models.utils module
---------------------------

.. automodule:: mlonmcu.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: mlonmcu.models
   :members:
   :undoc-members:
   :show-inheritance:
