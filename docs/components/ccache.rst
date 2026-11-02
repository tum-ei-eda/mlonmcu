Compiler caching with MLIF
=========================

The ``ccache`` feature caches MLIF C and C++ compilations. Install ``ccache``
on the build machine, then enable it with ``-f ccache``::

    export MLONMCU_HOME=/work/git/mlonmcu/mlonmcu/workspace_default
    source venv/bin/activate
    mlonmcu flow compile aww --target host_x86 --backend tflmi -f ccache

The feature is supported by ``mlif`` and its ``mlif_litex`` derivative. It
caches compilation, while configuration, linking, and artifact generation
still run normally. The executable defaults to ``ccache`` on the build
process's PATH; set ``-c ccache.exe=/path/to/ccache`` to override it.

Cache scopes
------------

Select the scope with ``-c ccache.mode=MODE``:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Mode
     - Cache directory
     - Configuration
   * - ``session`` (default)
     - ``<session directory>/ccache``
     - ``<session directory>/ccache/ccache.conf``
   * - ``workspace``
     - ``$MLONMCU_HOME/cache/ccache``
     - ``$MLONMCU_HOME/cache/ccache/ccache.conf``
   * - ``global``
     - User/system ccache directory
     - Normal ccache configuration and environment

Session mode shares one cache across all runs in that session, including
parallel workers. Different sessions get separate caches. It retains the
cache when the session closes so a resumed session can reuse it. To discard
only that session's cache on close, including on failure, set::

    -c session.ccache_cleanup=true

Workspace mode persists across sessions. The workspace comes from the active
MLonMCU context (including ``--home``), with ``MLONMCU_HOME`` as a fallback for
standalone platform use. Custom session destinations do not change the
workspace cache location. Global mode uses the existing user/system cache
and settings without modifying their configuration files.

Managed modes create an empty, editable ``ccache.conf`` on first use and keep
it on subsequent builds. They use ccache's built-in defaults until configured.
They isolate the child build processes from inherited ``CCACHE_*`` variables
and user/system ccache configuration, including inherited remote storage.
Settings in each managed cache's configuration file apply to that scope.
The parent process environment is preserved in all modes.

Settings
--------

The following overrides apply to build child processes in any mode; omitted
values preserve the selected scope's settings:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature configuration
     - Meaning
   * - ``ccache.max_size=2G``
     - Maximum cache size using ccache's size syntax; ``0`` means unlimited
   * - ``ccache.max_files=10000``
     - Maximum cached file count; ``0`` means unlimited
   * - ``ccache.compression=true``
     - Enable or disable cache compression
   * - ``ccache.compiler_check=content``
     - Compiler identity check; passed through to ccache

These map to ccache environment overrides and do not edit a shared
configuration file. For additional settings, edit the selected cache's
``ccache.conf`` using the syntax supported by your installed ccache version.
In global mode, normal ``CCACHE_*`` variables also apply.

For example, share a 2 GB cache across sessions::

    mlonmcu flow compile aww --target host_x86 --backend tflmi -f ccache \
      -c ccache.mode=workspace ccache.max_size=2G ccache.compression=true

Direct platform configuration is also supported through ``mlif.ccache=true``
and ``mlif.ccache_mode``, ``mlif.ccache_max_size``, ``mlif.ccache_max_files``,
``mlif.ccache_compression``, ``mlif.ccache_compiler_check``, and
``mlif.ccache_exe``. The feature's values take precedence when the feature is
supplied. Standalone MLIF without a session must select workspace or global
mode explicitly.

Reuse across run directories
---------------------------

MLonMCU sets ``CCACHE_BASEDIR`` to each run's root, rather than to the common
session/workspace parent. Matching generated source paths therefore normalize
to the same relative paths even when run IDs differ. MLIF adds
``-fdebug-prefix-map=<run root>=.`` for GCC/Clang C and C++ compilation so debug
paths are also normalized. It does not disable ccache's directory hashing.
These mechanisms follow ccache's guidance on
`compiling in different directories <https://ccache.dev/manual/4.8.2.html#_compiling_in_different_directories>`_.

CMake can put the full source path into dependency-target names for sources
outside its source/build trees. To keep these names stable, MLIF snapshots
generated model sources under ``<MLIF build directory>/ccache-model`` when
ccache is enabled. This also handles templates which resolve source symlinks.
The snapshot costs additional disk space and copying; original artifacts are
preserved. Existing sources inside the MLIF build tree need no snapshot.
Debug source paths are relative to the corresponding run directory.

To exercise two equivalent runs sequentially in one session::

    mlonmcu flow compile aww --target host_x86 --backend tflmi -f ccache \
      --config-gen mlif.num_threads=2 --config-gen mlif.num_threads=4

The thread-count variation creates two runs without changing compiler inputs.
Inspect the session cache without falling back to user/system settings::

    CCACHE_DIR=/path/to/session/ccache \
      CCACHE_CONFIGPATH=/path/to/session/ccache/ccache.conf ccache --show-stats

Cache hits require matching compiler inputs and settings. Changed model code,
headers, compilation options, or compiler identity can cause misses. Linking
and backend model generation are not cached by this feature.

Compiler identity
-----------------

By default, ccache identifies the compiler using its modification time and
size, and also includes the compiler's name in the cache key. The absolute
executable path alone does not identify the compiler. Replacing a compiler
at the same path normally causes misses; relocating an identical compiler
with the same name and metadata can still allow hits, provided the other
compiler inputs match.

For locally rebuilt or frequently redeployed compilers, use::

    -c ccache.compiler_check=content

This hashes the compiler executable's bytes. A byte-identical rebuild or copy
can reuse entries even when its timestamp changes; a different executable
gets a different identity. This does not fingerprint every component of a
toolchain: a GCC driver can invoke a separate ``cc1`` binary, and compilers
can load shared libraries. If those components change independently, use
an immutable full-toolchain build identifier with ccache's
``compiler_check=string:<identifier>`` setting, or configure
``extra_files_to_hash`` in the managed ``ccache.conf``. A version banner alone
may not distinguish locally patched compilers. See the
`ccache compiler-check documentation <https://ccache.dev/manual/3.7.7.html#_configuration_settings>`_.

Scheduling experiments
----------------------

``--parallel`` controls concurrent runs; ``mlif.num_threads`` controls make
parallelism inside each compilation. ``runs_per_stage=0`` processes complete
run pipelines concurrently. ``runs_per_stage=1`` processes all runs through
one stage before starting the next, and permits stage-specific limits such as
``--parallel-per-stage BUILD=2,COMPILE=4,RUN=4``. Unspecified stages use
``--parallel``. The limits count runs, not individual compiler processes.

An empty shared cache makes scheduling order significant: later runs can
reuse completed compilations, while concurrent equivalent cache misses can
repeat the same compilation. A warm cache changes the best allocation, since
compiler CPU work is replaced by hashing, file reads, configuration and
linking. Measure empty-cache batch time and warm-cache batch time separately,
and compare complete experiment wall time rather than summed per-run times.
For overlapping compile stages, the union of their time intervals measures
time with at least one active compile; the sum measures accumulated run time.
