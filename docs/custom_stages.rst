Custom run pipelines
====================

Runs accept an optional ``pipeline`` argument. A session schedules these run
pipelines using the existing executors. Omitting it preserves the default
``LOAD → TUNE → BUILD → COMPILE → RUN → POSTPROCESS`` workflow and its capability
checks. All default stages and ``flash`` are registered ``Stage`` classes, using
the same registry and result contract as extensions. Existing ``RunStage``
arguments and ``run.load()``, ``run.build()``, etc. remain compatibility APIs.
IDs are case-insensitive strings; ``DONE`` means the end of the selected pipeline.

Describe pipelines in Python, or in the existing YAML run initializer format::

    runs:
      - model_name: sine
        frontend_names: [tflite]
        framework_name: tflm
        backend_name: tflmi
        platform_names: [mlif]
        target_name: etiss
        pipeline:
          stages:
            - id: load
            - id: quantize
              uses: quantize
              inputs:
                model: load.model
              config:
                command: [/path/to/quantizer, --input, '{input}', --output, '{output}']
                output_name: quantized.tflite
            - id: build
              inputs:
                model: quantize.model
            - id: compile
            - id: run
            - id: postprocess
              optional: true

Select component names available in your environment. To use the example
quantization adapter, copy ``examples/custom_stages/extensions.py`` into the
environment's configured plugins directory and enable ``vars.allow_extensions``.
It runs a user-supplied external quantizer with an argv list, captures its logs,
and reports output size and conversion time. Supply calibration arguments and
data according to your quantizer; MLonMCU does not select a quantization algorithm.

Execute all stages, or stop at a custom stage::

    mlonmcu flow execute --initializer initializer.yml
    mlonmcu flow execute --initializer initializer.yml --until quantize

In Python::

    run = session.create_run(pipeline={"stages": [...]})
    session.process_runs(until="quantize", context=context)

An initializer describes a new run. Reloading it does not restore completed
stage state from disk. Within a live run, execution resumes at the first
incomplete stage. An explicit ``run.process(start="quantize", until="done")``
invalidates the selected stage and all following stage results before rerunning.

Writing a stage
---------------

Register a class from an ``extensions.py`` file::

    from mlonmcu.artifact import Artifact, ArtifactFormat
    from mlonmcu.session.stage import Stage, StageResult, register_stage

    @register_stage("synth", override=True)
    class Synthesis(Stage):
        def validate(self, context):
            if "rtl" not in context.inputs:
                raise ValueError("SYNTH requires an rtl input")

        def execute(self, context):
            artifacts, parents = {}, {}
            source = context.spec["inputs"]["rtl"].split(".", 1)[0]
            for variant, inputs in context.inputs["rtl"].items():
                # Invoke your synthesis tool using inputs[0].path and
                # context.config. Write outputs under context.workdir.
                output = synthesize(inputs, context.workdir, context.config)
                artifacts[variant] = [Artifact(
                    output.name, path=output, fmt=ArtifactFormat.PATH,
                    flags=["netlist"],
                )]
                parents[variant] = (source, variant)
            return StageResult(artifacts=artifacts, parents=parents)

``synthesize`` above is your tool adapter. The same contract supports
``VERILATE`` and ``GENERATE_BITSTREAM``. Register extensions in every execution
environment, including RPC workers. ``override=True`` permits reloading the
same extension when another context opens in the same Python process; omit it
to reject duplicate registration. The same explicit ``override=True`` permits
replacing a default implementation. ``NOP`` and ``DONE`` are reserved sentinels
and cannot be registered as implementations. Spawned process-pool workers load
their environment extensions, including overrides of default names. Extensions
already loaded in a worker or inherited through a fork are not registered again.

The context contains:

* ``run``: the selected model, backend, target, and platforms;
* ``spec`` and ``config``: this stage instance's description and configuration;
* ``inputs``: named mappings of variant names to matching artifacts;
* ``workdir``: a stage-specific directory, created before validation/execution.

A binding such as ``rtl: generate.rtl`` selects artifacts carrying the ``rtl``
flag from an earlier stage. ``generate.*`` selects all its artifacts. Required
inputs must exist and their producer must have completed. Inputs are exported
before custom execution, so tool adapters can use artifact paths.

``StageResult.artifacts`` maps output variant names to artifact lists.
``parents`` maps each output variant to its source ``(stage_id, variant)``;
this is required when inputs exist. Root stages omit parents. Variants may be
renamed, split, or dropped. Each output has one lineage parent; additional named
inputs can supply supporting artifacts. ``metrics`` optionally maps each output
variant to a dictionary of metric names and values. The engine handles metrics
CSV artifacts, inherited metrics, timing, completion, failures, locking, and
export for every implementation. ``StageResult.report`` optionally carries a
transformed report, as used by the registered postprocessing stage.

Artifacts from instances with custom IDs live under ``stages/<id>/``, including when
``run.stage_subdirs`` is disabled. Pipeline definitions are exported as
``pipeline.yml``. IDs must use letters, digits and underscores and start with a
letter. Legacy built-in IDs keep their existing artifact directories. Any
implementation, including a default one, may appear multiple times under distinct
IDs. Component-generated metrics filenames are adapted to the instance ID
without changing the component artifact. Legacy enum endpoints can resolve a
uniquely aliased implementation; use an explicit instance ID when ambiguous.

Default stage bindings and optional stages
------------------------------------------

Default stage logic lives in ``mlonmcu.session.builtin_stages``. Each class
implements the public ``Stage`` contract:

* ``available(run)`` describes required component capabilities;
* ``enabled(run)`` determines whether an optional instance applies;
* ``default_inputs(run, key)`` supplies implicit producer bindings;
* ``validate(context)`` checks stage configuration;
* ``execute(context)`` returns a ``StageResult`` without setting completion state.

The automatically generated default pipeline marks its stages optional, keeping
the existing capability checks and the ``run.tune_enabled`` switch. Explicit
pipeline instances are required unless marked ``optional: true``. A required
stage fails if its capabilities or inputs are missing; it is never silently
skipped. A required ``tune`` instance runs independently of ``run.tune_enabled``
but still requires a backend and tuning platform. Mark optional postprocessing
instances accordingly when no postprocessors are configured. Existing explicit
pipeline descriptions that relied on automatic skipping need these optional
markers.

Inputs default to the nearest enabled earlier instance of the appropriate
implementation. Explicit bindings override these defaults:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Implementation
     - Input names
     - Implicit producer
   * - ``load``
     - None
     - Selected model/frontend
   * - ``tune``
     - ``model``
     - Nearest ``load`` model artifacts
   * - ``build``
     - ``model``, ``records``
     - Nearest enabled ``tune`` records and its model, or nearest ``load`` model
   * - ``compile``
     - ``code``
     - Nearest enabled ``build``, falling back to ``load`` for program frontends
   * - ``run``
     - ``program``, ``deployment``
     - Nearest enabled ``compile``, falling back to ``build``; completed ``flash`` deployment if present
   * - ``postprocess``
     - ``artifacts``
     - Latest completed earlier stage with output variants
   * - ``flash``
     - ``program``
     - Explicit binding required

A compilation input selects the producer's exported code directory per variant.
A program input can select a particular artifact flag, avoiding assumptions about
artifact ordering. Default wildcard program bindings retain the existing
first-artifact convention. Tuning records retain their variant lineage back to
the model. Bind tuning and building to the same transformed model::

    - id: tune_int8
      uses: tune
      inputs: {model: quantize.model}
    - id: generate_int8
      uses: build
      inputs: {model: quantize.model, records: 'tune_int8.*'}
    - id: compile_int8
      uses: compile
      inputs: {code: 'generate_int8.*'}
    - id: measure_int8
      uses: run
      inputs: {program: compile_int8.executable}

The program selector requires the producer to mark its executable with that
flag; use ``compile_int8.*`` to retain the legacy artifact ordering instead.
Stage classes can declare ``input_names`` to reject unsupported bindings.
Custom stages accept arbitrary names unless they declare this set.
``context.source(name)`` returns a producer instance ID, while
``context.require_inputs(...)`` checks required names. ``context.spec`` contains
the effective bindings, including resolved defaults.

Omitted stages are not implicitly inserted. Postprocessing is identified by the
implementation's ``is_postprocess`` flag, including when its instance is renamed.
It must be the final pipeline stage. Sessions retain partial-run postprocessing,
and explicit restarts invalidate downstream artifacts, completion and reports.

Scheduling and hardware resources
---------------------------------

Stage-by-stage scheduling merges compatible pipeline orders and rejects cycles
between run pipelines. Runs without a stage are excluded from its batch.
``--parallel-per-stage QUANTIZE=4`` also accepts custom stage IDs.

Declare an exclusive resource when stages use shared hardware::

    config:
      runs_per_stage: 0
    pipeline:
      stages:
        - id: load
        - id: build
        - id: compile
        - id: flash
          uses: flash
          inputs: {program: 'compile.*'}
          resources: {exclusive: [board_0]}
        - id: run

The optional built-in ``flash`` implementation programs a single firmware
variant using the selected target platform. It captures ``flash.log`` and
supports a configurable ``timeout``. Inputs must contain exactly one ELF, BIN,
BIT, or ``executable``-flagged artifact. It supports platforms using the standard
``TargetPlatform.run`` flash/monitor implementation, such as ESP-IDF and Zephyr;
platforms with custom execution methods need their own stage adapter.

All exclusive resources are locked in stable order for the entire
``run.process`` invocation, across programming and execution. Locks coordinate
processes sharing the host's temporary directory. Resource-bearing pipelines
require ``runs_per_stage=0``; stage-by-stage scheduling would release the board
between programming and execution. All pipelines accessing a shared device
must declare the same resource name. Resources across separate hosts require
external coordination.

``FLASH`` followed by ``RUN`` in the same invocation programs the device once
and then monitors it. A later invocation of ``RUN`` uses the usual programming
behavior because the board state may have changed. Use separate runs for
multiple firmware variants. Stage completion is execution state, not a
persistent cache of device state.

ONNX quantization and TFLite conversion
--------------------------------------

The built-in ``convert`` and ``quantize`` stages require no extension file.
Install optional dependencies in the Python environment running MLonMCU::

    pip install 'mlonmcu[onnx,tflite2onnx]'

``convert`` detects each input artifact's format independently, follows downstream
model bindings through QUANTIZE to the backend's ``supported_formats``, and
preserves formats already accepted by the consumer. TFLite to ONNX uses
``tf2onnx.convert.from_tflite``. Other format changes fail with a conversion-path
error. With no consumer constraint, the input format is preserved. Conflicting
consumer constraints require separate conversion branches.

Override detection or selection with ``input_format`` and ``output_format``
(``onnx`` or ``tflite``). A downstream quantizer's explicit ``input_format`` also
constrains conversion. Thus one pipeline can process mixed ONNX/TFLite runs or
variants without imposing the same conversion on every model.

``quantize`` dispatches by input format and preserves that format. ONNX uses
``onnxruntime.quantization.quantize_dynamic`` for dynamic quantization or
``quantize_static`` with a calibration data reader. ``weight_type``
accepts ONNX Runtime's installed ``QuantType`` names, plus ``int8``/``uint8`` aliases;
actual operator/dtype support is enforced by ONNX Runtime. Operator support and
compatibility with quantized operators depend on the chosen backend. Choose a backend which accepts
ONNX for BUILD, even when the LOAD frontend is TFLite.

For example, use this pipeline in an initializer with your model and components::

    pipeline:
      stages:
        - id: load
        - id: convert
          inputs: {model: load.model}
          config: {opset: 15}
        - id: quantize
          inputs: {model: convert.model}
          config:
            weight_type: QInt8
            per_channel: false
            reduce_range: false
            op_types_to_quantize: [MatMul, Gemm]
        - id: build
          inputs: {model: quantize.model}

For an ONNX source, omit ``convert`` and bind ``quantize`` to ``load.model``.
For conversion alone, omit ``quantize`` and bind BUILD to ``convert.model``.
Bindings are explicit: BUILD's default model source remains LOAD. If using TUNE,
bind both TUNE and BUILD to the same transformed model.

Both stages preserve variants and parent relationships, emit a ``model``-flagged
artifact, and record elapsed time and output size. Optional ``output_name``
must be a filename whose extension matches the selected format. Defaults are ``converted.<format>`` and
``quantized.<format>``. Heavy dependencies load only when the stage executes.
The external-command example above overrides the built-in ``quantize`` if loaded;
use it only when you want that external adapter.


TFLite calibrated quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an existing floating-point TFLite model, install TensorFlow (included in
``mlonmcu[tflite2onnx]``). Set ``representative_data`` to an NPZ file. Keys must
match TFLite input tensor names; each array has shape ``[samples, *input_shape]``
and contains representative input values with the model's input dtype. No
calibration data is synthesized::

    - id: quantize
      inputs: {model: convert.model}
      config:
        representative_data: /path/to/calibration.npz
        input_type: uint8
        output_type: int8
        activation_type: int8
        per_channel: true
        allow_float: false

TFLite defaults to int8 input/output/activations and int8 weights. Dtypes are
resolved through TensorFlow; unsupported combinations raise TensorFlow's error.
``allow_float`` permits fallback for operators without integer kernels.
``reduce_range`` and ``op_types_to_quantize`` apply only to ONNX.
``input_type``, ``output_type``, and ``allow_float`` apply only to TFLite.
``activation_type`` applies to both calibrated paths; incompatible options are errors.
Requantizing an existing flatbuffer uses TensorFlow's internal Calibrator API,
so compatibility should be checked when changing TensorFlow versions. TFLite
weight-only/dynamic quantization is not implemented by this adapter.


Calibration sources and ONNX static/dynamic modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``mode: dynamic`` to select ONNX dynamic quantization, or ``mode: static``
for calibrated quantization. The default ``mode: auto`` selects static when a
calibration source is configured and dynamic otherwise. TFLite supports static
calibration only. For ONNX static quantization, ``quant_format`` defaults to
``QDQ`` (also supports ``QOperator``), ``calibration_method`` defaults to
``MinMax``, and ``activation_type`` defaults to ``QInt8``.

To reuse input data generated by ``validate_new`` or ``gen_data``::

    - id: quantize
      inputs: {model: convert.model}
      config:
        mode: static
        reuse_validation_inputs: true
        weight_type: int8
        activation_type: int8

The stage follows the selected model variant's parent chain to find the nearest
``inputs.npy`` artifact. Each run and variant uses its own inputs; reference
outputs are not used for calibration. The existing validation data format is
adapted to a representative dataset for TensorFlow and to a rewindable
``CalibrationDataReader`` for ONNX Runtime. No new random data is generated.
Missing or empty inputs raise an error rather than falling back to dynamic
quantization. ``representative_data: auto`` is an alias for artifact reuse.

For an explicit source, supply ``representative_data: /path/to/data.npz`` for
either framework, or use a Python reader such as ``scripts/my_data_reader.py``::

    config:
      mode: static
      calibration_data_reader: /path/to/scripts/my_data_reader.py:DummyDataReader
      data_reader_args: {}
      quant_format: QDQ
      calibration_method: MinMax

The reader class is instantiated with ``data_reader_args`` keyword arguments
and must yield named input dictionaries from ``get_next()``, ending with
``None``. Samples are normalized into the same dataset used by both adapters.
Reader paths execute user-supplied Python code in the worker and must be
available there. NPZ and reader sources are mutually exclusive and take
precedence over optional validation input reuse.

Names, shapes, layouts, and dtypes must match the model being calibrated. When
conversion renames inputs, set ``input_name_map: {old_name: new_name}``.
No positional mapping or layout/dtype conversion is guessed. Conversion tools
may require explicit preprocessing beyond renaming. Reusing validation inputs
is convenient for experiments; the calibration quality depends on how well
those samples represent the intended workload.

Tensor layouts at conversion boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ONNX output, ``convert`` can change the external input/output layouts while
preserving the graph's computation. This works for TFLite to ONNX conversion and
for ONNX to ONNX layout changes. Specify tensor names and explicit source/target
layout pairs; layouts are never inferred from tensor dimensions::

    - id: convert
      inputs: {model: load.model}
      config:
        output_format: onnx
        input_layouts:
          serving_default_input:0: [NHWC, NCHW]
        output_layouts:
          StatefulPartitionedCall:0: [NHWC, NCHW]

Use the actual graph boundary names. Both directions are supported for
``NWC``/``NCW``, ``NHWC``/``NCHW``, and ``NDHWC``/``NCDHW``. Unlisted tensors
retain their layouts. The stage changes boundary shapes and inserts transposes;
it does not promise a particular internal layout or optimize the transposes away.
Unknown tensor names and rank mismatches raise errors. TFLite output layout
rewriting is not supported.

Automatic calibration input reuse follows these input layout changes through
the model ancestry. Explicit NPZ files and custom readers must already supply
the converted model's layout. The original LOAD input/reference output artifacts
are preserved; external inference callers and downstream validation fixtures
must use or account for the converted model's input/output layouts. In particular,
changing output layout requires adjusting reference output comparison accordingly.

Enabling stages with the Features API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``quant_onnx`` enables conversion and quantization through the ordinary feature
CLI. No pipeline initializer is needed::

    mlonmcu flow build MODEL --backend BACKEND -f quant_onnx

Replace MODEL/BACKEND with the model and backend configured in your environment.
The backend must accept ONNX and the generated quantized operators. The feature
inserts ``quant_onnx_convert_load`` and ``quant_onnx_load`` after LOAD, then binds
TUNE and BUILD to the quantized model. Conversion preserves ONNX inputs and
converts TFLite inputs to ONNX. The default quantization mode is ``dynamic``.

For static quantization with generated inputs::

    mlonmcu flow build MODEL --backend BACKEND -f quant_onnx -f gen_data \
      -c quant_onnx.mode=static quant_onnx.reuse_validation_inputs=true gen_data.number=20

``-f validate_new`` can replace ``-f gen_data`` when you also want reference
outputs and the backend/platform supports its tensor I/O features. Calibration
only consumes reference inputs. You must supply representative samples suitable
for your model; the feature does not assess calibration quality.

For your own calibration reader::

    mlonmcu flow build MODEL --backend BACKEND -f quant_onnx \
      -c quant_onnx.mode=static \
      -c quant_onnx.calibration_data_reader=/absolute/path/scripts/my_data_reader.py:DummyDataReader

Quantizer options use the ``quant_onnx.`` prefix: ``mode``, ``weight_type``,
``activation_type``, ``per_channel``, ``reduce_range``, ``op_types_to_quantize``,
``representative_data``, ``reuse_validation_inputs``, ``calibration_data_reader``,
``data_reader_args``, ``input_name_map``, ``quant_format``, and
``calibration_method``. Conversion options ``opset``, ``input_layouts``, and
``output_layouts`` use the same prefix. For dictionary options, quote the value::

    -c 'quant_onnx.input_layouts={"input": ["NHWC", "NCHW"]}'

In Python, add ``quant_onnx`` using ``run.add_features_by_name`` with configuration
in the run's config dictionary before processing stages. In initializers, use
``feature_names: [quant_onnx]`` and put ``quant_onnx.*`` options in the run-level
``config`` mapping. Explicit stage configuration continues to work independently.

The feature handles repeated application without duplicating stages. Custom
pipelines get a chain for each LOAD. TUNE/BUILD bindings to another explicit
producer are preserved; absent model bindings and direct LOAD bindings select
the quantized model. Generated IDs include the LOAD ID. Existing generated
stage configurations are preserved on initializer reload. Pipeline-changing
features must be applied before any stage executes.

Extensions implementing a ``RunFeature`` can override
``get_pipeline(description)`` to return an adjusted serializable pipeline.
The run validates the resulting pipeline and initializes completion state for
new stages. Existing run feature compatibility checks still apply.

TFLite quantization feature and mixed inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``quant_tflite`` provides the corresponding feature for calibrated TFLite
quantization::

    mlonmcu flow build MODEL --backend BACKEND -f quant_tflite -f gen_data \
      -c quant_tflite.reuse_validation_inputs=true gen_data.number=20

Choose a TFLite backend available in the environment. Options use the
``quant_tflite.`` prefix: ``mode`` (``static`` or ``auto``), ``weight_type``
(int8), ``input_type``, ``output_type``, ``activation_type``, ``per_channel``,
``allow_float``, ``representative_data``, ``reuse_validation_inputs``,
``calibration_data_reader``, ``data_reader_args``, and ``input_name_map``.
For example, add ``-c quant_tflite.input_type=uint8`` for unsigned input I/O.

These features declare a target format for every model in the run, rather than
filtering out other formats. With ``quant_onnx``, TFLite models are converted and
ONNX models pass through conversion before quantization. With ``quant_tflite``,
TFLite models pass through conversion and ONNX models currently fail with an
unsupported ONNX-to-TFLite conversion-path error. In a session with separate
runs, each run may select its own feature. Enabling both quantization features
in one run raises a configuration error.

When the source already has the target format, conversion preserves the model
bytes unless an explicit layout rewrite was requested. This still emits the
conversion stage's artifact and preserves pipeline ancestry. Quantization is a
separate step and still runs; matching formats do not imply matching quantization
or tensor layouts. Already quantized models are not automatically classified or
silently skipped by these features.
