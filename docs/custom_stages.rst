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
