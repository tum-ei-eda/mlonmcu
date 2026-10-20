Custom run pipelines
====================

Runs accept an optional ``pipeline`` argument. A session schedules these run
pipelines using the existing executors. Omitting it preserves the default
``LOAD → TUNE → BUILD → COMPILE → RUN → POSTPROCESS`` workflow and its capability
checks. Existing ``RunStage`` arguments remain supported. Custom IDs are
case-insensitive strings; ``DONE`` means the end of the selected pipeline.

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
to reject duplicate registration. Built-in enum stage names cannot be replaced.

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
CSV artifacts, inherited metrics, timing, completion, failures, and export.

Custom artifacts live under ``stages/<id>/``, including when
``run.stage_subdirs`` is disabled. Pipeline definitions are exported as
``pipeline.yml``. IDs must use letters, digits and underscores and start with a
letter. Built-in IDs keep their existing identity. Custom implementations may
appear multiple times under distinct IDs.

Model transformations
---------------------

``BUILD.model`` and ``TUNE.model`` support explicit model bindings. Without a
binding, they retain their existing ``LOAD`` input. When tuning a transformed
model, bind both stages to the same producer::

    - id: tune
      inputs: {model: quantize.model}
    - id: build
      inputs: {model: quantize.model}

``BUILD`` still consumes tuning records from ``TUNE`` and follows its model
lineage. Other built-in stage input bindings are rejected explicitly; custom
stages can consume any named artifact flags. Omitted built-in stages are not
implicitly inserted. ``POSTPROCESS``, if present, must be last and retains the
existing session behavior of processing partial runs.

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
