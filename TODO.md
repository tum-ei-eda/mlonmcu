# TODOs

This is a list of pending tasks which did not make it into an own issue.

## Essential

- [x] Pull `mlonmcu-sw` on `mlonmcu init`
- [x] Lint everything
- [ ] Fix `--home` flag (use environments.ini)
- [x] Fix template issue `default -> d`
- [x] Remove hardcoded features/targets from `tasks.py` etc.
- ~~Discussion: split backend wrappers from their implementation~~ (See Issue #23)
- [x] Get default targets running (`host_x86` and `etiss_pulpino`)
- [x] TVM: Split src dir and build dir
- [x] Move cache handling out of CLI module
- [x] Move config/feature processing to another place
- [x] finish mlif class with debug support
- [x] Cleanup non-tflite frontends
- ~~Implement packing/packed features~~ (See Issue #9)
- ~~Replace tensorflow with tflite-micro~~ (See Issue #11)
- ~~MLIF: Replace `ChooseTarget.cmake` by `{target_name}.cmake` or custom file~~ (See tum-ei-eda/mlonmcu-sw#1)
- [x] Try out debugging and data validation
- [x] Complete demo notebook
- [x] Finish TVM backend wrappers
- ~~Discuss how to re-implement existing features (e.g. memplan)~~ (See Issue #9)
- [ ] Fix RuntimeWarning:
    ````
    /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'mlonmcu.flow.tflite.backend.tflmi' found in sys.modules after import of package 'mlonmcu.flow.tflite.backend', but prior to execution of 'mlonmcu.flow.tflite.backend.tflmi'; this may result in unpredictable behaviour
     warn(RuntimeWarning(msg))
    ````

    Discuss if backend implementation which dependents on specific versions of a lib shall be moved from the mlonmcu module to the deps? Call as subprocess instead which would help to split up the tools. Alternatively: submodule
    In the end it would be nice if we could stick with calling tflmc... however how to put in our optional optimizations in then? Possibly via features build into tvm itself which would be harder to maintain...

## General
- [x] Docker Setup
  - [x] CI Images
  - [x] Benchmark Images

## Tests
- [x] Lint tests
- [ ] Add tests: templates
- [ ] Add tests: model lookup

## CLI
- [ ] Implement: `mlonmcu models [filter]`

## Models
- [x] Finalize model metadata format
- [x] Finish model metadata parsing

## Features

- [x] Figure out how to implement autotuning
- [ ] Add postprocess and visualization features

## Targets

- [x] Add QEMU/Spike targets
- ~~Support real hardware?~~ (See Issue #3)

## Logging

- [x] Proper logging to file with level defined in env file


## Setup

- [x] Add `setup.verbose` config to enabled cmake/make stdout
- [x] Fix verbosity of cmake commands

## Docs

### Notebooks

- [x] Finish `Demo.ipynb`
- [x] Add more complex examples
- [ ] Add Google Colab support?
- [x] Visualization notebook

### Sphinx

- [ ] Add more doctrings
- [ ] Add more type annotations

### Misc

- [ ] Finish main README
- [ ] Add table with supported frontends/features/frameworks/backends/targets
- [x] Explain environments.yaml
- [ ] Explain sessions and runs
- [ ] Explain features
- [x] Explain logging
- [ ] Explain Docker
- [ ] Explain Autotuning
- [ ] Best practices/known issues
- [ ] Explain Setup
- [ ] Explain CLI
- [ ] Explain Python APIs
- [ ] Explain CI/CD
- [ ] Explain Cache
- [ ] Explain Models


# Optional

- [ ] Implement own error-types (e.g. `FlowError`, `TaskError`, `SessionError`)?
