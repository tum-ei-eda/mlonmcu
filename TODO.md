# TODOs

This is a list of pending tasks which did not make it into an own issue.

## Essential

- [x] Pull `mlonmcu-sw` on `mlonmcu init`
- [ ] Lint everything
- [ ] Fix `--home` flag (use environments.ini)
- [x] Fix template issue `default -> d`
- [ ] Remove hardcoded features/targets from `tasks.py` etc.
- [ ] Requirements.txt: Freeze vs. Latest?
- [ ] Discussion: split backend wrappers from their implementation
- [ ] Get default targets running (`host_x86` and `etiss_pulpino`)
- [x] TVM: Split src dir and build dir
- [x] Move cache handling out of CLI module
- [x] Move config/feature processing to another place
- [ ] finish mlif class with debug support
- [x] Cleanup non-tflite frontends
- [ ] Implement packing/packed features
- [ ] Replace tensorflow with tflite-micro
- [ ] MLIF: Replace `ChooseTarget.cmake` by `{target_name}.cmake` or custom file
- [ ] MLIF: ensure that cmake tests still work
- [ ] Try out debugging and data validation
- [ ] Complete demo notebook
- [x] Finish TVM backend wrappers

     Hint:

     ```
     --input-shapes "in0:[1,2,3] in1:[4,5,6]"
     ```

- [ ] Discuss how to re-implement existing features (e.g. memplan)
- [ ] Fix RuntimeWarning:
    ````
    /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'mlonmcu.flow.tflite.backend.tflmi' found in sys.modules after import of package 'mlonmcu.flow.tflite.backend', but prior to execution of 'mlonmcu.flow.tflite.backend.tflmi'; this may result in unpredictable behaviour
     warn(RuntimeWarning(msg))
    ````

    Discuss if backend implementation which dependents on specific versions of a lib shall be moved from the mlonmcu module to the deps? Call as subprocess instead which would help to split up the tools. Alternatively: submodule
    In the end it would be nice if we could stick with calling tflmc... however how to put in our optional optimizations in then? Possibly via features build into tvm itself which would be harder to maintain...

## Long-term
- [ ] Fix README Badges
- [ ] Enable GitHub Pages
- [ ] Open Source Release

## General
- [x] Docker Setup
  - [x] CI Images
  - [ ] Benchmark Images

## Tests
- [x] Lint tests
- [ ] Add tests: templates
- [ ] Add tests: model lookup

## CLI

- [ ] Implement: `mlonmcu models [filter]`

## Models
- [ ] Finalize model metadata format
- [ ] Finish model metadata parsing

## Features

- [ ] Figure out how to implement autotuning
- [ ] Add postprocess and visualization features

## Targets

- [ ] Add QEMU/Spike targets
- [ ] Support real hardware?

## Logging

- [x] Proper logging to file with level defined in env file


## Setup

- [x] Add `setup.verbose` config to enabled cmake/make stdout
- [x] Fix verbosity of cmake commands

## Docs

### Notebooks

- [ ] Finish `Demo.ipynb`
- [ ] Add more complex examples
- [ ] Add Google Colab support?
- [ ] Visualization notebook

### Sphinx

- [ ] Add more doctrings
- [ ] Add more type annotations

### Misc

- [ ] Finish main README
  - [ ] Add table with supported frameworks, targets,... and their completion/compatibility state
- [ ] Explain environments.yaml
- [ ] Explain sessions and runs
- [ ] Explain features
- [ ] Explain frameworks/backends
- [ ] Explain targets
- [ ] Explain logging
- [ ] Explain Docker
- [ ] Explain Setup
- [ ] Explain CLI
- [ ] Explain Python APIs
- [ ] Explain CI/CD
- [ ] Explain Cache
- [ ] Explain Models


# Optional

- [ ] Implement own error-types (e.g. `FlowError`, `TaskError`, `SessionError`)?
