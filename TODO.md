# TODOs

This is a list of pending tasks which did not make it into an own issue.

- [ ] Fix RuntimeWarning:
    ````
    /usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'mlonmcu.flow.tflite.backend.tflmi' found in sys.modules after import of package 'mlonmcu.flow.tflite.backend', but prior to execution of 'mlonmcu.flow.tflite.backend.tflmi'; this may result in unpredictable behaviour
     warn(RuntimeWarning(msg))
    ````

    Discuss if backend implementation which dependents on specific versions of a lib shall be moved from the mlonmcu module to the deps? Call as subprocess instead which would help to split up the tools. Alternatively: submodule
    In the end it would be nice if we could stick with calling tflmc... however how to put in our optional optimizations in then? Possibly via features build into tvm itself which would be harder to maintain...

- [ ] Add more tests

- [ ] Implement: `mlonmcu models [filter]` on cli

- [ ] Autotuning for ETISS targets

- [ ] IPYNB: Add Google Colab support?

- [ ] Add more doctrings

- [ ] Add more type annotations

- [ ] Finish main README

- [ ] Explain sessions and runs

- [ ] Explain features

- [ ] Explain Docker

- [ ] Explain Autotuning

- [ ] Best practices/known issues

- [ ] Explain Setup

- [ ] Explain CLI

- [ ] Explain Python APIs

- [ ] Explain CI/CD

- [ ] Explain Cache

- [ ] Explain Models
