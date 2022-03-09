# Postprocesses

Postprocesses are a feature which was recently added to MLonMCU which is intended to help with automating common tasks for benchmarking and visualizations. These processes mainly operate on the Rows and columns of the report generated after a completed run or session. Their complexity can vary from very minimal utilities to powerful evaluation scripts.

## Stages

Currently postprocesses can be applied at two different stages:

- after a `Run` which means you are only operating on a single row of a report
- or after a `Session` which has a variety of rows and columns

In the future it might be possible to also insert postprocesses at earlier stages.

## Examples

- `AverageCyclesPostprocess`: average over a number of similar runs (useful for non-deterministic targets)
- `DetailedCyclesPostprocess`: determine the number of cycles required for the model initialization and invocation from two runs with a different `--num` value
- `FlattenConfig`: Turn each dictionary item in the `Config` column into a new Column which makes their values filterable
- `FlattenFeatures`: Convert the list of enabled features into one column per feature with boolean values in every row



## Usage of Postprocesses


### Implement custom postprocesses

To extend the list of predefined post-processes with a custom implementation a Python class has to be developed as follows:

```
TODO

from mlonmcu.postproces import Postprocess, ProcessStage

class MyPostprocess(Postprocess):

    def __init__(self, features=None, config=None):
        super().__init__("foo", stage=ProcessStage.SESSION, features=features, config=config)

    def process(self, data):
        pass
```

It is also possible to inherit another base-class such "AggregatePostprocess" to reuse some of their functionalities.

To use the newly implemented process, it needs to be registered. There are two approaches to do so:

- Call the registration function manually:

  ```
  from mlonmcu.postprocess import register_postprocess

  register_postprocess("foo", MyPostprocess)
  ```

- Use a decorator for the registration. However this option is only available when playing the new postprocess in `mlonmcu/postprocess/postprocesses.py` as the decorators are only processed in this file.

  ```
  @register_postprocess("foo")
  class MyPostprocess(Postprocess):
      ...
  ```

TODO: implement the registration process!
