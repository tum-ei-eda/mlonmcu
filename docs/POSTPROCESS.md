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

Session postprocesses can also be applied after a flow has finished. The
`postprocess` command reads the saved report from `temp/sessions`, writes the
updated report back by default, and places generated artifacts in the same
directory:

```
mlonmcu postprocess --session 42 config2cols
mlonmcu postprocess --session 42 --output postprocessed.csv filter_cols -c filter_cols.drop=Config
mlonmcu postprocess --session 42 --print-report features2cols
mlonmcu postprocess --session 42 analyse_vivado_reports filter_cols -c analyse_vivado_reports.to_df=1 filter_cols.drop=Comment
```

Omit `--session` to use the latest saved session. Run postprocesses are applied
to every saved run; their input artifacts are restored from each run's
`artifacts.yml` metadata.
Multiple postprocess names are run in the order supplied, making it possible to
add Vivado metrics and then filter or rename the resulting report in one call.

For `analyse_vivado_reports`, enable report columns with
`analyse_vivado_reports.to_df=1`. Use `analyse_vivado_reports.limit` to add
only selected parsed metrics, for example
`-c analyse_vivado_reports.to_df=1 analyse_vivado_reports.limit=place_slice_luts_used,power_total_w`.
It also accepts category shortcuts: `identity`, `timing`, `total_area`,
`area_utilization`, `lut_breakdown`, `bram_breakdown`, `cpu_hierarchy`,
`d_cache_hierarchy`, `i_cache_hierarchy`, `cfu_hierarchy`, `power`, and
`thermal`. For example, `analyse_vivado_reports.limit=identity,timing,total_area`
adds the identity, timing, and total-area Vivado metrics. The pre-existing
`CPU Variant` and `Integrated RAM Size` report columns remain available for the
`identity` view. The `timing` category also adds `Estimated Fmax [MHz]` and,
when `Total Cycles` is available, `Estimated latency at Fmax [ms]`.


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
