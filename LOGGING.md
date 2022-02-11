# Logging and Verbosity

## Command-line output

On the CLI-side of MLonMCU two flags are provided to customized the verbosity:

- `--verbose` or `-v` sets the used log level from `INFO` to `DEBUG` for more detailed outputs
- `--quiet` or `-q` sets the used log level from `INFO` to `WARNING` to have minimal information printed to the command line

Of course these flags can **not** be used together!

## Debugging Errors and specific components

MLonMCUs logging output is designed to be very clean so user doe not have to deal with the things going on in the background. However in terms of an error a full stack trace of the exception which was raised is provided to ease debugging. Using the `--verbose` flag sets the loggers level to `DEBUG` which additionally print some useful information on the commands being executed etc. If required, most components feature a config such as `mlif.print_output` or `{backend_name}.print_output` which redirects all of its outputs to the command line.

## Writing log files

Is is possible to additionally write the messages produced by the MLonMCU logger to a log file in a user-specified directory. This feature can be enabled in the `environment.yml` file using the following options:

```
logging:
  level: DEBUG
  to_file: true
  rotate: false

```

By default a directory called `logs` in the environment directory is used, but this can be overwritten by the user itself. Enabling the `rotate` option may be helpful as well as it makes it easier to find logs related to a certain date. The log level configured in the environment file does only affect the logging to the file and not the command line output.


## Background

MLonMCUs logging is based on the `logging` package, however it comes with a set of functions which need to be executed to initialized the logger class. For this reason someone should always use the `get_logger` function from `mlonmcu.logging` instead the official one.
