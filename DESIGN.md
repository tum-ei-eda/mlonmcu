## Important Terms and Design Decisions (RFC)

MLonMCU offers a hand full of high level interfaces as well as a large number of internally used objects. You may use this document as a Glossary to understand the meaning of these some core concepts of the MLonMCU software infrastructure.

### Motivation and Goals

MLonMCU is basically a reimplemented version of a TinyML benchmarking project which was used internally before for about one year.

The open source design was approached with the following set of goals in mind:

- Split up previously used all-time growing Python script into a hierarchical Python package
- Dependency management should be mostly invisible to the user without interfering with other software installed on the system
- In addition to revamping the existing command line interface, a Python API should be integrated to ease scripting and access to intermediate results.
- Increase scalability of large benchmarking tasks by inherently supporting parallelisms in multiple dimensions (Model x Backend x Features x Targets)
- Improve expandability by providing "Plugin" interfaces for various features.
- Ability to support further targets and architecture as well as real Hardware
- Improve Code Quality by adding unit and integration tests and extensive CI/CD applications.
- Provide a common interface to all supported backends by adding wrappers for their Command Line Interfaces
- Offer a large number of examples and extensive documentation to enable the TinyML community to get started with MLonMCU easily
- Ensure reproducibility of research results by improved logging and import options and isolated environments.

### Fundamentals

#### Features and Configuration

Two types of options can be found in a large number of classes in the MLonMCU Package: `features : List[Feature]` and `config : Dict`. This design decision leads to unified command line interfaces and less framework/backend/target/frontend/feature specific code in higher levels of the codebase. A baseline requirement for all classes which implement those two concepts is the definition of the class variables `FEATURES`, `DEFAULTS` and `REQUIRED` .

Learn more about these features here.

#### Contexts and Environments

TODO

#### Session Management/Run Definition

#### Artifacts Handling


TODO

#### Abstraction at Various Levels

Inheritance is used at multiple levels in the MLonMCU project to introduce abstract interfaces for important objects.

Here ere the most relevant examples:

- **Backend:** A backend is a wrapper for a specific code generator
- **Framework:** The used framework is implicitly defined by the backend.
- **Target:** This contains definitions to interface with read hardware or a simulator.
- **Frontend:** Loading and converting models of various types and features is done by the Frontend classes.
- **Feature:** As features are a property of the aforementioned classes, they can have multiple base classes, e.g. `FrameworkFeature`, `TargetFeature`

There are two exceptions to this scheme:

**MLIF class**: The CMake code which is used in the MLonMCU flow could be replaced relatively easily as long the alternative is offering similar command line options or overwrites parts of the MLIF class definition. If this becomes the case, a abstract base class inherited by the new class Wes well as MLIF can be added. The only way why it does not yet exist is because I did not yet came up with a suitable name for this base class. If you have something better than `class Compile` in mind, please raise an issue to discuss your proposal.

**Setup class**: At the current point in time, it is quiet unrealistic, that the current dependency resolution mechanism would be replaced by an alternative tool. However there is at least one options which shall be evaluated in the future (See CK). We than might need to discuss how to rename the currently very generic Name `class Setup` of the original approach by a new name and what the base class should be called.
