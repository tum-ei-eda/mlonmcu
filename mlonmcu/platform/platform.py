import tempfile
import multiprocessing
import distutils.util
from pathlib import Path
from filelock import FileLock

from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features
from mlonmcu.feature.type import FeatureType
from mlonmcu.logging import get_logger

logger = get_logger()


class Platform:
    """Abstract platform class."""

    FEATURES = []

    DEFAULTS = {}

    REQUIRED = []

    def __init__(self, name, framework, backend, target, features=None, config=None, context=None):
        self.name = name
        self.framework = framework  # TODO: required? or self.target.framework?
        self.backend = backend
        self.target = target
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)
        self.context = context
        self.artifacts = []

    def set_directory(self, directory):
        raise NotImplementedError

    @property
    def supports_compile(self):
        return False

    @property
    def supports_flash(self):
        return False

    @property
    def supports_monitor(self):
        return False

    def process_features(self, features):
        if features is None:
            return []
        features = get_matching_features(features, FeatureType.PLATFORM)
        for feature in features:
            assert feature.name in self.FEATURES, f"Incompatible feature: {feature.name}"
            feature.add_platform_config(self.name, self.config)
        return features


class CompilePlatform(Platform):
    """Abstract compile platform class."""

    FEATURES = Platform.FEATURES + ["debug"]

    DEFAULTS = {
        **Platform.DEFAULTS,
        "print_output": False,
        "debug": False,
        "build_dir": None,
        "num_threads": multiprocessing.cpu_count(),
    }

    REQUIRED = []

    def __init__(self, name, framework, backend, target, features=None, config=None, context=None):
        super().__init__(
            name,
            framework,
            backend,
            target,
            features=features,
            config=config,
            context=context,
        )
        self.name = name
        self.framework = framework  # TODO: required? or self.target.framework?
        self.backend = backend
        self.target = target
        self.config = config if config else {}
        self.features = self.process_features(features)
        self.config = filter_config(self.config, self.name, self.DEFAULTS, self.REQUIRED)
        self.context = context

    @property
    def supports_compile(self):
        return True

    @property
    def debug(self):
        return bool(self.config["debug"])

    @property
    def num_threads(self):
        return int(self.config["num_threads"])

    @property
    def print_output(self):
        # TODO: get rid of this
        return (
            bool(self.config["print_output"])
            if isinstance(self.config["print_output"], (int, bool))
            else bool(distutils.util.strtobool(self.config["print_output"]))
        )

    def generate_elf(self, src=None, model=None, num=1, data_file=None):
        raise NotImplementedError

    def export_elf(self, path):
        assert len(self.artifacts) > 0, "No artifacts found, please run generate_elf() first"

        if not isinstance(path, Path):
            path = Path(path)
        assert (
            path.is_dir()
        ), "The supplied path does not exists."  # Make sure it actually exists (we do not create it by default)
        for artifact in self.artifacts:
            artifact.export(path)


class TargetPlatform(Platform):
    """Abstract target platform class."""

    FEATURES = Platform.FEATURES + []

    DEFAULTS = {
        **Platform.DEFAULTS,
    }

    REQUIRED = []

    @property
    def supports_flash(self):
        return True

    @property
    def supports_monitor(self):
        return True

    def flash(self, timeout=120):
        raise NotImplementedError

    def monitor(self, timeout=60):
        raise NotImplementedError

    def run(self, timeout=120):
        # Only allow one serial communication at a time
        with FileLock(Path(tempfile.gettempdir()) / "mlonmcu_serial.lock"):

            self.flash(timeout=timeout)
            output = self.monitor(timeout=timeout)

        return output
