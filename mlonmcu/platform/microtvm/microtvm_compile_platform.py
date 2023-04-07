from pathlib import Path
from typing import Tuple

from mlonmcu.artifact import Artifact, ArtifactFormat

from ..platform import CompilePlatform


class MicroTvmCompilePlatform(CompilePlatform):
    """MicroTVM compile platform class."""

    FEATURES = CompilePlatform.FEATURES + CompilePlatform.FEATURES + []

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
    }

    REQUIRED = CompilePlatform.REQUIRED + []

    def invoke_tvmc_micro_create(self, mlf_path, target=None, list_options=False, force=True):
        all_args = []
        if force:
            all_args.append("--force")
        all_args.append(self.project_dir)
        all_args.append(mlf_path)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("create", *all_args, target=target, list_options=list_options)

    def invoke_tvmc_micro_build(self, target=None, list_options=False, force=False):
        all_args = []
        if force:
            all_args.append("--force")
        all_args.append(self.project_dir)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("build", *all_args, target=target, list_options=list_options)

    def prepare(self, mlf, target):
        out = self.invoke_tvmc_micro_create(mlf, target=target)
        return out

    def compile(self, target):
        out = ""
        # TODO: build with cmake options
        out += self.invoke_tvmc_micro_build(target=target)
        return out

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        src = Path(src) / "default.tar"  # TODO: lookup for *.tar file
        artifacts = []
        out = self.prepare(src, target)
        out += self.compile(target)
        stdout_artifact = Artifact(
            "microtvm_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {}
