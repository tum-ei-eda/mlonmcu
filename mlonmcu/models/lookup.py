from pathlib import Path
import os
import glob
from itertools import product
import yaml
import logging
from enum import Enum

from .metadata import parse_metadata
from .model import Model, ModelFormat
from .group import ModelGroup

logger = logging.getLogger("mlonmcu")


def get_model_directories(context):
    dirs = context.environment.paths["models"]
    dirs = [d.path for d in dirs]
    return dirs


def find_metadata(directory, model_name=None):
    possible_basenames = ["model", "metadata", "definition"]
    possible_extensions = ["yaml", "yml"]
    directory = Path(directory)
    dirname = directory.name
    if dirname not in possible_basenames:
        possible_basenames.insert(0, dirname)
    if model_name and model_name not in possible_basenames:
        possible_basenames.insert(0, model_name)

    for combination in product(possible_basenames, possible_extensions):
        filename = f"{combination[0]}.{combination[1]}"
        fullpath = directory / filename
        if fullpath.is_file():
            # logger.debug("Found match. Ignoring other files")
            metadata = parse_metadata(fullpath)
            return metadata
    return None


def list_models(directory, depth=1):
    default_frontend = "tflite"  # TODO: configurable?
    if depth != 1:
        raise NotImplementedError  # TODO: implement for arm ml zoo
        # define all allowed extensions + search recusively (with limit?)
        # list(Path(".").rglob(f"*.{ext}")) for ext in allowed_ext
    if not os.path.isdir(directory):
        logger.debug("Not a directory: %s", str(directory))
        return []
    subdirs = [
        Path(directory) / o
        for o in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, o))
    ]
    models = []
    for subdir in subdirs:
        dirname = subdir.name
        if dirname.startswith("."):
            # logger.debug("Skipping hidden directory: %s", str(dirname))
            continue
        frontend = default_frontend
        main_model = subdir / f"{dirname}.{frontend}"
        if os.path.exists(main_model):
            main_model = f"{dirname}/{dirname}"
        else:
            main_model = None
        submodels = []
        for filename in glob.glob(str(subdir / f"*.{frontend}")):
            basename = "".join(Path(filename).name.split(".")[:-1])
            submodels.append(f"{dirname}/{basename}")

        if len(submodels) == 1:
            main_model = submodels[0]

        if main_model:

            submodels.remove(main_model)

            main_base = main_model.split("/")[-1]
            main_metadata = find_metadata(
                Path(directory) / dirname, model_name=main_base
            )

            models.append(
                Model(
                    main_base,
                    Path(directory) / f"{main_model}.{frontend}",
                    alt=main_model,
                    format=ModelFormat.TFLITE,
                    metadata=main_metadata,
                )
            )

        for submodel in submodels:
            sub_base = submodel.split("/")[-1]
            submodel_metadata = find_metadata(
                Path(directory) / dirname, model_name=sub_base
            )
            models.append(
                Model(
                    submodel,
                    Path(directory) / f"{submodel}.{frontend}",
                    format=ModelFormat.TFLITE,
                    metadata=submodel_metadata,
                )
            )

    return models


def list_modelgroups(directory):
    if not os.path.isdir(directory):
        logger.debug("Not a directory: %s", str(directory))
        return []
    groups = []
    directory = Path(directory)
    possible_basenames = ["modelgroups", "groups"]
    possible_extensions = ["yaml", "yml"]
    for combination in product(possible_basenames, possible_extensions):
        filename = f"{combination[0]}.{combination[1]}"
        fullpath = directory / filename
        if fullpath.is_file():
            logger.debug("Found match. Ignoring other files")
            with open(fullpath, "r") as yamlfile:
                try:
                    content = yaml.safe_load(yamlfile)
                    for groupname, groupmodels in content.items():
                        assert isinstance(
                            groupmodels, list
                        ), "Modelgroups should be defined as a YAML list"
                        modelgroup = ModelGroup(groupname, groupmodels)
                        groups.append(modelgroup)
                except yaml.YAMLError as err:
                    raise RuntimeError("Could not open YAML file") from err
            break
    return groups


def lookup_models_and_groups(directories):
    all_models = []
    all_groups = []
    duplicates = {}
    group_duplicates = {}
    for directory in directories:
        models = list_models(directory)
        if len(all_models) == 0:
            all_models = models
        else:
            all_model_names = [m.name for m in all_models]
            for model in models:
                name = model.name
                if name in all_model_names:
                    if name in duplicates:
                        duplicates[name] += 1
                    else:
                        duplicates[name] = 1
                else:
                    all_models.append(model)
        groups = list_modelgroups(directory)
        if len(all_groups) == 0:
            all_groups = groups
        else:
            all_group_names = [g.name for g in all_groups]
            for group in groups:
                name = group.name
                if name in all_group_names:
                    if name in group_duplicates:
                        group_duplicates[name] += 1
                    else:
                        group_duplicates[name] = 1
                else:
                    all_groups.append(group)

    all_models = sorted(all_models, key=lambda x: x.name)

    return all_models, all_groups, duplicates, group_duplicates


def print_paths(directories):
    print("Paths:")
    for directory in directories:
        exists = os.path.exists(directory)
        print("    " + str(directory), end="\n" if exists else " (skipped)\n")
        # if not exists:
        #     directories.remove(directory)
    print()


def print_models(models, duplicates=[], detailed=False):
    print("Models:")
    for model in models:
        name = model.name
        path = model.path
        meta = "available" if model.metadata is not None else "not available"
        has_backend_options = (
            model.metadata and model.metadata.backend_options_map is not None
        )
        print("    " + name, end="")
        if name in duplicates:
            num = duplicates[name]
            print(f" ({num} duplicates)")
        else:
            print()
        if detailed:
            print(f"        Path: {path}")
            print(f"        Metadata: {meta}")
            if has_backend_options:
                backends = " ".join(model.metadata.backend_options_map.keys())
                print(f"        Backend Options: {backends}")
            print()
    print()


def print_groups(groups, all_models=[], duplicates=[], detailed=False):
    print("Groups:")
    for group in groups:
        name = group.name
        models = group.models
        all_model_names = [m.name for m in all_models]
        for model in models:
            if model in all_model_names:
                # Groupname conflicts with modelname
                group.name = "~" + group.name
        size = len(models)
        print(f"    {name} [{size} models]", end="")
        if name in duplicates:
            num = duplicates[name]
            print(f" ({num} duplicates)")
        else:
            print()

        if detailed:
            groupmodels = " ".join(models)
            print(f"        Models: {groupmodels}")
            print()


def print_summary(context, detailed=False):
    directories = get_model_directories(context)

    models, groups, duplicates, group_duplicates = lookup_models_and_groups(directories)

    print("Models Summary\n")
    print_paths(directories)
    print_models(models, duplicates=duplicates, detailed=detailed)
    print_groups(
        groups, duplicates=group_duplicates, all_models=models, detailed=detailed
    )
