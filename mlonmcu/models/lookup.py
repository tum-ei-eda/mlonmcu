#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pathlib import Path
import os
import glob
from itertools import product
import yaml

from .model import Model, ModelFormats
from .group import ModelGroup

from mlonmcu.logging import get_logger

logger = get_logger()


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
            return fullpath
            # logger.debug("Found match. Ignoring other files")
    return None


def list_models(directory, depth=1, formats=None, config=None):  # TODO: get config from environment or cmdline!
    config = config if config is not None else {}
    formats = formats if formats else [ModelFormats.TFLITE]
    assert len(formats) > 0, "No formats provided for model lookup"
    models = []
    for fmt in formats:
        if depth != 1:
            raise NotImplementedError  # TODO: implement for arm ml zoo
            # define all allowed extensions + search recusively (with limit?)
            # list(Path(".").rglob(f"*.{ext}")) for ext in allowed_ext
        if not os.path.isdir(directory):
            logger.debug("Not a directory: %s", str(directory))
            return []
        subdirs = [Path(directory) / o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))]
        for subdir in subdirs:
            dirname = subdir.name
            if dirname.startswith("."):
                # logger.debug("Skipping hidden directory: %s", str(dirname))
                continue
            exts = fmt.extensions
            for ext in exts:
                main_model = subdir / f"{dirname}.{ext}"
                if os.path.exists(main_model):
                    main_model = f"{dirname}/{dirname}"
                else:
                    main_model = None
                submodels = []
                for filename in glob.glob(str(subdir / f"*.{ext}")):
                    basename = "".join(Path(filename).name.split(".")[:-1])
                    submodels.append(f"{dirname}/{basename}")

                if len(submodels) == 1:
                    main_model = submodels[0]

                if main_model:
                    submodels.remove(main_model)

                    main_base = main_model.split("/")[-1]
                    main_config = {}
                    main_metadata_path = find_metadata(Path(directory) / dirname, model_name=main_base)
                    if main_metadata_path:
                        main_config[f"{main_base}.metadata_path"] = main_metadata_path
                    main_config.update(config)

                    models.append(
                        Model(
                            main_base,
                            [Path(directory) / f"{main_model}.{ext}"],
                            config=main_config,
                            alt=main_model,
                            formats=[fmt],
                        )
                    )

                for submodel in submodels:
                    sub_base = submodel.split("/")[-1]
                    submodel_config = {}
                    submodel_metadata_path = find_metadata(Path(directory) / dirname, model_name=sub_base)
                    if submodel_metadata_path:
                        submodel_config[f"{sub_base}.metadata_path"] = submodel_metadata_path
                    submodel_config.update(config)
                    models.append(
                        Model(
                            submodel,
                            [Path(directory) / f"{submodel}.{ext}"],
                            config=submodel_config,
                            formats=[fmt],
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
                        assert isinstance(groupmodels, list), "Modelgroups should be defined as a YAML list"
                        modelgroup = ModelGroup(groupname, groupmodels)
                        groups.append(modelgroup)
                except yaml.YAMLError as err:
                    raise RuntimeError("Could not open YAML file") from err
            break
    return groups


def lookup_models_and_groups(directories, formats, config=None):
    all_models = []
    all_groups = []
    duplicates = {}
    group_duplicates = {}
    for directory in directories:
        models = list_models(directory, formats=formats, config=config)
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
        path = model.paths[0]
        meta = "available" if model.metadata is not None else "not available"
        print("    " + name, end="")
        if name in duplicates:
            num = duplicates[name]
            print(f" ({num} duplicates)")
        else:
            print()
        if detailed:
            print(f"        Path: {path}")
            print(f"        Metadata: {meta}")
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
    # TODO: get from context!
    formats = [ModelFormats.TFLITE]

    directories = get_model_directories(context)

    models, groups, duplicates, group_duplicates = lookup_models_and_groups(directories, formats)

    print("Models Summary\n")
    print_paths(directories)
    print_models(models, duplicates=duplicates, detailed=detailed)
    print_groups(groups, duplicates=group_duplicates, all_models=models, detailed=detailed)


def lookup_models(names, frontends=None, config=None, context=None):
    if frontends is None:
        assert context is not None
        # TODO: Get defaults frontends from environment (with no config/features)
        raise NotImplementedError
        # frontends = ?
    # allowed_ext = [frontend.fmt.extension for frontend in frontends]
    allowed_fmts = list(dict.fromkeys(sum([frontend.input_formats for frontend in frontends], [])))  # Remove duplicates
    allowed_exts = [fmt.extension for fmt in allowed_fmts]  # There should nit be duplicates

    if context:
        directories = get_model_directories(context)
        models, _, _, _ = lookup_models_and_groups(directories, allowed_fmts, config=config)
    else:
        models = []
    model_names = [model.name for model in models]

    hints = []
    for name in names:
        filepath = Path(name)
        real_name = filepath.stem
        ext = filepath.suffix[1:]
        if len(ext) > 0 and filepath.is_file():  # Explicit file
            assert (
                ext in allowed_exts
            ), f"Unsupported file extension for model which was explicitly passed by path: {ext}"
            paths = [filepath]
            metadata_path = find_metadata(filepath.parent.resolve(), model_name=real_name)
            if metadata_path:
                config[f"{real_name}.metadata_path"] = metadata_path
            hint = Model(real_name, paths, config=config, formats=[ModelFormats.from_extension(ext)])
            # TODO: look for metadata
            hints.append(hint)
        else:
            assert context is not None, "Context is required for passing models by name"
            assert len(model_names) > 0, "List of available models is empty"
            assert name in model_names, f"Could not find a model matching the name: {name}"
            index = model_names.index(name)
            hint = models[index]
            hints.append(hint)

    return hints


def apply_modelgroups(models, context=None):
    assert context is not None
    directories = get_model_directories(context)

    groups = {}
    for directory in directories:
        for group in list_modelgroups(directory):
            if group.name in groups and groups[group.name] != group.models:
                raise RuntimeError(
                    f"The model group '{group.name}' has conflicting definitions. Used the following directories:"
                    f" {directories}"
                )
            groups[group.name] = group.models

    out_models = []
    for model in models:
        out_models.extend(groups.get(model, [model]))
    return list(dict.fromkeys(out_models))  # Drop duplicates


def map_frontend_to_model(model, frontends, backend=None):
    model_fmts = model.formats
    backend_fmts = backend.get_supported_formats() if backend else []
    for frontend in frontends:
        ins = frontend.input_formats
        outs = frontend.output_formats
        assert len(ins) > 0
        # TODO: instead of picking the first frontend which at least one match, determine the highest overlap
        if any(in_fmt in model_fmts for in_fmt in ins):
            if len(backend_fmts) == 0 or any(out_fmt in backend_fmts for out_fmt in outs):
                new_model = model  # TODO: deepcopy?
                for i, fmt in enumerate(model_fmts):
                    if fmt not in ins:  # Only keep those formqts which are supported by the chosen frontend
                        path = model.paths[model_fmts.index(fmt)]
                        new_model.formats.remove(fmt)
                        new_model.paths.remove(path)
                return new_model, frontend
    raise RuntimeError(f"Unable to find a suitable frontend for model '{model.name}'")
