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
import pytest
import mock
from mlonmcu.setup.task import get_combs, TaskFactory, TaskType, TaskGraph
from mlonmcu.setup.setup import Setup

TestTaskFactory = TaskFactory()


def _validate_example_task1(context, params={}):
    print("_validate_example_task1", params)
    assert "foo" in params and "bar" in params and "special" in params
    if params["foo"] == 0 or params["bar"] == "B":
        return params["special"]
    return False


@TestTaskFactory.provides(["dep0"])
@TestTaskFactory.param("special", [True, False])
@TestTaskFactory.param("foo", [0, 1])
@TestTaskFactory.param("bar", ["A", "B"])
@TestTaskFactory.validate(_validate_example_task1)
@TestTaskFactory.register(category=TaskType.MISC)
def example_task1(context, params={}, rebuild=False):
    context.cache._vars["dep0"] = ""
    pass


@TestTaskFactory.needs(["dep0"])
@TestTaskFactory.provides(["dep1"])
@TestTaskFactory.register(category=TaskType.MISC)
def example_task2(context, params={}, rebuild=False):
    context.cache._vars["dep1"] = ""
    pass


def test_task_registry():
    assert len(TestTaskFactory.registry) == 2
    assert len(TestTaskFactory.validates) == 1
    assert len(TestTaskFactory.dependencies) == 1
    assert len(TestTaskFactory.providers) == 2
    assert len(TestTaskFactory.types) == 2
    assert len(TestTaskFactory.params) == 2
    assert len(TestTaskFactory.params["example_task1"]) == 3


def test_task_registry_reset_changes():
    TestTaskFactory.changed = ["dep0"]
    TestTaskFactory.reset_changes()
    assert len(TestTaskFactory.changed) == 0


@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("print_output", [False, True])
@pytest.mark.parametrize("rebuild", [False, True])  # TODO: actually test this
@pytest.mark.parametrize("write_cache", [False])  # TODO: True
@pytest.mark.parametrize("write_env", [False])  # TODO: True
def test_setup_install_dependencies(progress, print_output, rebuild, write_cache, write_env, fake_context):
    # example_task1_mock = mock.Mock(return_value=True)
    TestTaskFactory.registry["example_task1"] = mock.Mock(return_value=True)
    # example_task2_mock = mock.Mock(return_value=True)
    TestTaskFactory.registry["example_task2"] = mock.Mock(return_value=True)
    config = {"print_output": print_output}
    installer = Setup(config=config, context=fake_context, tasks_factory=TestTaskFactory)
    result = installer.install_dependencies(
        progress=progress, write_cache=write_cache, write_env=write_env, rebuild=rebuild
    )
    assert result
    # assert example_task1_mock.call_count == 3
    assert (
        TestTaskFactory.registry["example_task1"].call_count == 1
    )  # Due to the mock, the actual wrapper is not executed anymore, params are not considered etc
    # assert example_task2_mock.call_count == 1
    assert TestTaskFactory.registry["example_task2"].call_count == 1


def test_task_get_combs():
    assert get_combs({}) == []
    assert get_combs({"foo": []}) == []  # TODO: invalid?
    assert get_combs({"foo": [0]}) == [{"foo": 0}]
    # assert set(get_combs({"foo": [0, 1, 2]})) == set([{"foo": 0}, {"foo": 1}, {"foo": 2}])
    assert get_combs({"foo": [0, 1, 2]}) == [{"foo": 0}, {"foo": 1}, {"foo": 2}]
    assert len(get_combs({"foo": [0, 1], "bar": ["A", "B"]})) == 4
    assert len(get_combs({"foo": [0, 1], "bar": ["A", "B"]})[0].items()) == 2


def test_task_graph():
    names = ["NodeA", "NodeB", "NodeC"]
    dependencies = {"NodeA": [], "NodeB": ["foo", "bar"], "NodeC": ["foo"]}
    providers = {"foo": "NodeA", "bar": "NodeC"}
    task_graph = TaskGraph(names, dependencies, providers)
    nodes, edges = task_graph.get_graph()
    print("nodes", nodes)
    print("edges", edges)
    assert len(nodes) == len(names)
    assert len(edges) == sum([len(deps) for deps in dependencies.values()])
    order = task_graph.get_order()
    assert len(order) == len(nodes)
    assert order.index("NodeB") > order.index("NodeA") and order.index("NodeB") > order.index("NodeC")
    assert order.index("NodeC") > order.index("NodeA")
