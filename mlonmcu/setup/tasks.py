from mlonmcu.setup.task import Task, TaskType
import logging

logger = logging.getLogger('mlonmcu')
logger.setLevel(logging.DEBUG)

class Context:

    def __init__(self):
        self._vars = {"a": 42}

    def __setitem__(self, name, value):
        self._vars[name] = value

    def __getitem__(self, name):
        return self._vars[name]

ctx = Context()

######
# tf #
######

@Task.provides(["tf.src_dir"])
@Task.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(context):
    print("123")
    context["tf.src_dir"] = "qqq"


@Task.needs(["tf.src_dir"])
@Task.provides(["tf.dl_dir", "tf.lib_path"])
@Task.register(category=TaskType.FRAMEWORK)
def build_tensorflow(context):
    context["tf.dl_dir"] = "abc"
    context["tf.lib_path"] = "def"


#########
# tflmc #
#########

@Task.provides(["tflmc.src_dir"])
@Task.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(context):
    context["tflmc.src_dir"] = "qqq"


@Task.needs(["tflmc.src_dir"])
@Task.provides(["tflmc.build_dir", "tflmc.exe"])
@Task.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(context):
    context["tflmc.build_dir"] = "qqq"
    context["tflmc.exe"] = "qqq"


#############
# riscv_gcc #
#############

@Task.provides(["riscv_gcc.install_dir"])
@Task.register(category=TaskType.TOOLCHAIN)
def install_riscv_gcc(context):
    context["riscv_gcc.install_dir"] = "qqq"
    pass


########
# llvm #
########

@Task.provides(["llvm.install_dir"])
@Task.register(category=TaskType.MISC)
def install_llvm(context):
    context["llvm.install_dir"] = "qqq"
    pass


#########
# etiss #
#########

@Task.provides(["etiss.src_dir"])
@Task.register(category=TaskType.TARGET)
def clone_etiss(context):
    context["etiss.src_dir"] = "qqq"


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.build_dir"])
@Task.register(category=TaskType.TARGET)
def build_etiss(context):
    context["etiss.build_dir"] = "qqq"


@Task.needs(["etiss.src_dir"])
@Task.provides(["etiss.install_dir", "etissvp.src_dir"])
@Task.register(category=TaskType.TARGET)
def install_etiss(context):
    context["etiss.install_dir"] = "qqq"
    context["etissvp.src_dir"] = "qqq"


@Task.needs(["etissvp.src_dir"])
@Task.provides(["etissvp.build_dir", "etissvp.exe"])
@Task.register(category=TaskType.TARGET)
def build_etissvp(context):
    context["etissvp.build_dir"] = "qqq"
    context["etissvp.exe"] = "qqq"


#######
# tvm #
#######

@Task.provides(["tvm.src_dir"])
@Task.register(category=TaskType.FRAMEWORK)
def clone_tvm(context):
    context["tvm.src_dir"] = "qqq"


@Task.needs(["tvm.src_dir"])
@Task.provides(["tvm.build_dir", "tvm.lib", "tvm.pythonpath"])
@Task.register(category=TaskType.FRAMEWORK)
def build_tvm(context):
    context["tvm.build_dir"] = "qqq"
    context["tvm.lib"] = "qqq"
    context["tvm.pythonpath"] = "qqq"


##########
# utvmcg #
##########

@Task.provides(["utvmcg.src_dir"])
@Task.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(context):
    context["utvmcg.src_dir"] = "qqq"


@Task.needs(["utvmcg.src_dir"])
@Task.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Task.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(context):
    context["utvmcg.build_dir"] = "qqq"
    context["utvmcg.exe"] = "qqq"


def install_dependencies():
    # print("registry", Task.registry)
    # print("dependencies", Task.dependencies)
    # print("providers", Task.providers)
    V, E = Task.get_graph()
    # print("(V, E)", (V, E))
    order = Task.get_order()
    logger.debug("Determined dependency order: %s" % str(order))

    for task in order:
        func = Task.registry[task]
        func(ctx)

# install_dependencies()
