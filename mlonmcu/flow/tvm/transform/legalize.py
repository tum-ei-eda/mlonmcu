from contextlib import contextmanager, nullcontext


@contextmanager
def OptionallyDisableLegalize(disableLegalize):
    if not disableLegalize:
        yield nullcontext()
        return
    from tvm.relay.testing.temp_op_attr import TempOpAttr

    def do_not_legalize(attrs, inputs, types):
        return None

    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", do_not_legalize) as denseCtx:
        with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", do_not_legalize) as convCtx:
            yield (denseCtx, convCtx)
