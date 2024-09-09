import sys

import tvm

# from tvm.relay import transform
from tvm import relay

# from tvm import parser

assert len(sys.argv) in [2, 3], "Invalid number of arguments"

with open(sys.argv[1]) as f:
    text = f.read()

ir_mod = relay.fromtext(text)

with tvm.transform.PassContext():
    ir_mod = relay.transform.InferType()(ir_mod)

text = ir_mod.astext()
if len(sys.argv) == 3:
    with open(sys.argv[2], "w") as f:
        f.write(text)
else:
    print(text)
