
class CheckAttrs(relay.ExprVisitor):
    def visit_call(self, call):
        if not isinstance(call.op, relay.Function):
            if call.op.name == "nn.contrib_dense_pack":
                attrs = {}
                if call.attrs != None:
                    for attr in call.attrs.keys():
                        value = call.attrs[attr]
                        attrs[attr] = value
                print(attrs)
        super().visit_call(call)
