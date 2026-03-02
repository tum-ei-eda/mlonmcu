# TODO: license & docstring

import torch


class QuantAddTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        return a + a

    example_input = (torch.rand([13, 3], dtype=torch.float32),)
    can_delegate = True


class QuantAddTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        p = a + a
        q = b + b
        r = p + q
        return p, q, r

    example_input = (
        torch.randn([13, 7, 3], dtype=torch.float32),
        torch.randn([13, 7, 3], dtype=torch.float32),
    )
    can_delegate = True


class QuantOpTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w, x, y, z):
        o1 = w - x
        o2 = o1 + y
        o3 = o2 * z
        return o1, o2, o3

    example_input = (
        torch.randn([3, 1, 2], dtype=torch.float32),
        torch.randn([3, 5, 2], dtype=torch.float32),
        torch.randn([3, 5, 1], dtype=torch.float32) * -0.000001,
        torch.randn([3, 5, 2], dtype=torch.float32) * 1000,
    )
    can_delegate = True


class QuantLinearTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(61, 37)

    def forward(self, x):
        return self.linear(x)

    example_input = (torch.randn([8, 61], dtype=torch.float32),)
    can_delegate = True


MODELS = {
    "qadd": QuantAddTest,
    "qadd2": QuantAddTest2,
    "qops": QuantOpTest,
    "qlinear": QuantLinearTest,
}
