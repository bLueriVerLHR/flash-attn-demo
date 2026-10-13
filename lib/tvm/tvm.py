import numpy as np
import tvm
from tvm.script import tirx as T

@tvm.script.ir_module
class MyModule:
    @T.prim_func(s_tir=True)
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"),
                 B: T.Buffer((16, 128, 128), "float32"),
                 C: T.Buffer((16, 128, 128), "float32")):
        T.func_attr({
            "tirx.noalias": True
        })

        Y = T.alloc_buffer((16, 128, 128), "float32")
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.sblock("Y"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] += A[vn, vi, vk] * B[vn, vk, vj]
        for n, i, j in T.grid(16, 128, 128):
            with T.sblock("C"):
                vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], T.float32(0))


sch = tvm.s_tir.Schedule(MyModule)
block_Y = sch.get_sblock("Y", func_name="bmm_relu")
yn, _, yj, yk = sch.get_loops(block_Y)
yj0, yj1 = sch.split(yj, factors=[None, 8])

block_C = sch.get_sblock("C", func_name="bmm_relu")
sch.reverse_compute_at(block_C, yj0)

sch.parallel(yn)
block_Y_init = sch.decompose_reduction(block_Y, yj1)

_, _, _, cj1 = sch.get_loops(block_C)
sch.vectorize(cj1)

_, _, _, yij1 = sch.get_loops(block_Y_init)
sch.vectorize(yij1)

sch.reorder(yk, yj1)
yk0, yk1 = sch.split(yk, factors=[None, 4])
sch.unroll(yk1)

import IPython
print(IPython.display.Code(sch.mod.script(), language="python"))