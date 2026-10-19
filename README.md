# flash-attn-demo

Ampere CuTe attention，执行顺序对应 CUDA Kernel 2。当前支持 CUDA FP16、non-causal、
无 mask/dropout、等长 Q/K/V、D=64/128、L 为64倍数；没有 backward。

## 运行

```bash
python -m pip install -r requirements-cli.txt

python -m example.main --help
python -m example.main --preset tiny \
  --backend cute-ffi --backend prepared-ffi --backend cuda2

python -m example.main --shape 1,192,4,128,2 \
  --backend cute-ffi --backend cuda2 --mode graph --mode wall
```