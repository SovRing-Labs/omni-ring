"""CORTEX-2 step 2 — RENDER: the sift kernel generated into memory at run time (llvmlite / LLVM MCJIT).

"Render" in the field's sense: emit executable code into RAM (a JIT), not a picture. The kernel is specialised to the
program that asked for it (d is a compile-time constant, the phase tables are constants, the host CPU's features are the
target: AVX2 on this laptop, SSE on an OptiPlex — the same IR), compiled once per (d, variant) and cached by hash.
Arithmetic is exactly omniring_sift_cplx's: score[k] = sum_i C8[lat[k,i]]*qr[i] + S8[lat[k,i]]*qi[i], int32.
The hand-written C/AVX2, SSE and C++ kernels are the golden set every generated kernel must match bit-for-bit.
"""
from __future__ import annotations

import ctypes
import hashlib
import time
from typing import Dict, Tuple

import numpy as np

from . import seed

_llvm = None
_cache: Dict[Tuple[int, str], Tuple[object, object, float, tuple]] = {}   # (d, variant) -> (engine, cfunc, s, keepalive)

# O3-equivalent IR pipeline, spelled pass-by-pass because llvmlite dropped the one-call populator.
# Names absent from a given build are skipped (feature drift is common; every name here is optional).
_O3_PASSES = (
    "add_sroa_pass",              # mem2reg: the phase loads become scalars
    "add_ipsccp_pass",
    "add_instruction_combine_pass",
    "add_reassociate_pass",       # reassociate + GVN: expose the inner reduction to codegen
    "add_new_gvn_pass",
    "add_sccp_pass",
    "add_simplify_cfg_pass",
    "add_loop_rotate_pass",       # loop simplification so unroll/LSR can fire
    "add_lcssa_pass",
    "add_loop_simplify_pass",
    "add_loop_unroll_pass",
    "add_loop_strength_reduce_pass",
    "add_instruction_combine_pass",   # tail combine
    "add_aggressive_dce_pass",
)


def _init():
    global _llvm
    if _llvm is None:
        import llvmlite.binding as llvm
        for fn in ("initialize_native_target", "initialize_native_asmprinter", "initialize_native_asmparser"):
            try:
                getattr(llvm, fn)()
            except Exception:
                pass
        _llvm = llvm
    return _llvm


def ir_sift_lut(d: int, stride: int = 0) -> str:
    """JIT-2: the pshufb LUT form — omniring_sift_cplx's instruction mix rendered from IR: the 16-entry C8/S8 tables live in a
    register (pshufb), |lut| x sign-transferred query through pmaddubsw, pairs to int32 through pmaddwd. Same products, same
    int32 sums as the hand kernel (|lut|,|q| <= 127 so pmaddubsw never saturates: 2*127*127 = 32,258)."""
    assert d % 64 == 0
    stride = stride or d          # rows laid out `stride` apart; only the first d phases are scored (cascade prefix, no copy)
    lc = ", ".join(f"i8 {int(v)}" for v in list(seed.C8) * 2)
    ls = ", ".join(f"i8 {int(v)}" for v in list(seed.S8) * 2)
    ones = ", ".join(["i16 1"] * 16)
    half = []
    for h in (0, 1):   # two independent accumulators per row (unroll 2): the dependency chain halves
        half.append(f"""
  %o{h} = add i64 %i, {32 * h}
  %pp{h} = getelementptr i8, ptr %row, i64 %o{h}
  %p{h} = load <32 x i8>, ptr %pp{h}, align 1
  %c{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{lc}>, <32 x i8> %p{h})
  %s{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{ls}>, <32 x i8> %p{h})
  %rp{h} = getelementptr i8, ptr %qr, i64 %o{h}
  %mp{h} = getelementptr i8, ptr %qi, i64 %o{h}
  %r{h} = load <32 x i8>, ptr %rp{h}, align 1
  %m{h} = load <32 x i8>, ptr %mp{h}, align 1
  %ac{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %c{h}, i1 false)
  %as{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %s{h}, i1 false)
  %sr{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %r{h}, <32 x i8> %c{h})
  %sm{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %m{h}, <32 x i8> %s{h})
  %pc{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %ac{h}, <32 x i8> %sr{h})
  %ps{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %as{h}, <32 x i8> %sm{h})
  %wc{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %pc{h}, <16 x i16> <{ones}>)
  %ws{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %ps{h}, <16 x i16> <{ones}>)
  %t{h} = add <8 x i32> %acc{h}, %wc{h}
  %acc{h}n = add <8 x i32> %t{h}, %ws{h}""")
    return f"""
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.abs.v32i8(<32 x i8>, i1)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)

define void @sift_{d}(ptr noalias nocapture readonly %lat, i64 %n, ptr noalias nocapture readonly %qr,
                      ptr noalias nocapture readonly %qi, ptr noalias nocapture %out) {{
entry:
  br label %rows
rows:
  %k = phi i64 [0, %entry], [%k1, %rowend]
  %kn = icmp ult i64 %k, %n
  br i1 %kn, label %rowbody, label %done
rowbody:
  %base = mul i64 %k, {stride}
  %row = getelementptr i8, ptr %lat, i64 %base
  br label %dims
dims:
  %i = phi i64 [0, %rowbody], [%i1, %dims]
  %acc0 = phi <8 x i32> [zeroinitializer, %rowbody], [%acc0n, %dims]
  %acc1 = phi <8 x i32> [zeroinitializer, %rowbody], [%acc1n, %dims]
{half[0]}
{half[1]}
  %i1 = add nuw nsw i64 %i, 64
  %in = icmp ult i64 %i1, {d}
  br i1 %in, label %dims, label %rowend
rowend:
  %accs = add <8 x i32> %acc0n, %acc1n
  %sum = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %accs)
  %op = getelementptr i32, ptr %out, i64 %k
  store i32 %sum, ptr %op, align 4
  %k1 = add nuw nsw i64 %k, 1
  br label %rows
done:
  ret void
}}
"""


def ir_sift_p4(d: int, stride: int = 0) -> str:
    """JIT-4: the lut sift over a 4-bit packed lattice (seed.pack4 "block-64 split": per 64 dims, byte j = phase[64b+j] low
    nibble | phase[64b+32+j] high nibble). One 32-byte load per 64 dims, nibbles split in registers (and 15 / lshr 4), then
    the same pshufb LUT math as ir_sift_lut. stride = row bytes (default d/2); only the first d dims are scored."""
    assert d % 64 == 0
    stride = stride or d // 2
    lc = ", ".join(f"i8 {int(v)}" for v in list(seed.C8) * 2)
    ls = ", ".join(f"i8 {int(v)}" for v in list(seed.S8) * 2)
    ones = ", ".join(["i16 1"] * 16)
    f15 = ", ".join(["i8 15"] * 32)
    f4 = ", ".join(["i8 4"] * 32)
    half = []
    for h in (0, 1):
        half.append(f"""
  %o{h} = add i64 %i, {32 * h}
  %c{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{lc}>, <32 x i8> %p{h})
  %s{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{ls}>, <32 x i8> %p{h})
  %rp{h} = getelementptr i8, ptr %qr, i64 %o{h}
  %mp{h} = getelementptr i8, ptr %qi, i64 %o{h}
  %r{h} = load <32 x i8>, ptr %rp{h}, align 1
  %m{h} = load <32 x i8>, ptr %mp{h}, align 1
  %ac{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %c{h}, i1 false)
  %as{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %s{h}, i1 false)
  %sr{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %r{h}, <32 x i8> %c{h})
  %sm{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %m{h}, <32 x i8> %s{h})
  %pc{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %ac{h}, <32 x i8> %sr{h})
  %ps{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %as{h}, <32 x i8> %sm{h})
  %wc{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %pc{h}, <16 x i16> <{ones}>)
  %ws{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %ps{h}, <16 x i16> <{ones}>)
  %t{h} = add <8 x i32> %acc{h}, %wc{h}
  %acc{h}n = add <8 x i32> %t{h}, %ws{h}""")
    return f"""
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.abs.v32i8(<32 x i8>, i1)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)

define void @sift_{d}(ptr noalias nocapture readonly %lat, i64 %n, ptr noalias nocapture readonly %qr,
                      ptr noalias nocapture readonly %qi, ptr noalias nocapture %out) {{
entry:
  br label %rows
rows:
  %k = phi i64 [0, %entry], [%k1, %rowend]
  %kn = icmp ult i64 %k, %n
  br i1 %kn, label %rowbody, label %done
rowbody:
  %base = mul i64 %k, {stride}
  %row = getelementptr i8, ptr %lat, i64 %base
  br label %dims
dims:
  %i = phi i64 [0, %rowbody], [%i1, %dims]
  %acc0 = phi <8 x i32> [zeroinitializer, %rowbody], [%acc0n, %dims]
  %acc1 = phi <8 x i32> [zeroinitializer, %rowbody], [%acc1n, %dims]
  %bo = lshr i64 %i, 1
  %bp = getelementptr i8, ptr %row, i64 %bo
  %b = load <32 x i8>, ptr %bp, align 1
  %p0 = and <32 x i8> %b, <{f15}>
  %p1 = lshr <32 x i8> %b, <{f4}>
{half[0]}
{half[1]}
  %i1 = add nuw nsw i64 %i, 64
  %in = icmp ult i64 %i1, {d}
  br i1 %in, label %dims, label %rowend
rowend:
  %accs = add <8 x i32> %acc0n, %acc1n
  %sum = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %accs)
  %op = getelementptr i32, ptr %out, i64 %k
  store i32 %sum, ptr %op, align 4
  %k1 = add nuw nsw i64 %k, 1
  br label %rows
done:
  ret void
}}
"""


def ir_sift(d: int, variant: str = "rows") -> str:
    """LLVM IR text for sift_d(lat u8*, n i64, qr i8*, qi i8*, out i32*), d fixed. Written once, emitted per program.
    variant "rows" = scalar IR left to the autovectoriser (CORTEX-2 G2); "lut" = the pshufb LUT form (JIT-2)."""
    if variant == "lut":
        return ir_sift_lut(d)
    if variant == "p4" or variant.startswith("p4@"):    # packed 4-bit lattice; "p4@<row bytes>" for a prefix of wider rows
        return ir_sift_p4(d, int(variant[3:]) if "@" in variant else 0)
    if variant.startswith("lut@"):               # "lut@<stride>": prefix sift over wider rows, read in place
        return ir_sift_lut(d, int(variant[4:]))
    assert d % 32 == 0
    c8 = ", ".join(f"i8 {int(v)}" for v in seed.C8)
    s8 = ", ".join(f"i8 {int(v)}" for v in seed.S8)
    return f"""
@C8 = internal constant [16 x i8] [{c8}], align 16
@S8 = internal constant [16 x i8] [{s8}], align 16

define void @sift_{d}(ptr noalias nocapture readonly %lat, i64 %n, ptr noalias nocapture readonly %qr,
                      ptr noalias nocapture readonly %qi, ptr noalias nocapture %out) {{
entry:
  br label %rows
rows:
  %k = phi i64 [0, %entry], [%k1, %rowend]
  %kn = icmp ult i64 %k, %n
  br i1 %kn, label %rowbody, label %done
rowbody:
  %base = mul i64 %k, {d}
  %row = getelementptr i8, ptr %lat, i64 %base
  br label %dims
dims:
  %i = phi i64 [0, %rowbody], [%i1, %dims]
  %acc = phi i32 [0, %rowbody], [%acc2, %dims]
  %pp = getelementptr i8, ptr %row, i64 %i
  %p = load i8, ptr %pp, align 1
  %pz = zext i8 %p to i64
  %pm = and i64 %pz, 15
  %cp = getelementptr [16 x i8], ptr @C8, i64 0, i64 %pm
  %sp = getelementptr [16 x i8], ptr @S8, i64 0, i64 %pm
  %c = load i8, ptr %cp, align 1
  %s = load i8, ptr %sp, align 1
  %qrp = getelementptr i8, ptr %qr, i64 %i
  %qip = getelementptr i8, ptr %qi, i64 %i
  %r = load i8, ptr %qrp, align 1
  %m = load i8, ptr %qip, align 1
  %c32 = sext i8 %c to i32
  %s32 = sext i8 %s to i32
  %r32 = sext i8 %r to i32
  %m32 = sext i8 %m to i32
  %t1 = mul nsw i32 %c32, %r32
  %t2 = mul nsw i32 %s32, %m32
  %acc1 = add nsw i32 %acc, %t1
  %acc2 = add nsw i32 %acc1, %t2
  %i1 = add nuw nsw i64 %i, 1
  %in = icmp ult i64 %i1, {d}
  br i1 %in, label %dims, label %rowend
rowend:
  %op = getelementptr i32, ptr %out, i64 %k
  store i32 %acc2, ptr %op, align 4
  %k1 = add nuw nsw i64 %k, 1
  br label %rows
done:
  ret void
}}
"""


def compile_sift(d: int, variant: str = "rows"):
    """-> (ctypes function, compile seconds). Cached per (d, variant). Target = this host (features incl. AVX2/SSE)."""
    key = (d, variant)
    if key in _cache:
        _engine, f, t, _keep = _cache[key]
        return f, t
    llvm = _init()
    t0 = time.time()
    ir = ir_sift(d, variant)
    mod = llvm.parse_assembly(ir)
    mod.verify()
    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine(cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(),
                                      opt=3)
    pto = llvm.create_pipeline_tuning_options(speed_level=3)
    pto.loop_vectorization = True
    pto.slp_vectorization = True
    pto.loop_unrolling = True
    pb = llvm.create_pass_builder(tm, pto)
    mpm = pb.getModulePassManager()
    # The pipeline must be spelled out: getModulePassManager() starts empty (llvmlite dropped the one-call populator).
    # Integer-only IR (int32 mul/add) — reassociation is value-exact, so bit-exactness vs the AVX2 kernel is preserved.
    for name in _O3_PASSES:
        add = getattr(mpm, name, None)
        if add is not None:
            add()
    mpm.run(mod, pb)
    engine = llvm.create_mcjit_compiler(mod, tm)
    engine.finalize_object()
    addr = engine.get_function_address(f"sift_{d}")
    proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    f = proto(addr)
    t = time.time() - t0
    # Retain the pass-builder objects: MCJIT holds a raw pointer to the PassBuilder it was given, and letting
    # pto/pb/mpm be collected under a live engine segfaults on the first call into the generated kernel.
    _cache[key] = (engine, f, t, (pto, pb, mpm, tm, target))
    return f, t


_bcache: Dict[Tuple[int, str], Tuple[object, object, float, tuple]] = {}


def ir_batch(d: int, variant: str, stride: int, qstride: int) -> str:
    """JIT-5: the IR-0 schedule as code. batch_d(lat, n, qr_all, qi_all, nq, out, tile, t_begin, t_end): rows are strip-mined
    into tiles of `tile` rows (sized to stay in L2) and the query loop runs INSIDE each tile, so every query reuses the tile
    while it is cache-resident (loop interchange + tiling of the (query x row) nest; no dependences, so legal).
    out is nq x n int32. Tiles [t_begin, t_end) let threads take contiguous tile ranges."""
    return ir_sift(d, variant) + f"""
define void @batch_{d}(ptr %lat, i64 %n, ptr %qra, ptr %qia, i64 %nq, ptr %out, i64 %tile, i64 %tb, i64 %te) {{
entry:
  br label %tiles
tiles:
  %t = phi i64 [%tb, %entry], [%t1, %tend]
  %tl = icmp ult i64 %t, %te
  br i1 %tl, label %tbody, label %done
tbody:
  %r0 = mul i64 %t, %tile
  %rem = sub i64 %n, %r0
  %small = icmp ult i64 %rem, %tile
  %rn = select i1 %small, i64 %rem, i64 %tile
  %lo = mul i64 %r0, {stride}
  %lp = getelementptr i8, ptr %lat, i64 %lo
  br label %qs
qs:
  %q = phi i64 [0, %tbody], [%q1, %qs]
  %qo = mul i64 %q, {qstride}
  %rp = getelementptr i8, ptr %qra, i64 %qo
  %ip = getelementptr i8, ptr %qia, i64 %qo
  %oo = mul i64 %q, %n
  %oo2 = add i64 %oo, %r0
  %op = getelementptr i32, ptr %out, i64 %oo2
  call void @sift_{d}(ptr %lp, i64 %rn, ptr %rp, ptr %ip, ptr %op)
  %q1 = add i64 %q, 1
  %qm = icmp ult i64 %q1, %nq
  br i1 %qm, label %qs, label %tend
tend:
  %t1 = add i64 %t, 1
  br label %tiles
done:
  ret void
}}
"""


def ir_rows8(d: int, stride: int, qstride: int, packed: bool = False) -> str:
    """JIT-5: TILE-1's register blocking rendered from IR. rows8_d(lat, n, qr8, qi8, out, nstride): for each row, every
    32-phase block is loaded and looked up (pshufb C8/S8, abs) ONCE and applied to 8 queries whose accumulators stay in
    registers; out[q * nstride + k]. packed = seed.pack4 rows (one 32-byte load per 64 phases, nibbles split in registers)."""
    assert d % 64 == 0
    lc = ", ".join(f"i8 {int(v)}" for v in list(seed.C8) * 2)
    ls = ", ".join(f"i8 {int(v)}" for v in list(seed.S8) * 2)
    ones = ", ".join(["i16 1"] * 16)
    f15 = ", ".join(["i8 15"] * 32); f4 = ", ".join(["i8 4"] * 32)
    Q = 8
    phis = "\n".join(f"  %a{q} = phi <8 x i32> [zeroinitializer, %rowbody], [%a{q}n, %dims]" for q in range(Q))
    body = []
    if packed:
        body.append(f"""  %bo = lshr i64 %i, 1
  %bp = getelementptr i8, ptr %row, i64 %bo
  %b = load <32 x i8>, ptr %bp, align 1
  %ph0 = and <32 x i8> %b, <{f15}>
  %ph1 = lshr <32 x i8> %b, <{f4}>""")
    else:
        for h in (0, 1):
            body.append(f"""  %po{h} = add i64 %i, {32 * h}
  %pp{h} = getelementptr i8, ptr %row, i64 %po{h}
  %ph{h} = load <32 x i8>, ptr %pp{h}, align 1""")
    cur = {q: f"%a{q}" for q in range(Q)}
    for h in (0, 1):            # one half at a time: 4 lookup registers + 8 accumulators stay within 16 ymm (no spills)
        body.append(f"""  %c{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{lc}>, <32 x i8> %ph{h})
  %s{h} = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> <{ls}>, <32 x i8> %ph{h})
  %ac{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %c{h}, i1 false)
  %as{h} = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %s{h}, i1 false)
  %qo{h} = add i64 %i, {32 * h}""")
        for q in range(Q):
            body.append(f"""  %rq{q}_{h}o = add i64 %qo{h}, {q * qstride}
  %rp{q}_{h} = getelementptr i8, ptr %qr, i64 %rq{q}_{h}o
  %mp{q}_{h} = getelementptr i8, ptr %qi, i64 %rq{q}_{h}o
  %r{q}_{h} = load <32 x i8>, ptr %rp{q}_{h}, align 1
  %m{q}_{h} = load <32 x i8>, ptr %mp{q}_{h}, align 1
  %sr{q}_{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %r{q}_{h}, <32 x i8> %c{h})
  %sm{q}_{h} = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %m{q}_{h}, <32 x i8> %s{h})
  %pc{q}_{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %ac{h}, <32 x i8> %sr{q}_{h})
  %ps{q}_{h} = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %as{h}, <32 x i8> %sm{q}_{h})
  %wc{q}_{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %pc{q}_{h}, <16 x i16> <{ones}>)
  %ws{q}_{h} = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %ps{q}_{h}, <16 x i16> <{ones}>)
  %t{q}_{h} = add <8 x i32> {cur[q]}, %wc{q}_{h}
  %u{q}_{h} = add <8 x i32> %t{q}_{h}, %ws{q}_{h}""")
            cur[q] = f"%u{q}_{h}"
    for q in range(Q):
        body.append(f"  %a{q}n = add <8 x i32> {cur[q]}, zeroinitializer")
    stores = []
    for q in range(Q):
        stores.append(f"""  %sum{q} = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %a{q}n)
  %oq{q} = mul i64 {q}, %ns
  %ok{q} = add i64 %oq{q}, %k
  %op{q} = getelementptr i32, ptr %out, i64 %ok{q}
  store i32 %sum{q}, ptr %op{q}, align 4""")
    body_s = "\n".join(body); stores_s = "\n".join(stores)
    return f"""
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.abs.v32i8(<32 x i8>, i1)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
declare i32 @llvm.vector.reduce.add.v8i32(<8 x i32>)

define void @rows8_{d}(ptr noalias nocapture readonly %lat, i64 %n, ptr noalias nocapture readonly %qr,
                       ptr noalias nocapture readonly %qi, ptr noalias nocapture %out, i64 %ns) {{
entry:
  br label %rows
rows:
  %k = phi i64 [0, %entry], [%k1, %rowend]
  %kn = icmp ult i64 %k, %n
  br i1 %kn, label %rowbody, label %done
rowbody:
  %base = mul i64 %k, {stride}
  %row = getelementptr i8, ptr %lat, i64 %base
  br label %dims
dims:
  %i = phi i64 [0, %rowbody], [%i1, %dims]
{phis}
{body_s}
  %i1 = add nuw nsw i64 %i, 64
  %in = icmp ult i64 %i1, {d}
  br i1 %in, label %dims, label %rowend
rowend:
{stores_s}
  %k1 = add nuw nsw i64 %k, 1
  br label %rows
done:
  ret void
}}

define void @batch8_{d}(ptr %lat, i64 %n, ptr %qra, ptr %qia, i64 %ng, ptr %out, i64 %tile, i64 %tb, i64 %te) {{
entry:
  br label %tiles
tiles:
  %t = phi i64 [%tb, %entry], [%t1, %tend]
  %tl = icmp ult i64 %t, %te
  br i1 %tl, label %tbody, label %fin
tbody:
  %r0 = mul i64 %t, %tile
  %rem = sub i64 %n, %r0
  %small = icmp ult i64 %rem, %tile
  %rn = select i1 %small, i64 %rem, i64 %tile
  %lo = mul i64 %r0, {stride}
  %lp = getelementptr i8, ptr %lat, i64 %lo
  br label %gs
gs:
  %g = phi i64 [0, %tbody], [%g1, %gs]
  %go = mul i64 %g, {8 * qstride}
  %rp = getelementptr i8, ptr %qra, i64 %go
  %ip = getelementptr i8, ptr %qia, i64 %go
  %oo = mul i64 %g, 8
  %oo1 = mul i64 %oo, %n
  %oo2 = add i64 %oo1, %r0
  %op = getelementptr i32, ptr %out, i64 %oo2
  call void @rows8_{d}(ptr %lp, i64 %rn, ptr %rp, ptr %ip, ptr %op, i64 %n)
  %g1 = add i64 %g, 1
  %gm = icmp ult i64 %g1, %ng
  br i1 %gm, label %gs, label %tend
tend:
  %t1 = add i64 %t, 1
  br label %tiles
fin:
  ret void
}}
"""


def compile_batch8(d: int, stride: int, qstride: int, packed: bool):
    key = (d, f"batch8:{stride}:{qstride}:{int(packed)}")
    if key in _bcache:
        return _bcache[key][1]
    llvm = _init()
    mod = llvm.parse_assembly(ir_rows8(d, stride, qstride, packed)); mod.verify()
    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine(cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), opt=3)
    pto = llvm.create_pipeline_tuning_options(speed_level=3)
    pb = llvm.create_pass_builder(tm, pto); mpm = pb.getModulePassManager()
    for name in _O3_PASSES:
        add = getattr(mpm, name, None)
        if add is not None:
            add()
    mpm.run(mod, pb)
    engine = llvm.create_mcjit_compiler(mod, tm); engine.finalize_object()
    i64 = ctypes.c_int64; vp = ctypes.c_void_p
    f = ctypes.CFUNCTYPE(None, vp, i64, vp, vp, i64, vp, i64, i64, i64)(engine.get_function_address(f"batch8_{d}"))
    _bcache[key] = (engine, f, 0.0, (pto, pb, mpm, tm, target))
    return f


def sift_batch8_jit(queries: np.ndarray, lattice: np.ndarray, threads: int = 1, tile: int = 0, packed: bool = False,
                    d_used: int = 0) -> np.ndarray:
    """nq x n int32 == seed.sift_batch, by the register-blocked kernel (8 queries per row load). Queries are padded to a
    multiple of 8 with zero query vectors (they contribute nothing; their rows are dropped)."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    qs = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.uint8)
    n, rb = lat.shape
    d = d_used or (rb * 2 if packed else rb)
    nq = len(qs); ng = -(-nq // 8)
    qr = np.zeros((ng * 8, d), np.int8); qi = np.zeros((ng * 8, d), np.int8)
    for j, q in enumerate(qs):
        qr[j], qi[j] = seed.query_int8(q[:d])
    tile = tile or max(8, (128 * 1024) // rb)       # L2 rule (TILE-1): a tile's rows fill half of a 256 KB L2
    f = compile_batch8(d, rb, d, packed)
    out = np.empty((ng * 8, n), dtype=np.int32)
    nt = -(-n // tile)
    args = (lat.ctypes.data, n, qr.ctypes.data, qi.ctypes.data, ng, out.ctypes.data, tile)
    if threads <= 1:
        f(*args, 0, nt)
    else:
        step = -(-nt // threads)
        futs = [_threads(threads).submit(f, *args, t, min(nt, t + step)) for t in range(0, nt, step)]
        for x in futs:
            x.result()
    return out[:nq]


def compile_batch(d: int, variant: str = "lut", stride: int = 0, qstride: int = 0):
    """-> ctypes fn for batch_d. variant "lut" (u8 rows, stride d) or "p4" (packed rows, stride d/2)."""
    stride = stride or (d // 2 if variant.startswith("p4") else d)
    qstride = qstride or d
    v = variant if "@" in variant else f"{variant}@{stride}"
    key = (d, f"batch:{v}:{qstride}")
    if key in _bcache:
        return _bcache[key][1]
    llvm = _init()
    mod = llvm.parse_assembly(ir_batch(d, v, stride, qstride)); mod.verify()
    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine(cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), opt=3)
    pto = llvm.create_pipeline_tuning_options(speed_level=3); pto.loop_unrolling = True
    pb = llvm.create_pass_builder(tm, pto); mpm = pb.getModulePassManager()
    for name in _O3_PASSES:
        add = getattr(mpm, name, None)
        if add is not None:
            add()
    mpm.run(mod, pb)
    engine = llvm.create_mcjit_compiler(mod, tm); engine.finalize_object()
    i64 = ctypes.c_int64; vp = ctypes.c_void_p
    f = ctypes.CFUNCTYPE(None, vp, i64, vp, vp, i64, vp, i64, i64, i64)(engine.get_function_address(f"batch_{d}"))
    _bcache[key] = (engine, f, 0.0, (pto, pb, mpm, tm, target))
    return f


def sift_batch_jit(queries: np.ndarray, lattice: np.ndarray, threads: int = 1, tile: int = 0, packed: bool = False,
                   d_used: int = 0) -> np.ndarray:
    """nq x n int32 scores, == seed.sift_batch (or the p4 equivalent), by the tiled batch kernel. tile 0 = L2 rule:
    half of a 256 KB L2 for the tile's rows."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    qs = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.uint8)
    n, rb = lat.shape
    d = d_used or (rb * 2 if packed else rb)
    tile = tile or max(8, (128 * 1024) // rb)
    f = compile_batch(d, "p4" if packed else "lut", rb, d)
    qr = np.empty((len(qs), d), np.int8); qi = np.empty((len(qs), d), np.int8)
    for j, q in enumerate(qs):
        qr[j], qi[j] = seed.query_int8(q[:d])
    out = np.empty((len(qs), n), dtype=np.int32)
    nt = -(-n // tile)
    args = (lat.ctypes.data, n, qr.ctypes.data, qi.ctypes.data, len(qs), out.ctypes.data, tile)
    if threads <= 1:
        f(*args, 0, nt)
        return out
    step = -(-nt // threads)
    futs = [_threads(threads).submit(f, *args, t, min(nt, t + step)) for t in range(0, nt, step)]
    for x in futs:
        x.result()
    return out


_pool = None


def _threads(t: int):
    global _pool
    if _pool is None or _pool._max_workers != t:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=t)
    return _pool


def sift_p4_jit(q_idx: np.ndarray, packed: np.ndarray, d_used: int = 0, threads: int = 1) -> np.ndarray:
    """Scores of one phase query against a seed.pack4 lattice (n x row_bytes) on its first d_used dims (default all).
    Same int32 as seed.sift_phase(q[:d_used], unpack4(packed)[:, :d_used])."""
    pk = np.ascontiguousarray(packed, dtype=np.uint8)
    n, rb = pk.shape
    du = d_used or rb * 2
    f, _ = compile_sift(du, f"p4@{rb}")
    qr, qi = seed.query_int8(np.asarray(q_idx[:du], dtype=np.uint8))
    out = np.empty(n, dtype=np.int32)
    _run_rows(f, pk.ctypes.data, n, rb, qr, qi, out, threads)
    return out


def sift_prefix_jit(q_idx: np.ndarray, lattice: np.ndarray, d0: int, threads: int = 1) -> np.ndarray:
    """Cascade coarse pass: scores of the first d0 phases of every row, reading the full-width lattice IN PLACE (no
    lat[:, :d0] copy — the CORTEX-2 SCALE-0 defect). Same int32 as seed.sift_phase(q[:d0], lat[:, :d0])."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    n, d = lat.shape
    f, _ = compile_sift(d0, f"lut@{d}")
    qr, qi = seed.query_int8(np.asarray(q_idx[:d0], dtype=np.uint8))
    out = np.empty(n, dtype=np.int32)
    _run_rows(f, lat.ctypes.data, n, d, qr, qi, out, threads)
    return out


def _run_rows(f, a, n, stride, qr, qi, out, threads):
    b, o, oc = qr.ctypes.data, qi.ctypes.data, out.ctypes.data
    if threads <= 1 or n < 256:
        f(a, n, b, o, oc)
        return
    step = -(-n // threads)
    futs = [_threads(threads).submit(f, a + s * stride, min(step, n - s), b, o, oc + 4 * s) for s in range(0, n, step)]
    for x in futs:
        x.result()


def sift_jit(q_idx: np.ndarray, lattice: np.ndarray, variant: str = "rows", threads: int = 1) -> np.ndarray:
    """Scores of one phase query against every lattice row, by the generated kernel. Same int32 as seed.sift_phase.
    threads > 1 splits the rows into contiguous chunks run in parallel (ctypes drops the GIL inside the kernel)."""
    lat = np.ascontiguousarray(lattice, dtype=np.uint8)
    n, d = lat.shape
    f, _ = compile_sift(d, variant)
    qr, qi = seed.query_int8(q_idx)
    out = np.empty(n, dtype=np.int32)
    if threads <= 1 or n < 256:
        f(lat.ctypes.data, n, qr.ctypes.data, qi.ctypes.data, out.ctypes.data)
        return out
    step = -(-n // threads)
    a, b, o = lat.ctypes.data, qr.ctypes.data, qi.ctypes.data
    oc = out.ctypes.data
    futs = [_threads(threads).submit(f, a + s * d, min(step, n - s), b, o, oc + 4 * s) for s in range(0, n, step)]
    for x in futs:
        x.result()
    return out


def kernel_hash(d: int, variant: str = "rows") -> str:
    return hashlib.sha256(ir_sift(d, variant).encode()).hexdigest()[:16]
