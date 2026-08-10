using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using ZstdInflate
using InteractiveUtils

# ----------------------------------------------------------------
# _decode_4streams! codegen properties
#
# Guards against structural regressions to instruction count, call
# density, and register pressure in the hand-tuned SIMD Huffman decode
# loop: lost inlining, new dynamic dispatch, or register pressure
# explosions all show up here.
#
# This is a manual, local-only check, not part of `Pkg.test()`. It reads
# native instruction counts from `code_native`, which reflect the *host
# CPU's* SIMD feature set (AVX2 vs AVX-512, etc.) — GitHub-hosted CI
# runners don't pin a specific CPU, so the same code legitimately compiles
# to different instruction counts across runs. Run this on your own
# machine before/after touching _decode_4streams! or its callees, and
# compare against the baseline noted below (re-measure and update the
# baseline after an intentional change).
#
# Baseline (measured 2026-08-08, sentinel bit-position encoding; see git
# history for the machine/Julia
# version): instrs=1926, calls=30, spills=88.
#
# The signature below must be fully concrete. An abstract `HuffmanTable{11}`
# (or a missing scratch argument) matches no specialisation, `code_native`
# emits nothing, and the check silently degrades to the error below.
# ----------------------------------------------------------------

DataView = ZstdInflate.RBRView
HufTable = ZstdInflate.HuffmanTable{11, Vector{ZstdInflate.HuffmanTableEntry{11}}}
Scratch  = ZstdInflate.Huffman4StreamScratch{DataView}
buf = IOBuffer()
code_native(buf, ZstdInflate._decode_4streams!,
    (DataView, HufTable, Vector{UInt8}, Int, Scratch),
    syntax=:intel, debuginfo=:none)
asm = String(take!(buf))

isempty(asm) && error("specialisation not found — signature may have changed")

lines = split(asm, '\n')
instr_lines = filter(
    l -> occursin(r"^\t", l) && !occursin(r"^\t\.", l) && !occursin(r"^\t#", l),
    lines)

n_instrs = length(instr_lines)
n_calls  = count(l -> occursin(r"\bcall\b", l), instr_lines)
n_spills = count(l -> occursin("-byte Spill", l), instr_lines)

println("_decode_4streams! codegen on this machine:")
println("  instructions: $n_instrs  (baseline 1926)")
println("  calls:        $n_calls  (baseline 30)")
println("  spills:       $n_spills  (baseline 88)")
