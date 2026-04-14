using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using ZstdInflate
using Chairmarks
using CodecZstd
using CodecZstd: ZstdDecompressor, ZstdDecompressorStream, ZstdCompressorStream
# import Zstandard
using StatsBase
using Random
using Printf

# ----------------------------------------------------------------
# Data generators
# ----------------------------------------------------------------

# Uniformly random bytes — essentially incompressible.
incompressible(n) = rand(UInt8, n)

# Skewed alphabet — Huffman-compressible but not repetitive.
function huffman_compressible(n)
    w = 0.844 .^ (0:255)
    return sample(StatsBase.shuffle(UInt8.(0:255)), Weights(w), n)
end

# Short runs of repeated bytes — exercises back-reference matching.
function repetitive(n)
    x = UInt8[]
    sizehint!(x, n)
    while length(x) < n
        m = min(sample(1:64), n - length(x))
        append!(x, fill(rand(UInt8), m))
    end
    return x
end

# Source text: the ZstdInflate.jl source file repeated to fill n bytes.
#
# The source file is ~62 KB, which shapes three structurally different workloads:
#
#   small  (1 KB)   — 1.6% of one copy.  Output is almost entirely Huffman literals;
#                     CodecZstd's FFI overhead dominates, so we win.
#
#   medium (100 KB) — ~1.6 copies (~61 KB unique + ~39 KB repeat).  The Huffman
#                     decoder handles the full unique first pass (~61% of output),
#                     then sequence execution handles the repeat.  This maximises
#                     the Huffman-decode fraction and is our worst relative case.
#
#   large  (10 MB)  — ~162 copies.  Compressed size is only ~17 KB (≈ one copy +
#                     minimal bookkeeping), so >99% of output comes from sequence
#                     execution (copyto! back-references), which approaches C speeds.
#
# Consequence: performance against CodecZstd tracks Huffman-decode fraction, not
# total output size.  Improving Huffman throughput (e.g. wider table lookups) would
# close the medium-text gap more than any other change.
function text(n)
    src = read(pathof(ZstdInflate))
    reps = cld(n, length(src))
    raw = repeat(src, reps)
    return raw[1:n]
end

# ----------------------------------------------------------------
# Compression helper (CodecZstd)
# ----------------------------------------------------------------

compress_zstd(data::Vector{UInt8}; level=3) =
    read(ZstdCompressorStream(IOBuffer(data), level=level))

# Compress data as multiple concatenated frames of frame_size bytes each.
function compress_zstd_multiframe(data::Vector{UInt8}; level=3, frame_size=100_000)
    out = UInt8[]
    for i in 1:frame_size:length(data)
        chunk = view(data, i:min(i + frame_size - 1, length(data)))
        append!(out, read(ZstdCompressorStream(IOBuffer(chunk), level=level)))
    end
    return out
end

# ----------------------------------------------------------------
# Benchmark runner
# ----------------------------------------------------------------

const sizes = Dict(:small => 1_000, :medium => 100_000, :large => 10_000_000)
const generators = Dict(:incompressible => incompressible,
                        :huffman        => huffman_compressible,
                        :repetitive     => repetitive,
                        :text           => text)

function run_benchmarks()
    results = Dict{Any, Float64}()
    for (size_name, n) in sizes
        for (data_name, gen) in generators
            Random.seed!(42)
            x = gen(n)
            compressed = compress_zstd(x)

            GC.gc()
            results[(size_name, data_name, :codec_zstd, :in_memory)] =
                (@b transcode(ZstdDecompressor, $compressed) seconds=2).time
            GC.gc()
            results[(size_name, data_name, :zstandard, :in_memory)] =
                (@b inflate_zstd($compressed) seconds=2).time
            GC.gc()
            #results[(size_name, data_name, :zstandard_jl, :in_memory)] =
            #    (@b Zstandard.decompress($compressed) seconds=2).time
            results[(size_name, data_name, :zstandard_jl, :in_memory)] = NaN
            GC.gc()
            results[(size_name, data_name, :codec_zstd, :streaming)] =
                (@b IOBuffer($compressed) read(ZstdDecompressorStream(_)) evals=1 samples=30).time
            GC.gc()
            results[(size_name, data_name, :zstandard, :streaming)] =
                (@b IOBuffer($compressed) read(InflateZstdStream(_)) evals=1 samples=30).time
            GC.gc()
            #results[(size_name, data_name, :zstandard_jl, :streaming)] =
            #    (@b IOBuffer($compressed) read(Zstandard.ZstdDecompressorStream(_)) evals=1 samples=30).time
            results[(size_name, data_name, :zstandard_jl, :streaming)] = NaN
        end
    end
    return results
end

# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------

const CATEGORY_DESCRIPTIONS = Dict(
    :incompressible => "random bytes; exercises raw-block path",
    :huffman        => "skewed alphabet; exercises Huffman literals",
    :repetitive     => "short byte runs; exercises back-references",
    :text           => "Julia source repeated; mixed Huffman + back-refs, ratio varies by input size",
)

function print_results(results, mode)
    mode_str = mode == :in_memory ? "In-memory" : "Streaming"
    println("\n$(mode_str) decompression  (ratios vs CodecZstd, times in μs):")
    println()
    println("  Categories:")
    for data_name in [:incompressible, :huffman, :repetitive, :text]
        @printf("    %-15s  %s\n", data_name, CATEGORY_DESCRIPTIONS[data_name])
    end
    println()
    @printf("  %-10s  %-15s  %6s  %6s  %10s  %10s  %10s\n",
            "size", "data type", "ZstdI.", "Zstd.jl", "CodecZstd", "ZstdInflate.jl", "Zstandard.jl")
    println("  ", "-"^79)
    for size_name in [:small, :medium, :large]
        for data_name in [:incompressible, :huffman, :repetitive, :text]
            t_ref    = results[(size_name, data_name, :codec_zstd,    mode)]
            t_ours   = results[(size_name, data_name, :zstandard,     mode)]
            t_zstdjl = results[(size_name, data_name, :zstandard_jl,  mode)]
            @printf("  %-10s  %-15s  %6.2f  %6.2f  %10.1f  %10.1f  %10.1f\n",
                    size_name, data_name,
                    t_ours / t_ref, t_zstdjl / t_ref,
                    t_ref * 1e6, t_ours * 1e6, t_zstdjl * 1e6)
        end
    end
end

function print_markdown_table(results, mode)
    mode_str = mode == :in_memory ? "In-memory" : "Streaming"
    deps = Pkg.API.Context().env.manifest.deps
    version = only(filter(x -> x.name == "ZstdInflate",
                          collect(values(deps)))).version
    data_names = [:incompressible, :huffman, :repetitive, :text]

    println("\n| ZstdInflate.jl v$(version) — $(mode_str) | incompressible | huffman | repetitive | text |")
    println("| --- | --- | --- | --- | --- |")
    for size_name in [:small, :medium, :large]
        row = "| $(size_name) (ZstdInflate.jl)"
        for data_name in data_names
            t_ref  = results[(size_name, data_name, :codec_zstd, mode)]
            t_ours = results[(size_name, data_name, :zstandard,  mode)]
            row *= @sprintf(" | %.2f×", t_ours / t_ref)
        end
        println(row, " |")
        row = "| $(size_name) (Zstandard.jl)"
        for data_name in data_names
            t_ref    = results[(size_name, data_name, :codec_zstd,   mode)]
            t_zstdjl = results[(size_name, data_name, :zstandard_jl, mode)]
            row *= @sprintf(" | %.2f×", t_zstdjl / t_ref)
        end
        println(row, " |")
    end
end

results = run_benchmarks()

print_results(results, :in_memory)
print_results(results, :streaming)
print_markdown_table(results, :in_memory)
print_markdown_table(results, :streaming)

# ----------------------------------------------------------------
# Multi-frame benchmark: large Huffman data split into 100-KB frames
# ----------------------------------------------------------------
println("\nMulti-frame decompression (large huffman, 100 × 100 KB frames):")
println()
Random.seed!(42)
let data = huffman_compressible(10_000_000)
    mf = compress_zstd_multiframe(data)
    sf = compress_zstd(data)
    nframes = div(length(data), 100_000)
    nt = Threads.nthreads()
    nt ≥ 4 || @warn "Expected ≥ 4 Julia threads for parallel benchmark; got $nt. Run with: julia -t 4 ..."
    npar = min(nt, 4)
    @printf("  %-25s  %d frames, %d bytes compressed\n", "multi-frame", nframes, length(mf))
    @printf("  %-25s  %d (using %d for parallel benchmark)\n", "Julia threads", nt, npar)
    println()
    GC.gc()
    t_ref_sf   = (@b transcode(ZstdDecompressor, $sf)          seconds=2).time
    t_ref_mf   = (@b transcode(ZstdDecompressor, $mf)          seconds=2).time
    t_ours_sf  = (@b inflate_zstd($sf)                         seconds=2).time
    t_ours_mf  = (@b inflate_zstd($mf; nthreads=1)             seconds=2).time
    t_ours_par = (@b inflate_zstd($mf; nthreads=$npar)         seconds=2).time
    @printf("  %-25s  ratio=%.2f  CodecZstd=%6.1f μs  ZstdInflate.jl=%6.1f μs\n",
            "single-frame",    t_ours_sf/t_ref_sf, t_ref_sf*1e6, t_ours_sf*1e6)
    @printf("  %-25s  ratio=%.2f  CodecZstd=%6.1f μs  ZstdInflate.jl=%6.1f μs\n",
            "multi-frame serial",  t_ours_mf/t_ref_mf, t_ref_mf*1e6, t_ours_mf*1e6)
    @printf("  %-25s  ratio=%.2f  CodecZstd=%6.1f μs  ZstdInflate.jl=%6.1f μs\n",
            "multi-frame $(npar)-thread", t_ours_par/t_ref_mf, t_ref_mf*1e6, t_ours_par*1e6)
    if t_ours_mf > 0
        @printf("\n  Parallel speedup vs serial:  %.2f×\n", t_ours_mf / t_ours_par)
    end
end

