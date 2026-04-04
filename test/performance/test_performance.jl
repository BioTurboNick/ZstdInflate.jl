using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using ZstdInflate
using Chairmarks
using CodecZstd
using CodecZstd: ZstdDecompressor, ZstdDecompressorStream, ZstdCompressorStream
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
function text(n)
    src = read(pathof(ZstdInflate))
    reps = cld(n, length(src))
    return Vector{UInt8}(repeat(String(src), reps)[1:n])
end

# ----------------------------------------------------------------
# Compression helper (CodecZstd)
# ----------------------------------------------------------------

compress_zstd(data::Vector{UInt8}; level=3) =
    read(ZstdCompressorStream(IOBuffer(data), level=level))

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
            results[(size_name, data_name, :codec_zstd, :streaming)] =
                (@b IOBuffer($compressed) read(ZstdDecompressorStream(_)) evals=1 samples=30).time
            GC.gc()
            results[(size_name, data_name, :zstandard, :streaming)] =
                (@b IOBuffer($compressed) read(InflateZstdStream(_)) evals=1 samples=30).time
        end
    end
    return results
end

# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------

function print_results(results, mode)
    mode_str = mode == :in_memory ? "In-memory" : "Streaming"
    println("\n$(mode_str) decompression  (ratio = ZstdInflate.jl / CodecZstd, times in μs):")
    @printf("  %-10s  %-15s  %6s  %10s  %10s\n",
            "size", "data type", "ratio", "CodecZstd", "ZstdInflate.jl")
    println("  ", "-"^57)
    for size_name in [:small, :medium, :large]
        for data_name in [:incompressible, :huffman, :repetitive, :text]
            t_ref  = results[(size_name, data_name, :codec_zstd, mode)]
            t_ours = results[(size_name, data_name, :zstandard,  mode)]
            @printf("  %-10s  %-15s  %6.2f  %10.1f  %10.1f\n",
                    size_name, data_name, t_ours / t_ref,
                    t_ref * 1e6, t_ours * 1e6)
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
        row = "| $(size_name)"
        for data_name in data_names
            t_ref  = results[(size_name, data_name, :codec_zstd, mode)]
            t_ours = results[(size_name, data_name, :zstandard,  mode)]
            row *= @sprintf(" | %.2f×", t_ours / t_ref)
        end
        println(row, " |")
    end
end

results = run_benchmarks()

print_results(results, :in_memory)
print_results(results, :streaming)
print_markdown_table(results, :in_memory)
print_markdown_table(results, :streaming)

