# Flamegraph of xxhash64 inside Zstandard.jl decompression.
#
# xxhash64 is called once per Zstd frame when the content-checksum flag is set
# (RFC 8878 §3.1.5).  To make it visible in the profile we:
#   1. Compress large data with checksums enabled at several compression levels.
#   2. Warm up all code paths (so JIT compilation doesn't pollute the samples).
#   3. Collect a CPU profile over many decompression iterations.
#   4. Save the flamegraph as an HTML file next to this script.
#
# Run with:
#   julia test/performance/profile_xxhash.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using Profile
using ProfileCanvas
using Random
using ZstdInflate
using CodecZstd.LibZstd

# ----------------------------------------------------------------
# Compression helper (libzstd, checksum enabled)
# ----------------------------------------------------------------

function compress_with_checksum(data::Vector{UInt8}; level=3)::Vector{UInt8}
    cctx = LibZstd.ZSTD_createCStream()
    try
        LibZstd.ZSTD_CCtx_setParameter(cctx, LibZstd.ZSTD_c_compressionLevel, level)
        LibZstd.ZSTD_CCtx_setParameter(cctx, LibZstd.ZSTD_c_checksumFlag, 1)
        out    = Vector{UInt8}(undef, LibZstd.ZSTD_compressBound(length(data)))
        inbuf  = LibZstd.ZSTD_inBuffer_s(pointer(data),  length(data),  0)
        outbuf = LibZstd.ZSTD_outBuffer_s(pointer(out),   length(out),   0)
        LibZstd.ZSTD_compressStream2(cctx, Ref(outbuf), Ref(inbuf), LibZstd.ZSTD_e_end)
        return resize!(out, outbuf.pos)
    finally
        LibZstd.ZSTD_freeCStream(cctx)
    end
end

# ----------------------------------------------------------------
# Test payloads
# ----------------------------------------------------------------

Random.seed!(42)

# Use a large payload so xxhash over the full decompressed output is non-trivial.
# ~4 MB of text-like data (repetitive enough to compress well, large enough for
# the checksum to be visible alongside the decompression work).
src_text  = read(pathof(ZstdInflate), String)
text_4mb  = Vector{UInt8}(repeat(src_text, cld(4_000_000, length(src_text)))[1:4_000_000])

# Random bytes: xxhash runs on the decompressed output, not the compressed input,
# so even incompressible data exercises the hash over the full 4 MB.
rand_4mb  = rand(UInt8, 4_000_000)

payloads = [
    ("text_l3",  compress_with_checksum(text_4mb; level=3)),
    ("text_l19", compress_with_checksum(text_4mb; level=19)),
    ("rand_l3",  compress_with_checksum(rand_4mb; level=3)),
]

# Pre-decompress so we can profile xxhash64 directly on raw bytes.
decompressed = [zstd_decompress(c) for (_, c) in payloads]

# ----------------------------------------------------------------
# Warm-up  (compile all paths before profiling)
# ----------------------------------------------------------------

println("Warming up …")
for data in decompressed
    ZstdInflate.xxhash64(data)
end
GC.gc()

# ----------------------------------------------------------------
# Profile
# ----------------------------------------------------------------

# Enough iterations to get clear samples.  Each call hashes ~4 MB.
const ITERS = 1000

println("Profiling ($ITERS iterations × $(length(decompressed)) payloads) …")
Profile.clear()
@profile for _ in 1:ITERS
    for data in decompressed
        ZstdInflate.xxhash64(data)
    end
end

# ----------------------------------------------------------------
# Save flamegraph
# ----------------------------------------------------------------

out_path = joinpath(@__DIR__, "flamegraph_xxhash.html")
ProfileCanvas.html_file(out_path, Profile.fetch())
println("Flamegraph saved to: $out_path")
