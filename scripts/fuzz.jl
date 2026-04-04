# Fuzz/differential testing for Zstandard.jl
#
# This script is NOT part of the regular test suite. Run it with:
#   julia --project scripts/fuzz.jl [seed]
#
# It performs two kinds of checks:
#
#   Part 1 — Round-trip: Compress a wide variety of data with CodecZstd, then
#             decompress with both CodecZstd and Zstandard.jl.  All outputs must
#             match.
#
#   Part 2 — Adversarial: Feed random and semi-structured byte sequences to both
#             decompressors.  Two invariants must hold:
#               • If CodecZstd raises an error, Zstandard must also raise an error.
#               • If CodecZstd succeeds, Zstandard must succeed with the same bytes.
#
# Exit code 0 → all invariants satisfied.
# Exit code 1 → one or more invariants violated (details printed to stdout).

using Random
using Zstandard
using CodecZstd: ZstdCompressorStream, ZstdDecompressorStream
using CodecZstd.LibZstd

const ZSTD_MAGIC = UInt8[0x28, 0xB5, 0x2F, 0xFD]
const SKIP_MAGIC  = UInt8[0x50, 0x2A, 0x4D, 0x18]   # one variant; 0x50–0x5F valid

seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 42
rng  = MersenneTwister(seed)
println("=== Zstandard.jl fuzz script  seed=$seed ===\n")

# ============================================================
# Infrastructure
# ============================================================

pass_count      = Ref(0)
fail_count      = Ref(0)   # Zstandard missed an error that CodecZstd raised
mismatch_count  = Ref(0)   # both succeeded but outputs differ
zstd_only_error = Ref(0)   # Zstandard errored but CodecZstd succeeded (informational)
failures        = String[]

function compress_data(data::Vector{UInt8}; level=3, checksum=false)::Vector{UInt8}
    cctx = LibZstd.ZSTD_createCStream()
    try
        LibZstd.ZSTD_CCtx_setParameter(cctx, LibZstd.ZSTD_c_compressionLevel, level)
        checksum && LibZstd.ZSTD_CCtx_setParameter(cctx, LibZstd.ZSTD_c_checksumFlag, 1)
        out    = Vector{UInt8}(undef, LibZstd.ZSTD_compressBound(length(data)))
        inbuf  = LibZstd.ZSTD_inBuffer_s(pointer(data), length(data), 0)
        outbuf = LibZstd.ZSTD_outBuffer_s(pointer(out),  length(out),  0)
        LibZstd.ZSTD_compressStream2(cctx, Ref(outbuf), Ref(inbuf), LibZstd.ZSTD_e_end)
        return resize!(out, outbuf.pos)
    finally
        LibZstd.ZSTD_freeCStream(cctx)
    end
end

# Returns (:ok, bytes) or (:error, exception).
function try_codec_decompress(data::Vector{UInt8})
    try
        result = read(ZstdDecompressorStream(IOBuffer(data)))
        return (:ok, result)
    catch e
        return (:error, e)
    end
end

function try_zstd_decompress(data::Vector{UInt8})
    try
        return (:ok, zstd_decompress(data))
    catch e
        return (:error, e)
    end
end

function try_zstd_stream(data::Vector{UInt8})
    try
        return (:ok, read(ZstandardStream(IOBuffer(data))))
    catch e
        return (:error, e)
    end
end

# Check a single (label, compressed-bytes) pair against both decompressors.
function check_decompress(label::String, data::Vector{UInt8})
    codec_res   = try_codec_decompress(data)
    zstd_res    = try_zstd_decompress(data)
    stream_res  = try_zstd_stream(data)

    codec_ok  = codec_res[1]  === :ok
    zstd_ok   = zstd_res[1]  === :ok
    stream_ok = stream_res[1] === :ok

    ok = true

    if codec_ok
        # Both Zstandard APIs must also succeed and match.
        expected = codec_res[2]::Vector{UInt8}

        if !zstd_ok
            msg = "[$label] CodecZstd OK but zstd_decompress errored: $(zstd_res[2])"
            push!(failures, msg)
            println("  FAIL (zstd error): $msg")
            zstd_only_error[] += 1
            ok = false
        elseif zstd_res[2] != expected
            msg = "[$label] zstd_decompress output mismatch (got $(length(zstd_res[2])) bytes, want $(length(expected)))"
            push!(failures, msg)
            println("  FAIL (mismatch):  $msg")
            mismatch_count[] += 1
            ok = false
        end

        if !stream_ok
            msg = "[$label] CodecZstd OK but ZstandardStream errored: $(stream_res[2])"
            push!(failures, msg)
            println("  FAIL (stream error): $msg")
            zstd_only_error[] += 1
            ok = false
        elseif stream_res[2] != expected
            msg = "[$label] ZstandardStream output mismatch (got $(length(stream_res[2])) bytes, want $(length(expected)))"
            push!(failures, msg)
            println("  FAIL (stream mismatch): $msg")
            mismatch_count[] += 1
            ok = false
        end
    else
        # CodecZstd errored — Zstandard must also error.
        if zstd_ok
            msg = "[$label] CodecZstd errored but zstd_decompress succeeded"
            push!(failures, msg)
            println("  VIOLATION: $msg")
            fail_count[] += 1
            ok = false
        end
        if stream_ok
            msg = "[$label] CodecZstd errored but ZstandardStream succeeded"
            push!(failures, msg)
            println("  VIOLATION: $msg")
            fail_count[] += 1
            ok = false
        end
    end

    if ok
        pass_count[] += 1
    end
    return ok
end

# ============================================================
# Part 1 — Round-trip over a wide variety of data types
# ============================================================

println("--- Part 1: Round-trip compression/decompression ---")

# Helper: compress then check
function roundtrip(label::String, data::Vector{UInt8}; level=3, checksum=false)
    compressed = compress_data(data; level=level, checksum=checksum)
    check_decompress("roundtrip/$label", compressed)
end

# Empty
roundtrip("empty", UInt8[])

# Short ASCII
roundtrip("ascii_short", Vector{UInt8}("Hello, world!"))
roundtrip("ascii_punct", Vector{UInt8}("!@#\$%^&*()_+-=[]{}|;':\",./<>?"))

# UTF-8 text
roundtrip("utf8_emoji",   Vector{UInt8}("🦀 🐍 🎯 👾 ∑√π∞"))
roundtrip("utf8_cjk",     Vector{UInt8}("日本語テスト — 中文测试"))
roundtrip("utf8_arabic",  Vector{UInt8}("مرحبا بالعالم"))
roundtrip("utf8_mixed",   Vector{UInt8}("café résumé naïve Ångström Σ≠Ω"))

# Newline / whitespace heavy
roundtrip("newlines",     Vector{UInt8}(join(fill("line", 1000), "\n")))
roundtrip("tabs",         Vector{UInt8}(join(fill("col", 500),  "\t")))

# Random bytes — various sizes, exercises raw/rle/compressed blocks
for (n, lbl) in [(0,"r0"),(1,"r1"),(127,"r127"),(128,"r128"),(255,"r255"),
                 (1024,"r1k"),(4096,"r4k"),(16383,"r16k-1"),(16384,"r16k"),
                 (16385,"r16k+1"),(65536,"r64k"),(200_000,"r200k")]
    Random.seed!(rng, hash(lbl))
    roundtrip("random/$lbl", rand(rng, UInt8, n))
end

# Limited alphabet — high Huffman compressibility
for (n, lbl) in [(100,"h100"),(10_000,"h10k"),(100_000,"h100k")]
    Random.seed!(rng, hash(lbl))
    roundtrip("huffman/$lbl", rand(rng, UInt8, n) .& 0x0f)
end

# Highly repetitive — exercises back-reference matches
roundtrip("zeros_small",  fill(UInt8(0x00),  1_000))
roundtrip("zeros_large",  fill(UInt8(0x00), 100_000))
roundtrip("ones",         fill(UInt8(0xFF),  50_000))
roundtrip("pattern_8",    repeat(UInt8[1,2,3,4,5,6,7,8], 10_000))
roundtrip("pattern_prime",repeat(UInt8[1,2,3,5,7,11,13,17,19,23,29,31], 8_000))

# Numeric type arrays serialised as bytes
for T in [Int8, UInt16, Int32, UInt32, Int64, Float32, Float64]
    n   = 2_000
    arr = T == Int8 ? rand(rng, Int8, n) :
          T <: AbstractFloat ? rand(rng, T, n) :
          rand(rng, T, n)
    roundtrip("typed/$T", collect(reinterpret(UInt8, arr)))
end

# Incrementing / sawtooth patterns
roundtrip("increment_u8",  UInt8.(0:255))
roundtrip("sawtooth",      repeat(UInt8.(0:127), 800))
roundtrip("reverse",       reverse(collect(UInt8, 0:255)))

# Mixed: text header + binary payload
header  = Vector{UInt8}("HEADER:version=1\n")
payload = rand(rng, UInt8, 50_000)
roundtrip("mixed_header_binary", vcat(header, payload))

# Very compressible: one unique byte repeated, then random tail
roundtrip("rle_then_random", vcat(fill(UInt8(0xAB), 50_000), rand(rng, UInt8, 1_000)))

# Compression levels
for lvl in [1, 3, 9, 19]
    data = rand(rng, UInt8, 20_000) .& 0x3f
    roundtrip("level/l$lvl", data; level=lvl)
end

# With content checksum
for n in [0, 1, 100, 10_000, 50_000]
    data = rand(rng, UInt8, n)
    roundtrip("checksum/n$n", data; checksum=true)
end

# Multi-frame concatenation (construct manually)
let
    fa = compress_data(UInt8[1, 2, 3])
    fb = compress_data(UInt8[4, 5, 6])
    fc = compress_data(rand(rng, UInt8, 1_000))
    check_decompress("multiframe/ab",    vcat(fa, fb))
    check_decompress("multiframe/abc",   vcat(fa, fb, fc))
    check_decompress("multiframe/empty", vcat(compress_data(UInt8[]), fa))
end

# Skippable frame interleaved
let
    fa   = compress_data(UInt8[42, 43, 44])
    skip = UInt8[0x50, 0x2A, 0x4D, 0x18, 0x03, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC]
    check_decompress("skippable/before", vcat(skip, fa))
    check_decompress("skippable/after",  vcat(fa, skip))
    check_decompress("skippable/both",   vcat(skip, fa, skip))
    empty_skip = UInt8[0x51, 0x2A, 0x4D, 0x18, 0x00, 0x00, 0x00, 0x00]
    check_decompress("skippable/empty",  vcat(empty_skip, fa))
end

println("  Part 1 done.  passes so far: $(pass_count[])\n")

# ============================================================
# Part 2 — Adversarial / malformed inputs
# ============================================================

println("--- Part 2: Adversarial / malformed inputs ---")

# Grab a few valid compressed frames to use as a base for mutations.
base_frames = [
    compress_data(UInt8[]),
    compress_data(UInt8[0x42]),
    compress_data(Vector{UInt8}("The quick brown fox jumps over the lazy dog.")),
    compress_data(fill(UInt8(0x00), 1_000)),
    compress_data(rand(rng, UInt8, 1_000)),
    compress_data(rand(rng, UInt8, 10_000) .& 0x0f),
    compress_data(repeat(UInt8[1,2,3,4,5,6,7,8], 200); checksum=true),
]

# --- 2a. Completely random byte sequences ---
println("  2a. Random byte sequences …")
for trial in 1:500
    n    = rand(rng, 0:4096)
    data = rand(rng, UInt8, n)
    check_decompress("random_input/t$trial", data)
end

# --- 2b. Valid magic + random garbage ---
println("  2b. Magic + random garbage …")
for trial in 1:300
    n    = rand(rng, 0:1024)
    data = vcat(ZSTD_MAGIC, rand(rng, UInt8, n))
    check_decompress("magic_garbage/t$trial", data)
end

# --- 2c. Truncated valid frames ---
println("  2c. Truncated frames …")
for frame in base_frames
    # Try every truncation length, capped at 200 to avoid extreme counts on large frames.
    maxlen = min(length(frame) - 1, 200)
    for n in 0:maxlen
        check_decompress("truncated/$(n)of$(length(frame))", frame[1:n])
    end
end

# --- 2d. Single-byte mutations of valid frames ---
println("  2d. Single-byte mutations …")
for frame in base_frames
    # Mutate every byte position; limit to frames ≤ 200 bytes to stay tractable.
    if length(frame) > 200
        positions = rand(rng, 1:length(frame), 80)
    else
        positions = 1:length(frame)
    end
    for pos in positions
        for delta in [0x01, 0x80, 0xFF]
            mutated = copy(frame)
            mutated[pos] = mutated[pos] ⊻ delta
            check_decompress("mutated/pos$(pos)_d$(delta)", mutated)
        end
    end
end

# --- 2e. Bit-level single-bit flips ---
println("  2e. Single-bit flips …")
for frame in base_frames[1:3]   # small frames only
    for pos in 1:length(frame)
        for bit in 0:7
            mutated = copy(frame)
            mutated[pos] ⊻= UInt8(1 << bit)
            check_decompress("bitflip/p$(pos)b$(bit)", mutated)
        end
    end
end

# --- 2f. Appended random garbage after a valid frame ---
println("  2f. Valid frame + trailing garbage …")
for frame in base_frames
    for trial in 1:10
        n    = rand(rng, 1:256)
        data = vcat(frame, rand(rng, UInt8, n))
        check_decompress("trailing_garbage/$(trial)", data)
    end
end

# --- 2g. Wrong magic bytes (one byte off from ZSTD magic) ---
println("  2g. Near-magic bytes …")
for i in 1:4
    for delta in [0x01, 0x02, 0x80]
        bad_magic = copy(ZSTD_MAGIC)
        bad_magic[i] ⊻= delta
        check_decompress("bad_magic/i$(i)_d$(delta)", vcat(bad_magic, rand(rng, UInt8, 32)))
    end
end

# --- 2h. Skippable frame with oversized / truncated payloads ---
println("  2h. Malformed skippable frames …")
for magic_byte in [0x50, 0x55, 0x5F]
    for payload_field in UInt32[0, 1, 3, 0x00FFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        le = reinterpret(UInt8, [htol(payload_field)])
        frame = vcat(UInt8[magic_byte, 0x2A, 0x4D, 0x18], le)
        # No actual payload bytes — forces a read-past-end for nonzero sizes.
        check_decompress("skip_malformed/m$(magic_byte)_s$(payload_field)", frame)
    end
end

# --- 2i. Oversized FCS fields ---
println("  2i. Oversized FCS …")
# FCS = 2^31 via 4-byte field (fcs_flag=2, single_segment=1) → FHD = 0xA0
fcs32_frame = vcat(ZSTD_MAGIC, UInt8[0xA0, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00])
check_decompress("fcs/2^31", fcs32_frame)

# FCS = 2^33 via 8-byte field (fcs_flag=3, single_segment=1) → FHD = 0xE0
fcs64_frame = vcat(ZSTD_MAGIC, UInt8[0xE0, 0x00,0x00,0x00,0x00, 0x02,0x00,0x00,0x00, 0x01,0x00,0x00])
check_decompress("fcs/2^33", fcs64_frame)

# --- 2j. Reserved bits set in Frame Header Descriptor ---
println("  2j. Reserved FHD bits …")
base = base_frames[3]  # a real frame
for bit in [0x08]      # bit 3 of FHD is reserved and must be 0
    bad = copy(base)
    bad[5] |= bit
    check_decompress("fhd_reserved/bit$(bit)", bad)
end

# --- 2k. Corrupted checksums ---
println("  2k. Corrupted checksums …")
for data in [UInt8[0x42], rand(rng, UInt8, 100), rand(rng, UInt8, 10_000)]
    frame = compress_data(data; checksum=true)
    for delta in [0x01, 0x80, 0xFF]
        bad = copy(frame)
        bad[end] ⊻= delta
        check_decompress("bad_checksum/delta$(delta)", bad)
    end
end

# --- 2l. Two valid frames where the second is corrupted ---
println("  2l. Concatenated frames with corrupted second …")
fa = compress_data(UInt8[1, 2, 3])
fb = compress_data(UInt8[4, 5, 6])
for pos in 1:length(fb)
    bad_fb = copy(fb)
    bad_fb[pos] ⊻= 0xFF
    check_decompress("concat_corrupt2/p$(pos)", vcat(fa, bad_fb))
end

# --- 2m. Pseudo-structured: valid FHD with garbage block data ---
println("  2m. Valid header, garbage block …")
for trial in 1:200
    n     = rand(rng, 3:512)
    block = rand(rng, UInt8, n)
    data  = vcat(ZSTD_MAGIC, UInt8[0x00], block)   # FHD=0x00 (no FCS, no dict, no checksum)
    check_decompress("hdr_garbage_block/t$trial", data)
end

# --- 2n. Stress: large volume of random inputs at scale ---
println("  2n. Bulk random inputs (1000 trials) …")
for trial in 1:1000
    n    = rand(rng, 0:16384)
    data = rand(rng, UInt8, n)
    check_decompress("bulk/t$trial", data)
end

# --- 2o. Stress: magic + random at scale ---
println("  2o. Bulk magic+random (500 trials) …")
for trial in 1:500
    n    = rand(rng, 1:8192)
    data = vcat(ZSTD_MAGIC, rand(rng, UInt8, n))
    check_decompress("bulk_magic/t$trial", data)
end

# ============================================================
# Summary
# ============================================================

total = pass_count[] + fail_count[] + mismatch_count[] + zstd_only_error[]
println()
println("=" ^ 60)
println("SUMMARY")
println("  Total checks:             $total")
println("  Passed:                   $(pass_count[])")
println("  VIOLATIONS (Zstd missed error):  $(fail_count[])")
println("  VIOLATIONS (output mismatch):    $(mismatch_count[])")
println("  Zstandard-only errors (info):    $(zstd_only_error[])")
println("=" ^ 60)

if !isempty(failures)
    println("\nFAILED CASES:")
    for f in failures
        println("  • $f")
    end
end

n_violations = fail_count[] + mismatch_count[]
if n_violations == 0
    println("\nAll invariants satisfied.")
    exit(0)
else
    println("\n$n_violations invariant violation(s) found.")
    exit(1)
end
