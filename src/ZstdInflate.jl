# Pure Julia implementation of Zstandard (Zstd) decompression,
# as specified by RFC 8878.
#
# Reference: https://www.rfc-editor.org/rfc/rfc8878

"""
    ZstdInflate

Pure Julia implementation of decompression of the Zstandard format.

In-memory decompression:

| function | decompresses |
| -------- | ------------ |
| `inflate_zstd(data::Vector{UInt8})` | Zstandard frame |
| `inflate_zstd(filename::AbstractString)` | Zstandard file |

Streaming decompression:

| stream | decompresses |
| ------ | ------------ |
| `InflateZstdStream(stream::IO)` | Zstandard stream |

Reference: [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878)
"""
module ZstdInflate

using SIMD

# Groups the three backing arrays for one FSE decode table (LL, OF, or ML).
# Having all three in one object lets callers pass a single slot instead of
# three separate vectors, and keeps DecompressState compact.
mutable struct FSEDistTableSlot
    syms ::Vector{UInt8}
    nb   ::Vector{UInt8}
    base ::Vector{UInt32}
end

FSEDistTableSlot(n::Int) = FSEDistTableSlot(Vector{UInt8}(undef, n),
                                     Vector{UInt8}(undef, n),
                                     Vector{UInt32}(undef, n))

include("util.jl")
include("xxhash.jl")
include("forwardbitreader.jl")
include("reversebitreader.jl")
include("disttables.jl")
include("huffman.jl")
include("dictionary.jl")
include("constants.jl")

export inflate_zstd, InflateZstdStream, ZstdDict


# ============================================================
# Section 8: Decompression state + literals section
#   Reference: RFC 8878 §3.1.1.3
# ============================================================

mutable struct DecompressState
    rep         ::NTuple{3,Int}
    huffman     ::Union{HuffmanTable,Nothing}
    ll_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    ml_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    of_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    dict_content::Vector{UInt8}   # dictionary content prefix for match references
    # Reusable sequence buffers — grown on demand, never shrunk
    ll_vals        ::Vector{Int}
    ml_vals        ::Vector{Int}
    of_vals        ::Vector{Int}
    # Reusable literals buffer — holds decoded literals for the current block
    literals_buf   ::Vector{UInt8}
    # Reusable Huffman build scratch — pre-sized to maximum, never shrunk
    huf_dtable     ::Vector{UInt32}  # full 2^max_bits decode table
    huf_rank_count ::Vector{Int}
    huf_rank_start ::Vector{Int}
    huf_weights    ::Vector{UInt8}
    # Reusable FSE table backing arrays — one slot per table (LL, ML, OF).
    # Slots cannot be shared because all three tables are live simultaneously
    # during sequence decoding.
    ll_slot ::FSEDistTableSlot
    ml_slot ::FSEDistTableSlot
    of_slot ::FSEDistTableSlot
    # Shared FSE build scratch — safe to share because LL/ML/OF are built sequentially
    fse_occ  ::Vector{Int}
    fse_norm ::Vector{Int16}
end

include("decompress.jl")

const _FSE_MAX_TABLE = 512   # 1 << max accuracy_log (9 for LL/ML, 8 for OF)
const ZSTD_BLOCKSIZE_MAX = 131072  # maximum decompressed size of any single block (RFC 8878)

DecompressState() = DecompressState(
    INIT_REPEAT_OFFSETS, nothing, nothing, nothing, nothing, UInt8[],
    Int[], Int[], Int[],
    UInt8[],
    Vector{UInt32}(undef, 1 << HUFTABLE_LOG_MAX),
    zeros(Int, HUFTABLE_LOG_MAX + 1),
    zeros(Int, HUFTABLE_LOG_MAX + 1),
    UInt8[],
    FSEDistTableSlot(_FSE_MAX_TABLE),
    FSEDistTableSlot(_FSE_MAX_TABLE),
    FSEDistTableSlot(_FSE_MAX_TABLE),
    Int[], Int16[])

function DecompressState(dict::ZstdDict)
    DecompressState(
        dict.rep, dict.huffman, dict.ll_tab, dict.ml_tab, dict.of_tab, dict.content,
        Int[], Int[], Int[],
        UInt8[],
        Vector{UInt32}(undef, 1 << HUFTABLE_LOG_MAX),
        zeros(Int, HUFTABLE_LOG_MAX),
        zeros(Int, HUFTABLE_LOG_MAX),
        UInt8[],
        FSEDistTableSlot(_FSE_MAX_TABLE),
        FSEDistTableSlot(_FSE_MAX_TABLE),
        FSEDistTableSlot(_FSE_MAX_TABLE),
        Int[], Int16[])
end

# ============================================================
# Section 11: Public API
#   Mirrors Inflate.jl interface.
# ============================================================

@inline _is_skippable(magic::UInt32) = (magic & 0xFFFFFFF0) == 0x184D2A50

"""
    inflate_zstd(data::Vector{UInt8}; dict=nothing, nthreads=Threads.nthreads()) -> Vector{UInt8}

Decompress one or more concatenated Zstandard frames from `data` and return
the raw bytes.  Skippable frames (RFC 8878 §3.1.2) are silently ignored.

When `nthreads ≥ 2` and `data` contains two or more independent zstd frames,
each frame is decompressed in a separate Julia task (capped at `nthreads`
concurrent tasks).  Results are concatenated in frame order.  With `nthreads=1`
or a single-frame input the existing serial path is taken unchanged.

`nthreads` must be ≥ 1; passing 0 or negative throws `ArgumentError`.

If the frame was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
function inflate_zstd(data::Vector{UInt8}; dict::Union{ZstdDict,Nothing}=nothing, nthreads::Int=Threads.nthreads())
    nthreads ≥ 1 || throw(ArgumentError("zstd: nthreads must be ≥ 1, got $nthreads"))
    isempty(data) && throw(ArgumentError("zstd: empty input"))
    d = dict

    # Parallel path: only when nthreads ≥ 2 and there are ≥ 2 independent frames.
    # _scan_frames is O(compressed size) but reads only header/block-header bytes.
    if nthreads ≥ 2
        frames, _ = _scan_frames(data, 1, d)
        if length(frames) ≥ 2
            sem       = Base.Semaphore(min(nthreads, length(frames)))
            frame_bufs = Vector{Vector{UInt8}}(undef, length(frames))
            @sync for (i, frame) in enumerate(frames)
                Threads.@spawn Base.acquire(sem) do
                    buf = UInt8[]
                    frame.fcs > 0 && sizehint!(buf, frame.fcs)
                    _decompress_frame!(data, frame.data_start, buf, d)
                    frame_bufs[i] = buf
                end
            end
            total = sum(length, frame_bufs)
            out   = Vector{UInt8}(undef, total)
            wp    = 1
            for buf in frame_bufs
                n = length(buf)
                copyto!(out, wp, buf, 1, n)
                wp += n
            end
            return out
        end
    end

    # Serial fast-path: nthreads=1, or input has ≤ 1 zstd frame.
    # One-time hint: for incompressible data compressed ≈ raw size; for
    # compressible data this underestimates but limits reallocation to O(1)
    # doublings regardless of frame count.
    pos = 1
    out = UInt8[]
    sizehint!(out, length(data))
    while pos ≤ length(data)
        magic = _read_magic(data, pos)
        if _is_skippable(magic)
            pos = _skip_frame(data, pos)
        elseif magic == ZSTD_MAGIC
            pos = _decompress_frame!(data, pos, out, d)
        else
            throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base=16))"))
        end
    end
    return out
end

"""
    inflate_zstd(filename::AbstractString; dict=nothing, nthreads=Threads.nthreads()) -> String

Read a `.zst` file and return the decompressed content as a `String`.
"""
function inflate_zstd(filename::AbstractString; dict::Union{ZstdDict,Nothing}=nothing, nthreads::Int=Threads.nthreads())
    data = read(filename)
    String(inflate_zstd(data; dict=dict, nthreads=nthreads))
end

# ============================================================
# Section 12: Streaming interface
#   InflateZstdStream wraps any IO and decompresses eagerly.
# ============================================================

"""
    InflateZstdStream(io::IO; dict=nothing, nthreads=Threads.nthreads())

Create a readable stream that decompresses Zstandard data from `io`.
The stream reads all compressed data at construction time; subsequent
reads deliver decompressed bytes.

If the data was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
mutable struct InflateZstdStream <: IO
    buf::Vector{UInt8}
    pos::Int
end

function InflateZstdStream(io::IO; dict::Union{ZstdDict,Nothing}=nothing, nthreads::Int=Threads.nthreads())
    compressed = read(io)
    decompressed = inflate_zstd(compressed; dict=dict, nthreads=nthreads)
    InflateZstdStream(decompressed, 1)
end

Base.eof(s::InflateZstdStream) = s.pos > length(s.buf)

function Base.read(s::InflateZstdStream, ::Type{UInt8})
    s.pos ≤ length(s.buf) || throw(EOFError())
    b = s.buf[s.pos]
    s.pos += 1
    return b
end

function Base.readbytes!(s::InflateZstdStream, b::AbstractVector{UInt8}, nb=length(b))
    available = length(s.buf) - s.pos + 1
    n = min(nb, available)
    n ≤ 0 && return 0
    length(b) < n && resize!(b, n)
    copyto!(b, 1, s.buf, s.pos, n)
    s.pos += n
    return n
end

Base.bytesavailable(s::InflateZstdStream) = max(0, length(s.buf) - s.pos + 1)

end  # module ZstdInflate
