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

include("util.jl")
include("xxhash.jl")
include("forwardbitreader.jl")
include("reversebitreader.jl")
include("fse.jl")
include("huffman.jl")
include("dictionary.jl")
include("constants.jl")

export inflate_zstd, InflateZstdStream, ZstdDict


# ============================================================
# Section 8: Decompression state + literals section
#   Reference: RFC 8878 §3.1.1.3
# ============================================================

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

function read_distribution_table!(br::ForwardBitReader, default::FSEDistTable,
                         prev::Union{FSEDistTable, RLEDistTable, Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         syms::Vector{UInt8}, nb::Vector{UInt8},
                         base::Vector{UInt32}, occ::Vector{Int},
                         norm::Vector{Int16})
    if mode == 0 # Predefined_Mode
        return default
    elseif mode == 1 # RLE_Mode
        sym = read(br, 8)
        return RLEDistTable(UInt8(sym))
    elseif mode == 2 # FSE_Compressed_Mode
        al, dist = read_fse_dist!(br, norm)
        al ≤ max_al || throw(ArgumentError("zstd: accuracy log $al exceeds maximum $max_al"))
        length(dist) ≤ max_sym + 1 || throw(ArgumentError("zstd: FSE distribution has $(length(dist)) symbols, maximum is $(max_sym + 1)"))
        return build_fse_table(dist, al, syms, nb, base, occ)
    else # mode == 3; Repeat_Mode
        prev !== nothing || throw(ArgumentError("zstd: repeat mode but no previous distribution table"))
        return prev
    end
end

function read_distribution_table!(br::ForwardBitReader, default::FSEDistTable,
                         prev::Union{FSEDistTable,RLEDistTable,Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         slot::FSEDistTableSlot, state::DecompressState)
    return read_distribution_table!(br, default, prev, mode, max_sym, max_al,
                           slot.syms, slot.nb, slot.base,
                           state.fse_occ, state.fse_norm)
end


# ============================================================
# Section 8: Sequences section
#   Reference: RFC 8878 §3.1.1.3.3
# ============================================================

# Copy n bytes from src to dst using 16-byte SIMD chunks.
# Modelled on ZSTD_wildcopy in zstd/lib/common/zstd_internal.h: copies always
# proceed in full 16-byte chunks, deliberately over-reading/over-writing by up
# to 15 bytes into pre-allocated slack to avoid a branch on the tail.
# Requires src to have ≥15 bytes of allocated slack past its valid content,
# and dst to have ≥15 bytes of allocated slack past the write end.
# Both src and dst must not overlap.
@inline function _wildcopy16!(dst::Ptr{UInt8}, src::Ptr{UInt8}, n::Int)
    n == 0 && return
    if n < 16
        @inbounds vstore(vload(Vec{16, UInt8}, src), dst)
        return
    end
    k = 0
    while k + 16 ≤ n
        @inbounds vstore(vload(Vec{16, UInt8}, src + k), dst + k)
        k += 16
    end
    k < n && @inbounds vstore(vload(Vec{16, UInt8}, src + n - 16), dst + n - 16)
end

# ============================================================
# Section 10: Frame decompression
#   Reference: RFC 8878 §3.1
# ============================================================

@inline function _read_magic(data::Vector{UInt8}, pos::Int)
    length(data) ≥ pos + 3 ||
        throw(ArgumentError("zstd: truncated frame (magic)"))
    _le32(data, pos)
end

# Skip a skippable frame (RFC 8878 §3.1.2).  Returns the position after the frame.
function _skip_frame(data::Vector{UInt8}, pos::Int)
    pos += 4 # past magic

    length(data) ≥ pos + 3 ||
        throw(ArgumentError("zstd: truncated skippable frame (size)"))

    frame_size = Int64(_le32(data, pos)) # 4-byte LE size field (may exceed Int32)
    pos += 4

    pos + frame_size - 1 ≤ length(data) ||
        throw(ArgumentError("zstd: truncated skippable frame (data)"))

    pos += Int(frame_size)
    return pos
end

function _read_frame_header_descriptor(data::Vector{UInt8}, pos::Int)
    length(data) ≥ pos ||
        throw(ArgumentError("zstd: truncated frame (FHD)"))

    fhd = Int(data[pos])
    fcs_flag = (fhd >> 6) & 0x03
    single_segment_flag = Bool((fhd >> 5) & 0x01)
    content_checksum_flag = Bool((fhd >> 2) & 0x01)
    dict_id_flag = fhd & 0x03
    (fhd >> 3) & 0x01 == 0 ||
        throw(ArgumentError("zstd: reserved bit set in frame header descriptor"))
    fcs_size = (fcs_flag == 0 && !single_segment_flag) ?
        0 :
        1 << fcs_flag
    dict_id_size = (dict_id_flag == 0) ?
        0 :
        1 << (dict_id_flag - 1)
    
    pos += 1

    return fcs_size, single_segment_flag, content_checksum_flag, dict_id_size, pos
end

function _read_and_validate_dict_id(data::Vector{UInt8}, pos::Int, dict_id_size::Int, dict::Union{ZstdDict, Nothing})
    dict_id_size > 0 ||
        return pos # no dictionary ID field
    length(data) ≥ pos + dict_id_size - 1 ||
        throw(ArgumentError("zstd: truncated frame (FHD)"))

    dict_id_size != 0 && dict === nothing &&
        throw(ArgumentError("zstd: frame requires a dictionary but none was provided"))
    if dict_id_size > 0 && dict !== nothing && dict.id != 0
        frame_dict_id = (dict_id_size == 1) ? UInt32(data[pos]) :
                        (dict_id_size == 2) ? UInt32(_le16(data, pos)) :
                        _le32(data, pos)
        frame_dict_id == dict.id ||
            throw(ArgumentError("zstd: dictionary ID mismatch (frame=0x$(string(frame_dict_id, base = 16)), dict=0x$(string(dict.id, base = 16)))"))
    end

    pos += dict_id_size

    return pos
end

function _read_frame_content_size(data::Vector{UInt8}, pos::Int, fcs_size::Int)
    fcs_size > 0 ||
        return -1, pos # unknown content size
    length(data) ≥ pos + fcs_size - 1 ||
        throw(ArgumentError("zstd: truncated frame (FHD)"))
    
    fcs_u64 =
        fcs_size == 0 ? UInt64(0) : # unknown
        fcs_size == 1 ? UInt64(data[pos]) :
        fcs_size == 2 ? UInt64(_le16(data, pos)) + 256 :
        fcs_size == 4 ? UInt64(_le32(data, pos)) :
                        _le64(data, pos)
    fcs_u64 ≤ typemax(Int) ||
        throw(ArgumentError("zstd: frame content size $fcs_u64 exceeds addressable range"))

    pos += fcs_size

    return Int(fcs_u64), pos
end

function _read_window_descriptor(data::Vector{UInt8}, pos::Int, single_segment_flag::Bool)
    single_segment_flag && return 0, pos
    length(data) ≥ pos ||
        throw(ArgumentError("zstd: truncated frame (WD)"))
    return Int(data[pos]), pos + 1
end

function _read_frame_header(data::Vector{UInt8}, pos::Int, dict::Union{ZstdDict, Nothing})
    # Frame Header Descriptor (RFC 8878 §3.1.1.1.1)
    fcs_size, single_segment_flag, content_checksum_flag, dict_id_size, pos = _read_frame_header_descriptor(data, pos)

    # Dictionary ID (RFC 8878 §3.1.1.1.3)
    pos = _read_and_validate_dict_id(data, pos, dict_id_size, dict)

    # Window Descriptor (RFC 8878 §3.1.1.1.2, omitted when Single_Segment_Flag is set)
    window_descriptor, pos = _read_window_descriptor(data, pos, single_segment_flag)

    # Frame Content Size (RFC 8878 §3.1.1.1.4)
    frame_content_size, pos = _read_frame_content_size(data, pos, fcs_size)

    # Set Window Size
    if single_segment_flag
        frame_content_size ≥ 0 ||
            throw(ArgumentError("zstd: single-segment frame with unknown content size"))
        window_size = frame_content_size
    else
        exponent = window_descriptor >> 4
        mantissa = window_descriptor & 0x0f
        window_base = 1 << (10 + exponent)
        window_size = window_base + (window_base >> 3) * mantissa
    end

    return window_size, frame_content_size, content_checksum_flag, pos
end

"""
    FrameInfo

Lightweight descriptor for a single non-skippable zstd frame found during
pre-scan.  `data_start` is the 1-based byte offset of the frame's 4-byte
magic number in the input vector.  `fcs` is the declared decompressed size
in bytes, or -1 when the Frame Content Size field is absent from the header.
"""
struct FrameInfo
    data_start ::Int   # 1-based byte offset of the frame's magic number
    fcs        ::Int   # frame content size in bytes; -1 if absent
end

"""
    _scan_frames(data, pos, dict) -> (Vector{FrameInfo}, Int)

Walk `data` starting at byte offset `pos`, reading frame and block headers
without decompressing, and return a `Vector{FrameInfo}` (one entry per
non-skippable zstd frame found) together with the position after the last
consumed byte.

Skippable frames are silently skipped and excluded from the result.
Throws `ArgumentError` on any structural violation (truncated headers,
reserved bits, oversized blocks, bad magic numbers) using the same error
messages as `_decompress_frame!`.
"""
function _scan_frames(data::Vector{UInt8}, pos::Int, dict::Union{ZstdDict, Nothing})
    frames = FrameInfo[]
    while pos ≤ length(data)
        frame_start = pos
        magic = _read_magic(data, pos)
        if _is_skippable(magic)
            pos = _skip_frame(data, pos)
        elseif magic == ZSTD_MAGIC
            pos += 4  # advance past magic number

            # Read frame header fields — validates reserved bits and dict ID
            fcs_size, single_segment_flag, content_checksum_flag, dict_id_size, pos =
                _read_frame_header_descriptor(data, pos)
            pos = _read_and_validate_dict_id(data, pos, dict_id_size, dict)
            # Call _read_window_descriptor solely to advance pos past the
            # window descriptor byte.  The parsed window size is not needed
            # for scanning; discard the first return value.
            _, pos = _read_window_descriptor(data, pos, single_segment_flag)
            fcs, pos = _read_frame_content_size(data, pos, fcs_size)

            # Scan block headers to advance past the frame without decompressing.
            # Block header is 3 bytes; block_type distinguishes advance amount:
            #   0 (raw)        → advance by block_size bytes
            #   1 (RLE)        → advance by 1 byte (single repeated byte)
            #   2 (compressed) → advance by block_size bytes
            #   3 (reserved)   → error; guard here so _scan_frames produces a
            #                    clean error rather than silently miscomputing
            #                    frame boundaries for subsequent frames
            while true
                length(data) ≥ pos + 2 ||
                    throw(ArgumentError("zstd: truncated block header"))
                bh1 = Int(data[pos])
                bh2 = Int(data[pos + 1])
                bh3 = Int(data[pos + 2])
                pos += 3
                last_block = bh1 & 0x01
                block_type = (bh1 >> 1) & 0x03
                block_size = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
                block_type != 3 ||
                    throw(ArgumentError("zstd: reserved block type"))
                block_size ≤ 131072 ||
                    throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))
                if block_type == 1  # RLE: 1-byte payload regardless of block_size
                    pos += 1
                else
                    pos += block_size
                end
                last_block != 0 && break
            end

            # Skip optional 4-byte content checksum
            if content_checksum_flag
                length(data) ≥ pos + 3 ||
                    throw(ArgumentError("zstd: truncated content checksum"))
                pos += 4
            end

            push!(frames, FrameInfo(frame_start, fcs))
        else
            throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base=16))"))
        end
    end
    return frames, pos
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
