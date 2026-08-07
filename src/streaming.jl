# ============================================================
# Streaming interface
#   InflateZstdStream decompresses incrementally: compressed
#   bytes are read from the source IO one block at a time as
#   output is consumed, and decompressed output is discarded
#   once it has been read.
# ============================================================

"""
    InflateZstdStream(io::IO; dict = nothing)

Create a readable stream that incrementally decompresses Zstandard data
from `io`. Compressed bytes are read from `io` one block at a time as
output is consumed, and decompressed output is discarded once it has been
read and has aged past the frame's window, so memory use is bounded by the
frame's declared window size plus one block (128 KiB).

If the data was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
mutable struct InflateZstdStream{T <: IO} <: IO
    io          ::T
    dict        ::Union{ZstdDict, Nothing}
    out         ::Vector{UInt8}      # retained output: window history + unconsumed bytes
    read_pos    ::Int                # next unconsumed byte in out
    wpos        ::Int                # next write position in out
    inbuf       ::Vector{UInt8}      # reusable compressed-payload buffer
    hdrbuf      ::Vector{UInt8}      # reusable small buffer for headers/checksums
    # Per-frame decode state (valid while in_frame)
    state       ::DecompressState
    in_frame    ::Bool
    frame_start ::Int                # out-coordinate where the current frame began (≥ 0)
    window_size ::Int
    fcs         ::Int                # declared frame content size; -1 if absent
    frame_len   ::Int                # bytes produced by the current frame so far
    check_flag  ::Bool
    hasher      ::XXH64Stream
    source_done ::Bool               # source IO exhausted at a frame boundary
    mark_pos    ::Int                # out-coordinate of the active mark; -1 if unmarked
    dropped     ::Int                # total bytes ever discarded from out by compaction
end

function InflateZstdStream(io::IO; dict::Union{ZstdDict, Nothing} = nothing)
    s = InflateZstdStream{typeof(io)}(io, dict, UInt8[], 1, 1, UInt8[], UInt8[],
                                      DecompressState(), false, 0, 0, -1, 0, false,
                                      XXH64Stream(), false, -1, 0)
    # Parse the first frame header eagerly so structural errors (empty input,
    # bad magic, missing dictionary) surface at construction time.
    eof(io) &&
        throw(ArgumentError("zstd: empty input"))
    _start_frame!(s)
    return s
end

# Read exactly n bytes from io into buf (resized to n).
function _read_exact!(io::IO, buf::Vector{UInt8}, n::Int, what::String)
    resize!(buf, n)
    readbytes!(io, buf, n) == n ||
        throw(ArgumentError("zstd: truncated $what"))
    return buf
end

# Read and discard n bytes (skippable frame payload).
function _discard!(io::IO, scratch::Vector{UInt8}, n::Int)
    while n > 0
        k = min(n, 1 << 16)
        resize!(scratch, k)
        nr = readbytes!(io, scratch, k)
        nr > 0 ||
            throw(ArgumentError("zstd: truncated skippable frame (data)"))
        n -= nr
    end
    return
end

# Advance past skippable frames and begin the next zstd frame, parsing its
# header and initialising per-frame state. Returns false if the source ended
# cleanly at a frame boundary.
function _start_frame!(s::InflateZstdStream)
    while true
        eof(s.io) && (s.source_done = true; return false)
        magic = _le32(_read_exact!(s.io, s.hdrbuf, 4, "frame (magic)"), 1)
        if _is_skippable(magic)
            n = Int(Int64(_le32(_read_exact!(s.io, s.hdrbuf, 4, "skippable frame (size)"), 1)))
            _discard!(s.io, s.inbuf, n)
        elseif magic == ZSTD_MAGIC
            break
        else
            throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base = 16))"))
        end
    end

    # Frame Header Descriptor first: it determines how many more header bytes
    # follow (dictionary ID + window descriptor + frame content size).
    fhd = _read_exact!(s.io, s.hdrbuf, 1, "frame (FHD)")[1]
    fcs_size, single_segment_flag, _, dict_id_size, _ =
        _read_frame_header_descriptor(UInt8[fhd], 1)
    rest = dict_id_size + (single_segment_flag ? 0 : 1) + fcs_size

    hdr = Vector{UInt8}(undef, 1 + rest)
    hdr[1] = fhd
    rest > 0 && copyto!(hdr, 2, _read_exact!(s.io, s.hdrbuf, rest, "frame (header)"), 1, rest)

    window_size, frame_content_size, content_checksum_flag, _ = _read_frame_header(hdr, 1, s.dict)
    window_size ≤ STREAM_WINDOW_SIZE_MAX ||
        throw(ArgumentError("zstd: window size $window_size exceeds maximum supported for " *
                            "streaming ($STREAM_WINDOW_SIZE_MAX bytes)"))

    s.state       = s.dict !== nothing ? DecompressState(s.dict) : DecompressState()
    s.in_frame    = true
    s.frame_start = s.wpos - 1
    s.window_size = window_size
    s.fcs         = frame_content_size
    s.frame_len   = 0
    s.check_flag  = content_checksum_flag
    s.check_flag && xxh_reset!(s.hasher)
    return true
end

# Read and decode the next block of the current frame; on the last block,
# validate the frame content size and content checksum.
function _decode_next_block!(s::InflateZstdStream)
    bh = _read_exact!(s.io, s.hdrbuf, 3, "block header")
    bh1, bh2, bh3 = Int(bh[1]), Int(bh[2]), Int(bh[3])

    last_block = bh1 & 0x01
    block_type = (bh1 >> 1) & 0x03
    block_size = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
    block_size ≤ ZSTD_BLOCKSIZE_MAX ||
        throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))

    payload_len = block_type == 1 ? 1 : block_size  # RLE payload is a single byte
    payload_len > 0 && _read_exact!(s.io, s.inbuf, payload_len, "block payload")

    wpos0 = s.wpos
    s.wpos = _apply_block!(block_type, s.inbuf, 1, block_size, s.state, s.out, s.wpos,
                           false, s.frame_start, typemax(Int) - WILDCOPY_SLACK)
    s.frame_len += s.wpos - wpos0
    s.check_flag && xxh_update!(s.hasher, @view s.out[wpos0:s.wpos - 1])

    if last_block != 0
        s.fcs < 0 || s.fcs == s.frame_len ||
            throw(ArgumentError("zstd: decompressed size $(s.frame_len) does not match frame content size $(s.fcs)"))
        if s.check_flag
            stored = _le32(_read_exact!(s.io, s.hdrbuf, 4, "content checksum"), 1)
            computed = UInt32(xxh_finalize(s.hasher) & 0xFFFFFFFF)
            stored == computed ||
                throw(ArgumentError("zstd: content checksum mismatch (stored=0x$(string(stored, base = 16)), computed=0x$(string(computed, base = 16)))"))
        end
        s.in_frame = false
    end
    return
end

# Drop retained output that the consumer has read (or that is pinned by an
# active mark) and that has aged past the window (no future match can
# reference it). Compaction runs only when the droppable prefix is at least
# as large as the retained tail, so the total cost of all copies is O(total
# output).
function _compact!(s::InflateZstdStream)
    consumer_floor = ismarked(s) ? s.mark_pos : s.read_pos
    keep_from = min(consumer_floor, s.wpos - s.window_size)
    drop = keep_from - 1
    (drop ≥ 1 << 16 && drop ≥ s.wpos - keep_from) ||
        return
    copyto!(s.out, 1, s.out, keep_from, s.wpos - keep_from)
    s.wpos        -= drop
    s.read_pos    -= drop
    s.frame_start -= drop
    s.dropped     += drop
    ismarked(s) && (s.mark_pos -= drop)
    if s.frame_start < 0
        # In-frame history older than the window has been dropped; conformant
        # matches can no longer reach the frame start, so the dictionary is
        # unreachable too. Clearing it makes the dict_pos ≥ 1 guard in
        # execute_sequences! reject any (malformed) offset that tries.
        s.frame_start = 0
        empty!(s.state.dict_content)
    end
    return
end

# Decode until at least one unconsumed byte is available or the stream ends.
# Returns true if bytes are available.
function _fill!(s::InflateZstdStream)
    while s.read_pos ≥ s.wpos
        if s.in_frame
            _decode_next_block!(s)
            _compact!(s)
        else
            s.source_done && return false
            _start_frame!(s) || return false
        end
    end
    return true
end

Base.eof(s::InflateZstdStream) = !_fill!(s)

function Base.read(s::InflateZstdStream, ::Type{UInt8})
    _fill!(s) || throw(EOFError())
    b = s.out[s.read_pos]
    s.read_pos += 1
    return b
end

function Base.readbytes!(s::InflateZstdStream, b::AbstractVector{UInt8}, nb = length(b))
    n = 0
    while n < nb && _fill!(s)
        k = min(nb - n, s.wpos - s.read_pos)
        length(b) < n + k && resize!(b, n + k)
        copyto!(b, n + 1, s.out, s.read_pos, k)
        s.read_pos += k
        n += k
    end
    return n
end

# Decoded-but-unconsumed bytes only; does not trigger further decoding.
Base.bytesavailable(s::InflateZstdStream) = s.wpos - s.read_pos

Base.close(s::InflateZstdStream) = close(s.io)
Base.isopen(s::InflateZstdStream) = isopen(s.io)

# Absolute count of decompressed bytes consumed so far, stable across
# internal buffer compaction.
Base.position(s::InflateZstdStream) = s.dropped + s.read_pos - 1

Base.ismarked(s::InflateZstdStream) = s.mark_pos != -1

"""
    mark(s::InflateZstdStream) -> Int64

Mark the current position in the decompressed output. A later `reset(s)`
rewinds to this position. While a mark is held, decompressed output back to
the mark is retained even past the frame's window, so an outstanding mark
disables the usual bounded-memory compaction until it is cleared by `reset`
or `unmark`.
"""
function Base.mark(s::InflateZstdStream)
    s.mark_pos = s.read_pos
    return position(s)
end

"""
    unmark(s::InflateZstdStream) -> Bool

Remove any mark on `s` without resetting to it. Returns whether a mark was
present.
"""
function Base.unmark(s::InflateZstdStream)
    had = ismarked(s)
    s.mark_pos = -1
    return had
end

"""
    reset(s::InflateZstdStream) -> Int64

Rewind `s` to its last `mark`ed position and remove the mark. Throws
`ArgumentError` if `s` is not marked.
"""
function Base.reset(s::InflateZstdStream)
    ismarked(s) || throw(ArgumentError("InflateZstdStream not marked"))
    s.read_pos = s.mark_pos
    s.mark_pos = -1
    return position(s)
end

"""
    seekstart(s::InflateZstdStream) -> s

Rewind `s` to the beginning of the decompressed output by seeking the
underlying `io` back to its start and restarting decoding from there.
Requires that `io` itself supports `seekstart`.
"""
function Base.seekstart(s::InflateZstdStream)
    seekstart(s.io)
    eof(s.io) &&
        throw(ArgumentError("zstd: empty input"))
    resize!(s.out, 0)
    s.read_pos    = 1
    s.wpos        = 1
    s.dropped     = 0
    s.mark_pos    = -1
    s.in_frame    = false
    s.source_done = false
    _start_frame!(s)
    return s
end
