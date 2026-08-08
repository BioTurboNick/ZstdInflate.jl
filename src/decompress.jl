# Owns the backing entry array for one FSE decode table (LL, OF, or ML), plus a
# persistent `FSEDistTable` wrapper aliasing that same array. `table`'s array
# field never changes identity after construction (resize! in
# `_fill_fse_table!` mutates the vector in place); only its `accuracy_log`
# needs updating per call, in `build_fse_table!` below — so reusing `table`
# across blocks needs no resynchronization beyond that.
struct FSEDistTableSlot
    entries::Vector{UInt64}
    table  ::FSEDistTable
end

function FSEDistTableSlot(n::Int)
    entries = Vector{UInt64}(undef, n)
    FSEDistTableSlot(entries, FSEDistTable(0, entries))
end

# Hot path: mutate the slot's persistent FSEDistTable in place instead of
# allocating a fresh wrapper every FSE_Compressed-mode block. The backing
# array is already reused (resize! in `_fill_fse_table!`); this reuses the
# small wrapper object too.
function build_fse_table!(norm::AbstractVector{<:Integer}, accuracy_log::Int,
                           slot::FSEDistTableSlot, occ::Vector{Int}, kind)
    _fill_fse_table!(norm, accuracy_log, slot.entries, occ, kind)
    slot.table.accuracy_log = accuracy_log
    return slot.table
end

mutable struct DecompressState
    rep         ::NTuple{3,Int}
    huffman     ::Union{HuffmanTable,Nothing}
    ll_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    ml_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    of_tab      ::Union{FSEDistTable,RLEDistTable,Nothing}
    dict_content::Vector{UInt8}   # dictionary content prefix for match references
    # Reusable literals buffer — holds decoded literals for the current block
    literals_buf   ::Vector{UInt8}
    # Reusable Huffman build scratch — pre-sized to maximum, never shrunk
    huf_rank_count ::Vector{Int}
    huf_rank_start ::Vector{Int}
    huf_weights    ::Vector{UInt8}
    huf_nbits_sym1 ::Vector{UInt8}   # build_huffman_table!'s pass-1 stream_nbits snapshot
    # Reusable FSE table backing arrays — one slot per table (LL, ML, OF).
    # Slots cannot be shared because all three tables are live simultaneously
    # during sequence decoding.
    ll_slot ::FSEDistTableSlot
    ml_slot ::FSEDistTableSlot
    of_slot ::FSEDistTableSlot
    # Shared FSE build scratch — safe to share because LL/ML/OF are built sequentially
    fse_occ  ::Vector{Int}
    fse_norm ::Vector{Int16}
    # Reusable reverse-bitstream readers. Each frame decode gets its own
    # DecompressState (frames run on separate tasks under nthreads > 1), so
    # reusing these across blocks within one frame's decode is safe: the
    # sequences reader and the single-stream literals reader are never live
    # at the same time as each other, and `huf4` bundles the 4-stream
    # Huffman decoder's own internal pool (see reversebitreader.jl).
    rb_seq  ::ReverseBitReader{RBRView}
    rb_lit1 ::ReverseBitReader{RBRView}
    huf4    ::Huffman4StreamScratch{RBRView}
    # Shared forward-bitstream reader for the three header-parsing readers
    # that used to be constructed fresh each time: read_literals's own
    # literals-section header, read_sequences!'s Symbol_Compression_Modes +
    # distribution-table header, and (via read_huffman_description's
    # fbr_scratch) _read_fse_weights!'s FSE-encoded-weights header. All three
    # run to completion and are discarded before the next one is ever
    # constructed within one block's processing, so one shared field is safe.
    fbr ::ForwardBitReader{RBRView}
end

const _FSE_MAX_TABLE = 512   # 1 << max accuracy_log (9 for LL/ML, 8 for OF)
const _HUF_MAX_TABLE = 1 << HUFTABLE_LOG_MAX  # largest possible Huffman decode table (2048 entries)
const _HUF_MAX_SYMBOLS = 256  # one weight per possible byte value
const ZSTD_BLOCKSIZE_MAX = 131072  # maximum decompressed size of any single block (RFC 8878)

# Ceiling on the compression ratio used to size the output buffer when a frame
# declares no Frame_Content_Size (see _decompress_frame!). Bounds how far a
# misleading prefix can inflate the reservation; ratios above it just grow the
# buffer again, as before.
const _GROWTH_RATIO_CLAMP = 8

# Placeholder data for scratch readers, overwritten by `reinit!` before real
# use. Must be a valid (non-empty, sentinel-terminated) reverse bitstream.
_dummy_rbr_view() = @view UInt8[0x01][1:1]

# fse_occ/fse_norm are shared scratch for read_distribution_table! (LL/OF/ML,
# built sequentially, see DecompressState's field comment). For any input
# that doesn't get rejected by read_distribution_table!'s own
# length(dist) ≤ max_sym+1 check, norm can never exceed MAX_MATCH_LENGTH+1
# (53) entries -- the largest of the three real max_sym+1 values (LL: 36,
# OF: 32, ML: 53). Pre-sizing to that bound means the push! loop in
# read_fse_dist! never has to grow the buffer for any input that will
# actually succeed; a fresh DecompressState is constructed per frame, so
# without this every LL/OF/ML call on every frame would otherwise re-grow
# from empty via several reallocations to reach the same ~53 elements.
_new_fse_scratch() = (sizehint!(Int[], MAX_MATCH_LENGTH + 1), sizehint!(Int16[], MAX_MATCH_LENGTH + 1))

DecompressState() = DecompressState(
    INIT_REPEAT_OFFSETS, nothing, nothing, nothing, nothing, UInt8[],
    UInt8[],
    zeros(Int, HUFTABLE_LOG_MAX + 1),
    zeros(Int, HUFTABLE_LOG_MAX + 1),
    sizehint!(UInt8[], _HUF_MAX_SYMBOLS),
    Vector{UInt8}(undef, _HUF_MAX_TABLE),
    FSEDistTableSlot(_FSE_MAX_TABLE),
    FSEDistTableSlot(_FSE_MAX_TABLE),
    FSEDistTableSlot(_FSE_MAX_TABLE),
    _new_fse_scratch()...,
    ReverseBitReader(_dummy_rbr_view()),
    ReverseBitReader(_dummy_rbr_view()),
    Huffman4StreamScratch(_dummy_rbr_view()),
    ForwardBitReader(_dummy_rbr_view()))

DecompressState(dict::ZstdDict) =
    DecompressState(
        dict.rep, dict.huffman, dict.ll_tab, dict.ml_tab, dict.of_tab, dict.content,
        UInt8[],
        zeros(Int, HUFTABLE_LOG_MAX),
        zeros(Int, HUFTABLE_LOG_MAX),
        sizehint!(UInt8[], _HUF_MAX_SYMBOLS),
        Vector{UInt8}(undef, _HUF_MAX_TABLE),
        FSEDistTableSlot(_FSE_MAX_TABLE),
        FSEDistTableSlot(_FSE_MAX_TABLE),
        FSEDistTableSlot(_FSE_MAX_TABLE),
        _new_fse_scratch()...,
        ReverseBitReader(_dummy_rbr_view()),
        ReverseBitReader(_dummy_rbr_view()),
        Huffman4StreamScratch(_dummy_rbr_view()),
        ForwardBitReader(_dummy_rbr_view()))


"""
    inflate_zstd(data::Vector{UInt8}; dict = nothing, nthreads = Threads.nthreads()) -> Vector{UInt8}

Decompress one or more concatenated Zstandard frames from `data` and return
the raw bytes. Skippable frames (RFC 8878 §3.1.2) are silently ignored.

When `nthreads ≥ 2` and `data` contains two or more independent zstd frames,
each frame is decompressed in a separate Julia task (capped at `nthreads`
concurrent tasks). Results are concatenated in frame order. With `nthreads=1`
or a single-frame input the existing serial path is taken unchanged.

`nthreads` must be ≥ 1; passing 0 or negative throws `ArgumentError`.

If the frame was compressed with a dictionary, pass a `ZstdDict` as `dict`.
Use `Base.parse(ZstdDict, bytes)` to construct one from raw bytes.
"""
function inflate_zstd(data::Vector{UInt8}; dict::Union{ZstdDict,Nothing} = nothing, nthreads::Int = Threads.nthreads())
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
    inflate_zstd(filename::AbstractString; dict = nothing, nthreads = Threads.nthreads()) -> String

Read a `.zst` file and return the decompressed content as a `String`.
"""
function inflate_zstd(filename::AbstractString; dict::Union{ZstdDict,Nothing} = nothing, nthreads::Int = Threads.nthreads())
    data = read(filename)
    String(inflate_zstd(data; dict=dict, nthreads=nthreads))
end


@inline _is_skippable(magic::UInt32) = (magic & 0xFFFFFFF0) == ZSTD_SKIPPABLE_FRAME_MAGIC

# Skip a skippable frame (RFC 8878 §3.1.2). Returns the position after the frame.
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
    window_descriptor = Int(data[pos])
    # Exponent is bits 7-3, mantissa bits 2-0
    exponent = window_descriptor >> 3
    mantissa = window_descriptor & 0x07
    # Widened deliberately to Int64 to avoid overflow on 32-bit builds
    window_base = Int64(1) << (10 + exponent)
    window_size = window_base + (window_base >> 3) * mantissa
    window_size ≤ WINDOW_SIZE_MAX ||
        throw(ArgumentError("zstd: window size $window_size exceeds maximum representable"))
    return Int(window_size), pos + 1
end

function _read_frame_header(data::Vector{UInt8}, pos::Int, dict::Union{ZstdDict, Nothing})
    # Frame Header Descriptor (RFC 8878 §3.1.1.1.1)
    fcs_size, single_segment_flag, content_checksum_flag, dict_id_size, pos = _read_frame_header_descriptor(data, pos)

    # Window Descriptor (RFC 8878 §3.1.1.1.2, omitted when Single_Segment_Flag is set).
    window_size, pos = _read_window_descriptor(data, pos, single_segment_flag)

    # Dictionary ID (RFC 8878 §3.1.1.1.3)
    pos = _read_and_validate_dict_id(data, pos, dict_id_size, dict)

    # Frame Content Size (RFC 8878 §3.1.1.1.4)
    frame_content_size, pos = _read_frame_content_size(data, pos, fcs_size)

    if single_segment_flag
        frame_content_size ≥ 0 ||
            throw(ArgumentError("zstd: single-segment frame with unknown content size"))
        window_size = frame_content_size
    end

    return window_size, frame_content_size, content_checksum_flag, pos
end

"""
    FrameInfo

Lightweight descriptor for a single non-skippable zstd frame found during
pre-scan. `data_start` is the 1-based byte offset of the frame's 4-byte
magic number in the input vector. `fcs` is the declared decompressed size
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
            fcs_size, single_segment_flag, content_checksum_flag, dict_id_size, pos = _read_frame_header_descriptor(data, pos)
            _, pos = _read_window_descriptor(data, pos, single_segment_flag)
            pos = _read_and_validate_dict_id(data, pos, dict_id_size, dict)
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
                block_size ≤ ZSTD_BLOCKSIZE_MAX ||
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

# RFC 8878 §3.1.1
function _decompress_frame!(data::Vector{UInt8}, pos::Int, out::Vector{UInt8},
                            dict::Union{ZstdDict, Nothing} = nothing)
    # Magic number (4 bytes, little-endian)
    magic = _read_magic(data, pos)
    magic == ZSTD_MAGIC ||
        throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base = 16))"))
    pos += 4

    # Frame Header Descriptor (FHD)
    window_size, frame_content_size, content_checksum_flag, pos = _read_frame_header(data, pos, dict)

    state = dict !== nothing ? DecompressState(dict) : DecompressState()
    frame_start = length(out)
    # When FCS is known, resize to the exact frame size upfront so that all
    # per-block writes go directly into pre-allocated space — no per-block
    # resize! needed. When FCS is unknown, rely on the caller's one-time
    # sizehint! and fall back to per-block resize!.
    #
    # out_limit is the last byte position any block may write. Every write
    # path validates against it before touching memory, so a malicious frame
    # that declares a small FCS but encodes more output cannot write past the
    # preallocated buffer.
    preallocated = frame_content_size ≥ 0
    if preallocated
        resize!(out, frame_start + frame_content_size + WILDCOPY_SLACK)
    end
    out_limit = preallocated ? frame_start + frame_content_size : typemax(Int) - WILDCOPY_SLACK
    wpos = frame_start + 1
    data_start = pos

    # Decode blocks
    while true
        # With no Frame_Content_Size to size `out` up front, the fallback is
        # read_sequences!'s per-block resize!, which asks for one block more
        # than it has. Julia's geometric over-allocation absorbs most of that,
        # but stepping to 10 MB a block at a time still costs 10 allocations and
        # 40 MB of copying where knowing the size costs 2 and 9.5 MB.
        #
        # Once a block or two has been decoded the frame's own compression ratio
        # is known, so extrapolate the rest of the buffer from it in one step.
        # `remaining` counts every byte left in `data`; for concatenated frames
        # that includes bytes belonging to later frames, but those append to
        # this same `out`, so extrapolating over all of them estimates the final
        # length rather than overshooting it.
        #
        # The guess is only as good as the ratio, and a degenerate opening block
        # (an RLE block is 128 KB of output from three bytes of input) would
        # extrapolate to something absurd, so the ratio is clamped. Real data
        # sits well under the clamp, so it does not bind; input that genuinely
        # exceeds it merely falls back to growing again, which is what would
        # have happened anyway. Only ever grows, and read_sequences! keeps its
        # own resize! as the correctness backstop.
        if !preallocated
            produced = wpos - 1 - frame_start
            consumed = pos - data_start
            if produced ≥ ZSTD_BLOCKSIZE_MAX && consumed > 0
                remaining = length(data) - pos + 1
                # Multiply before dividing: the ratio is usually between 1 and 2,
                # so computing it as an integer first would floor it to 1 and
                # collapse the estimate back to plain growth. widemul keeps the
                # product exact for inputs where it would otherwise overflow.
                rest   = min(widemul(remaining, produced) ÷ consumed,
                             widemul(remaining, _GROWTH_RATIO_CLAMP))
                target = frame_start + produced +
                         Int(min(rest, typemax(Int) ÷ 4)) + WILDCOPY_SLACK
                length(out) < target && resize!(out, target)
            end
        end

        length(data) ≥ pos + 2 ||
            throw(ArgumentError("zstd: truncated block header"))
        # Block header is 3 bytes
        bh1, bh2, bh3 = Int(data[pos]), Int(data[pos + 1]), Int(data[pos + 2])
        pos += 3

        last_block  = bh1 & 0x01
        block_type  = (bh1 >> 1) & 0x03
        # block_size for compressed/raw/RLE: 21-bit field in remaining 21 bits
        block_size  = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
        block_size ≤ ZSTD_BLOCKSIZE_MAX ||
            throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))

        wpos = _apply_block!(block_type, data, pos, block_size, state, out, wpos,
                             preallocated, frame_start, out_limit)
        pos += block_type == 1 ? 1 : block_size  # RLE payload is a single byte

        last_block != 0 && break
    end

    # Trim slack bytes added for wildcopy16 over-writes
    resize!(out, wpos - 1)

    frame_len = wpos - 1 - frame_start

    # Validate Frame Content Size (RFC 8878 §3.1.1.1.4)
    frame_content_size < 0 || frame_content_size == frame_len ||
        throw(ArgumentError("zstd: decompressed size $frame_len does not match frame content size $frame_content_size"))

    # Content checksum (optional 4 bytes)
    if content_checksum_flag
        length(data) ≥ pos + 3 ||
            throw(ArgumentError("zstd: truncated content checksum"))
        stored = _le32(data, pos)
        pos += 4
        computed = UInt32(xxhash64(@view(out[frame_start+1:end])) & 0xFFFFFFFF)
        stored == computed ||
            throw(ArgumentError("zstd: content checksum mismatch (stored=0x$(string(stored,base=16)), computed=0x$(string(computed,base=16)))"))
    end

    return pos
end

@inline function _read_magic(data::Vector{UInt8}, pos::Int)
    length(data) ≥ pos + 3 ||
        throw(ArgumentError("zstd: truncated frame (magic)"))
    _le32(data, pos)
end

# Decode one block's payload (starting at data[pos]) into out at wpos and
# return the new write position. Shared by the in-memory frame decoder
# (_decompress_frame!) and the incremental stream decoder (streaming.jl).
function _apply_block!(block_type::Int, data::Vector{UInt8}, pos::Int, block_size::Int,
                       state::DecompressState, out::Vector{UInt8}, wpos::Int,
                       preallocated::Bool, frame_start::Int, out_limit::Int)
    if block_type == 0      # raw
        pos + block_size - 1 ≤ length(data) ||
            throw(ArgumentError("zstd: truncated raw block"))
        wpos - 1 + block_size ≤ out_limit ||
            throw(ArgumentError("zstd: block output exceeds declared frame content size"))
        preallocated ||
            resize!(out, wpos - 1 + block_size)
        GC.@preserve out data Base.memcpy(pointer(out, wpos), pointer(data, pos), block_size)
        return wpos + block_size
    elseif block_type == 1  # RLE: 1-byte payload repeated block_size times
        pos ≤ length(data) ||
            throw(ArgumentError("zstd: truncated RLE block"))
        wpos - 1 + block_size ≤ out_limit ||
            throw(ArgumentError("zstd: block output exceeds declared frame content size"))
        preallocated ||
            resize!(out, wpos - 1 + block_size)
        fill!(view(out, wpos:wpos + block_size - 1), data[pos])
        return wpos + block_size
    elseif block_type == 2  # compressed
        return _decompress_block!(data, pos, block_size, state, out, wpos, preallocated,
                                  frame_start, out_limit)
    else
        throw(ArgumentError("zstd: unsupported block type (reserved)"))
    end
end

function _decompress_block!(data::Vector{UInt8}, pos::Int, block_size::Int, state::DecompressState,
                           out::Vector{UInt8}, wpos::Int, preallocated::Bool,
                           frame_start::Int, out_limit::Int)
    limit = pos + block_size - 1
    limit ≤ length(data) ||
        throw(ArgumentError("zstd: truncated compressed block"))
    literals, lit_consumed = read_literals(data, pos, limit, state)
    seq_pos = pos + lit_consumed
    nextwpos = wpos
    if seq_pos ≤ limit
        nextwpos = read_sequences!(data, seq_pos, limit, state, literals, out, wpos, preallocated,
                                   frame_start, out_limit)
    end
    if nextwpos == wpos
        # No sequences section: the block is all literals (plus WILDCOPY_SLACK slack bytes)
        lit_len = length(literals) - WILDCOPY_SLACK
        wpos - 1 + lit_len ≤ out_limit ||
            throw(ArgumentError("zstd: block output exceeds declared frame content size"))
        if !preallocated
            resize!(out, wpos - 1 + lit_len + WILDCOPY_SLACK)
        end
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, 1), lit_len)
        return wpos + lit_len
    else
        return nextwpos
    end
end

# Reference: RFC 8878 §3.1.1.3.1
# `limit` is the last byte of the enclosing block; the literals section must
# fit entirely within it.
function read_literals(data::Vector{UInt8}, pos::Int, limit::Int, state::DecompressState)
    br = reinit!(state.fbr, @view data[pos:end])
    litblock_type = read(br, 2)
    size_format = peek(br, 2)

    is_raw = litblock_type == 0
    is_rle = litblock_type == 1
    is_treeless = litblock_type == 3

    if is_raw || is_rle
        size_format_nbits = iseven(size_format) ? 1 :
                                                  2
        size_nbits, header_nbytes = iseven(size_format) ? ( 5, 1) :
                                    size_format == 1    ? (12, 2) :
                                                          (20, 3)
        skip(br, size_format_nbits)
        regen_size = Int(read(br, size_nbits))
        compressed_size = is_raw ? regen_size :
                                   1

        regen_size ≤ ZSTD_BLOCKSIZE_MAX ||
            throw(ArgumentError("zstd: literals regenerated size $regen_size exceeds maximum block size"))
        pos + header_nbytes + compressed_size - 1 ≤ limit ||
            throw(ArgumentError("zstd: literals section exceeds block size"))

        literals = state.literals_buf
        resize!(literals, regen_size + WILDCOPY_SLACK)

        block_start = pos + header_nbytes

        if is_raw
            copyto!(literals, 1, data, block_start, regen_size)
        else
            literals[1:regen_size] .= data[block_start]
        end

        return literals, header_nbytes + compressed_size
    else # Compressed (2) or Treeless (3)
        size_nbits, header_nbytes = size_format < 2  ? (10, 3) :
                                    size_format == 2 ? (14, 4) :
                                                       (18, 5)
        num_streams = size_format == 0 ? 1 :
                                         4

        skip(br, 2)
        regen_size = Int(read(br, size_nbits))
        compressed_size = Int(read(br, size_nbits))

        regen_size ≤ ZSTD_BLOCKSIZE_MAX ||
            throw(ArgumentError("zstd: literals regenerated size $regen_size exceeds maximum block size"))
        pos + header_nbytes + compressed_size - 1 ≤ limit ||
            throw(ArgumentError("zstd: literals section exceeds block size"))
        compressed_size ≥ 1 ||
            throw(ArgumentError("zstd: empty compressed literals payload"))

        payload_start = pos + header_nbytes
        payload_end = payload_start + compressed_size - 1

        if is_treeless
            state.huffman !== nothing ||
                throw(ArgumentError("zstd: treeless literals but no prior Huffman table"))
            ht = state.huffman
            huf_start = payload_start
        else # Compressed
            # Safe to reuse rb_lit1 here: it's only touched later in this same
            # function, in the num_streams == 1 decode branch, well after the
            # Huffman table (and this reader's brief use inside it) is done.
            # A dedicated field for this alone was measured to cost more in
            # per-frame construction overhead than it saves at the call site
            # (see the "Reuse ReverseBitReader in _read_fse_weights!" work).
            ht, hdr_len = read_huffman_description((@view data[payload_start:payload_end]); scratch_buffers = (state.huf_weights, state.huf_rank_count, state.huf_rank_start, state.huf_nbits_sym1), rb_scratch = state.rb_lit1, fbr_scratch = state.fbr)
            state.huffman = ht
            huf_start = payload_start + hdr_len
        end

        # Each Huffman stream must be non-empty; the 4-stream layout additionally
        # needs a 6-byte jump table (RFC 8878 §3.1.1.3.1.6).
        huf_len = payload_end - huf_start + 1
        huf_len ≥ (num_streams == 4 ? 10 : 1) ||
            throw(ArgumentError("zstd: literals section too small for Huffman-coded streams"))

        literals = state.literals_buf
        resize!(literals, regen_size + WILDCOPY_SLACK)

        if num_streams == 1
            rb = reinit!(state.rb_lit1, @view data[huf_start:payload_end])
            let p = 1
                while p ≤ regen_size
                    p += decode1x2_tail!(rb, ht, literals, p)
                end
            end
        else
            _decode_4streams!((@view data[huf_start:payload_end]), ht, literals, regen_size, state.huf4)
        end

        resize!(literals, regen_size + WILDCOPY_SLACK)  # trim dual-symbol slack

        return literals, header_nbytes + compressed_size
    end
end

# Reference: RFC 8878 §3.1.1.3.2
function read_sequences!(data::Vector{UInt8}, pos::Int, limit::Int,
                         state::DecompressState, literals::Vector{UInt8},
                         out::Vector{UInt8}, wpos::Int, preallocated::Bool,
                         frame_start::Int, out_limit::Int)
    pos > limit && return wpos

    # RFC 8878 §3.1.1.3.2
    b0 = Int(data[pos]);  pos += 1
    local num_seqs::Int
    if b0 < 128
        num_seqs = b0
    elseif b0 < 255
        num_seqs = ((b0 - 128) << 8) | Int(data[pos]);  pos += 1
    else
        num_seqs = Int(data[pos]) + (Int(data[pos+1]) << 8) + 0x7F00;  pos += 2
    end

    num_seqs == 0 && return wpos

    # Symbol Compression Modes byte (RFC 8878 §3.1.1.3.2.1)
    modes_byte = Int(data[pos]);  pos += 1
    modes_byte & 0x03 == 0 || throw(ArgumentError("zstd: reserved bits set in Symbol_Compression_Modes, must be zero"))
    ll_mode = (modes_byte >> 6) & 0x03
    of_mode = (modes_byte >> 4) & 0x03
    ml_mode = (modes_byte >> 2) & 0x03

    br = reinit!(state.fbr, @view data[pos:limit])
    ll_tab = read_distribution_table!(br, DEFAULT_LITERALS_LENGTH_TABLE, state.ll_tab, ll_mode, MAX_LITERALS_LENGTH, 9,
                             state.ll_slot, state, SeqLL())
    of_tab = read_distribution_table!(br, DEFAULT_OFFSET_TABLE, state.of_tab, of_mode, MAX_OFFSET_CODE, 8,
                             state.of_slot, state, SeqOF())
    ml_tab = read_distribution_table!(br, DEFAULT_MATCH_LENGTH_TABLE, state.ml_tab, ml_mode, MAX_MATCH_LENGTH, 9,
                             state.ml_slot, state, SeqML())
    state.ll_tab = ll_tab
    state.of_tab = of_tab
    state.ml_tab = ml_tab

    # The bitstream for sequences is a reverse bitstream starting right after
    # the distribution table descriptions.
    seq_start = byte_pos(br)
    seq_len = limit - seq_start + 1
    seq_len > 0 || throw(ArgumentError("zstd: no data for sequences bitstream"))

    rb = reinit!(state.rb_seq, @view data[seq_start:seq_start + seq_len - 1])

    # Init distribution table states
    ll_state = dist_table_init!(rb, ll_tab)
    of_state = dist_table_init!(rb, of_tab)
    ml_state = dist_table_init!(rb, ml_tab)

    lit_len = length(literals) - WILDCOPY_SLACK

    # Upper bound on where this block may write. Having a constant bound lets the fused
    # loop below check each sequence with a single compare instead of pre-scanning every
    # sequence length.
    block_limit = min(out_limit, wpos - 1 + ZSTD_BLOCKSIZE_MAX)

    # When the frame size is unknown the caller has not pre-sized `out`; reserve a
    # whole block once, here, rather than growing per sequence. Only ever grows; trimmed
    # at the end.
    if !preallocated
        need = block_limit + WILDCOPY_SLACK
        length(out) < need && resize!(out, need)
    end

    # Each table being a union makes this call dynamic due to too many combinations.
    # The dynamic call leads to boxing of the three Int arguments. By manually type
    # checking the most common combinations (FFF, FFR, FRF), these calls can be made static.
    if ll_tab isa FSEDistTable
        if of_tab isa FSEDistTable && ml_tab isa FSEDistTable
            # Empirically overwhelming case
            return _run_sequences!(ll_tab, of_tab, ml_tab, rb, ll_state, of_state, ml_state,
                                num_seqs, literals, lit_len, state, out, wpos,
                                frame_start, block_limit, out_limit)
        elseif of_tab isa FSEDistTable && ml_tab isa RLEDistTable
            return _run_sequences!(ll_tab, of_tab, ml_tab, rb, ll_state, of_state, ml_state,
                                num_seqs, literals, lit_len, state, out, wpos,
                                frame_start, block_limit, out_limit)
        elseif of_tab isa RLEDistTable && ml_tab isa FSEDistTable
            return _run_sequences!(ll_tab, of_tab, ml_tab, rb, ll_state, of_state, ml_state,
                                num_seqs, literals, lit_len, state, out, wpos,
                                frame_start, block_limit, out_limit)
        end
    end

    # Dynamic dispatch for all other combinations
    return _run_sequences!(ll_tab, of_tab, ml_tab, rb, ll_state, of_state, ml_state,
                           num_seqs, literals, lit_len, state, out, wpos,
                           frame_start, block_limit, out_limit)
end

@noinline _throw_seq_bitstream() =
    throw(ArgumentError("zstd: unexpected end of sequence bitstream"))
@noinline _throw_literals_overrun() =
    throw(ArgumentError("zstd: sequences reference more literals than the block provides"))
@noinline _throw_repeat_offset_zero() =
    throw(ArgumentError("zstd: repeat offset - 1 is zero"))
@noinline _throw_dict_offset(offset) =
    throw(ArgumentError("zstd: match offset $offset beyond dictionary and output"))
@noinline function _throw_block_output(needed::Int, out_limit::Int)
    needed > out_limit &&
        throw(ArgumentError("zstd: block output exceeds declared frame content size"))
    throw(ArgumentError("zstd: block decompressed size exceeds maximum (128 KB)"))
end

# Decode and execute the sequences section in a single pass, so the sequence values never
# reach memory.
#
# Reference: RFC 8878 §3.1.1.3.2 (decode) and §3.1.1.4 (execute)
function _run_sequences!(ll_tab::T1, of_tab::T2, ml_tab::T3,
                         rb::ReverseBitReader, ll_state::Int, of_state::Int, ml_state::Int,
                         num_seqs::Int, literals::Vector{UInt8}, lit_len::Int,
                         state::DecompressState, out::Vector{UInt8}, wpos::Int,
                         frame_start::Int, block_limit::Int, out_limit::Int) where {T1, T2, T3}
    lit_pos = 1
    # Repeat offsets and dictionary content are loop-invariant; keeping `rep` in a
    # local tuple avoids a load and store through the mutable state per sequence.
    rep      = state.rep
    dict     = state.dict_content
    dict_len = length(dict)

    @inbounds for i in 1:num_seqs
        # One packed entry per table carries this state's value baseline, its
        # extra-bit count, and its state transition, so each table is a single
        # bounds-checked load. Nothing here depends on a symbol first, which is
        # the point: `total_n` below used to sit behind a second, dependent
        # lookup into the baseline/extra-bits tables.
        ll_e = dist_table_entry(ll_tab, ll_state)
        ml_e = dist_table_entry(ml_tab, ml_state)
        of_e = dist_table_entry(of_tab, of_state)

        of_n  = _fse_add(of_e)     # offset codes are their own extra-bit count
        ml_n  = _fse_add(ml_e)
        ll_n  = _fse_add(ll_e)

        # State-transition widths (skip on last sequence)
        update = i < num_seqs
        ll_nb = update ? _fse_nb(ll_e) : 0
        ml_nb = update ? _fse_nb(ml_e) : 0
        of_nb = update ? _fse_nb(of_e) : 0

        total_n = of_n + ml_n + ll_n + ll_nb + ml_nb + of_nb

        # Read in batches for optimal ILP
        if total_n ≤ 57
            # Fast path: a single refill guarantees ≥ 57 bits available.
            rb.nbits < total_n && refill!(rb)
            rb.nbits ≥ total_n || _throw_seq_bitstream()

            # Cumulative bit offsets into the frozen snapshot
            c_ml  = of_n
            c_ll  = c_ml + ml_n
            c_llb = c_ll + ll_n
            c_mlb = c_llb + ll_nb
            c_ofb = c_mlb + ml_nb

            bits     = rb.bits
            of_extra = Int(ifelse(of_n  > 0, _shr(bits,              64 - of_n ), 0))
            ml_extra = Int(ifelse(ml_n  > 0, _shr(_shl(bits, c_ml ), 64 - ml_n ), 0))
            ll_extra = Int(ifelse(ll_n  > 0, _shr(_shl(bits, c_ll ), 64 - ll_n ), 0))
            ll_bits  = Int(ifelse(ll_nb > 0, _shr(_shl(bits, c_llb), 64 - ll_nb), 0))
            ml_bits  = Int(ifelse(ml_nb > 0, _shr(_shl(bits, c_mlb), 64 - ml_nb), 0))
            of_bits  = Int(ifelse(of_nb > 0, _shr(_shl(bits, c_ofb), 64 - of_nb), 0))

            # Single bulk consume
            rb.bits   = bits << total_n
            rb.nbits -= total_n

        else
            # Slow path: large offset (of_code > ~20); sequential reads.
            of_extra = Int(read(rb, of_n ))
            ml_extra = Int(read(rb, ml_n))
            ll_extra = Int(read(rb, ll_n))
            ll_bits  = Int(read(rb, ll_nb))
            ml_bits  = Int(read(rb, ml_nb))
            of_bits  = Int(read(rb, of_nb))
        end

        # Baselines come straight out of the entries. The offset baseline is
        # 1 << of_code, baked at build time, and of_code ≤ MAX_OFFSET_CODE (31)
        # is enforced there too, so the sum cannot overflow Int.
        of = _fse_base(of_e) + of_extra
        ml = _fse_base(ml_e) + ml_extra
        ll = _fse_base(ll_e) + ll_extra

        # Advance the distribution-table states before emitting output; allows the CPU
        # to issue loads for the next sequence while the current sequence is being executed.
        if update
            ll_state = _fse_next(ll_e) + ll_bits
            ml_state = _fse_next(ml_e) + ml_bits
            of_state = _fse_next(of_e) + of_bits
        end

        # Bounds, in place of the pre-scan the two-pass version used to do. Both are
        # loop-carried compares against values already in registers.
        lit_pos + ll - 1 ≤ lit_len       || _throw_literals_overrun()
        wpos - 1 + ll + ml ≤ block_limit || _throw_block_output(wpos - 1 + ll + ml, out_limit)

        # Copy ll literal bytes. out and literals are distinct arrays so no overlap is possible.
        if ll > 0
            GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), ll)
            wpos    += ll
            lit_pos += ll
        end

        # Determine actual offset from repeat-offset table
        # of is the raw Offset_Value; 1/2/3 are repeat codes, ≥4 is a new offset.
        local offset::Int
        if of > 3
            offset = of - 3
            rep = (offset, rep[1], rep[2])
        elseif ll > 0
            # Normal repeat-offset rules
            if of == 1
                offset = rep[1]
                # no rep update
            elseif of == 2
                offset = rep[2]
                rep = (rep[2], rep[1], rep[3])
            else  # of == 3
                offset = rep[3]
                rep = (rep[3], rep[1], rep[2])
            end
        else
            # LL==0: repeat-offset references shift up by 1
            if of == 1
                offset = rep[2]
                rep = (rep[2], rep[1], rep[3])
            elseif of == 2
                offset = rep[3]
                rep = (rep[3], rep[1], rep[2])
            else  # of == 3
                offset = rep[1] - 1
                offset > 0 || _throw_repeat_offset_zero()
                rep = (offset, rep[1], rep[2])
            end
        end

        # Copy match of length ml from offset back in output.
        # The match may reach behind the current frame's start, in which case
        # the bytes come from the dictionary content prefix — never from a
        # preceding frame's output (RFC 8878 §3.1.1.4).
        # wpos - 1 is the logical end of written output; match_pos is 1-indexed into out.
        match_pos = wpos - offset   # = (wpos - 1) - offset + 1
        if match_pos ≤ frame_start
            # Offset reaches into dictionary content. match_pos advances in
            # lockstep with dict_pos so that when the copy crosses back into
            # frame output it continues from out[frame_start + 1].
            dict_pos = dict_len + (match_pos - frame_start)   # 1-indexed into dict
            dict_pos ≥ 1 || _throw_dict_offset(offset)
            for _ in 1:ml
                if dict_pos ≤ dict_len
                    out[wpos] = dict[dict_pos]
                    dict_pos += 1
                else
                    out[wpos] = out[match_pos]
                end
                wpos      += 1
                match_pos += 1
            end
        else
            if offset ≥ ml
                # Non-overlapping match. For short copies, _wildcopy16! avoids the
                # libc memcpy FFI call; for larger copies memcpy wins (wider SIMD).
                if ml ≤ 64
                    GC.@preserve out _wildcopy16!(pointer(out, wpos), pointer(out, match_pos), ml)
                else
                    GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), ml)
                end
            elseif offset == 1
                # Single-byte repeat: fill
                fill!(view(out, wpos:wpos+ml-1), out[match_pos])
            else
                # Overlapping repeat pattern: copy base pattern once, then
                # keep doubling by copying already-written output. Each
                # memcpy is non-overlapping (filled bytes precede dest).
                GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), offset)
                filled = offset
                while filled < ml
                    to_copy = min(filled, ml - filled)
                    GC.@preserve out Base.memcpy(pointer(out, wpos + filled), pointer(out, wpos), to_copy)
                    filled += to_copy
                end
            end
            wpos += ml
        end
    end

    # Publish the repeat offsets once, after the whole block.
    state.rep = rep

    # Remaining literals after the last sequence.
    rem = lit_len - lit_pos + 1
    if rem > 0
        wpos - 1 + rem ≤ block_limit || _throw_block_output(wpos - 1 + rem, out_limit)
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), rem)
        wpos += rem
    end
    return wpos
end

function read_distribution_table!(br::ForwardBitReader, default::FSEDistTable,
                         prev::Union{FSEDistTable, RLEDistTable, Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         slot::FSEDistTableSlot, occ::Vector{Int},
                         norm::Vector{Int16}, kind)
    if mode == 0 # Predefined_Mode
        return default
    elseif mode == 1 # RLE_Mode
        sym = Int(read(br, 8))
        # `_rle_table` range-checks the symbol against the kind before baking
        # its baseline and extra-bit count into the entry.
        return _rle_table(kind, sym)
    elseif mode == 2 # FSE_Compressed_Mode
        al, dist = read_fse_dist!(br, norm)
        al ≤ max_al || throw(ArgumentError("zstd: accuracy log $al exceeds maximum $max_al"))
        length(dist) ≤ max_sym + 1 || throw(ArgumentError("zstd: FSE distribution has $(length(dist)) symbols, maximum is $(max_sym + 1)"))
        return build_fse_table!(dist, al, slot, occ, kind)
    else # mode == 3; Repeat_Mode
        prev !== nothing || throw(ArgumentError("zstd: repeat mode but no previous distribution table"))
        return prev
    end
end

function read_distribution_table!(br::ForwardBitReader, default::FSEDistTable,
                         prev::Union{FSEDistTable,RLEDistTable,Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         slot::FSEDistTableSlot, state::DecompressState, kind)
    return read_distribution_table!(br, default, prev, mode, max_sym, max_al,
                           slot, state.fse_occ, state.fse_norm, kind)
end
