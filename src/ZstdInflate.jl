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

include("constants.jl")
include("util.jl")
include("xxhash.jl")
include("forwardbitreader.jl")
include("reversebitreader.jl")
include("fse.jl")
include("huffman.jl")
include("dictionary.jl")

export inflate_zstd, InflateZstdStream, ZstdDict, parse_dictionary


const _LL_TABLE = Ref{FSETable}()
const _ML_TABLE = Ref{FSETable}()
const _OF_TABLE = Ref{FSETable}()

function __init__()
    _LL_TABLE[] = build_fse_table(LITERALS_LENGTH_DEFAULT_DIST, LITERALS_LENGTH_ACCURACY_LOG)
    _ML_TABLE[] = build_fse_table(MATCH_LENGTH_DEFAULT_DIST, MATCH_LENGTH_ACCURACY_LOG)
    _OF_TABLE[] = build_fse_table(OFFSET_DEFAULT_DIST, OFFSET_ACCURACY_LOG)
end


# ============================================================
# Section 8: Decompression state + literals section
#   Reference: RFC 8878 §3.1.1.3
# ============================================================

# Groups the three backing arrays for one FSE decode table (LL, OF, or ML).
# Having all three in one object lets callers pass a single slot instead of
# three separate vectors, and keeps DecompressState compact.
mutable struct FSETableSlot
    syms ::Vector{UInt8}
    nb   ::Vector{UInt8}
    base ::Vector{UInt32}
end

FSETableSlot(n::Int) = FSETableSlot(Vector{UInt8}(undef, n),
                                     Vector{UInt8}(undef, n),
                                     Vector{UInt32}(undef, n))

mutable struct DecompressState
    rep         ::NTuple{3,Int}
    huffman     ::Union{HuffmanTable,Nothing}
    ll_tab      ::Union{FSETable,RLEFSETable,Nothing}
    ml_tab      ::Union{FSETable,RLEFSETable,Nothing}
    of_tab      ::Union{FSETable,RLEFSETable,Nothing}
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
    ll_slot ::FSETableSlot
    ml_slot ::FSETableSlot
    of_slot ::FSETableSlot
    # Shared FSE build scratch — safe to share because LL/ML/OF are built sequentially
    fse_occ  ::Vector{Int}
    fse_norm ::Vector{Int16}
end

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
    FSETableSlot(_FSE_MAX_TABLE),
    FSETableSlot(_FSE_MAX_TABLE),
    FSETableSlot(_FSE_MAX_TABLE),
    Int[], Int16[])

function DecompressState(dict::ZstdDict)
    DecompressState(
        dict.rep, dict.huffman, dict.ll_tab, dict.ml_tab, dict.of_tab, dict.content,
        Int[], Int[], Int[],
        UInt8[],
        Vector{UInt32}(undef, 1 << HUFTABLE_LOG_MAX),
        zeros(Int, HUFTABLE_LOG_MAX + 1),
        zeros(Int, HUFTABLE_LOG_MAX + 1),
        UInt8[],
        FSETableSlot(_FSE_MAX_TABLE),
        FSETableSlot(_FSE_MAX_TABLE),
        FSETableSlot(_FSE_MAX_TABLE),
        Int[], Int16[])
end

# State-aware overload: routes FSE table backing storage through a slot and state.
# Defined here (after FSETableSlot and DecompressState) to avoid forward references.
function read_fse_table!(br::ForwardBitReader, default::FSETable,
                         prev::Union{FSETable,RLEFSETable,Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         slot::FSETableSlot, state::DecompressState)
    return read_fse_table!(br, default, prev, mode, max_sym, max_al,
                           slot.syms, slot.nb, slot.base,
                           state.fse_occ, state.fse_norm)
end

# Hot-path variant of build_huffman_table: reuses scratch buffers from DecompressState.
# huf_dtable is pre-sized to 1 << HUFTABLE_LOG_MAX and used as the full overflow table.
# huf_primary_table is pre-sized to 1 << HUF_PRIMARY_BITS and used as the hot-path table.
# Both buffers are shared across blocks (safe because literals are decoded before the
# next block's table is built, so the current HuffmanTable is not live during a rebuild).
function build_huffman_table(weights::Vector{UInt8}, max_bits::Int, state::DecompressState)
    nsyms = length(weights)
    max_bits > 0 || throw(ArgumentError("zstd: all-zero Huffman weights"))
    max_bits ≤ HUFTABLE_LOG_MAX || throw(ArgumentError("zstd: Huffman table log $max_bits exceeds maximum ($HUFTABLE_LOG_MAX)"))

    table_size      = 1 << max_bits
    dtable          = state.huf_dtable
    rank_count      = state.huf_rank_count
    next_rank_start = state.huf_rank_start

    # Pass 1: fill dtable with single-symbol entries (nb_total=nb1, nb2=0, sym2=0).
    fill!(view(rank_count,      1:max_bits+1), 0)
    fill!(view(next_rank_start, 1:max_bits+1), 0)

    for sym in 0:nsyms-1
        w = Int(weights[sym+1])
        w > 0 && (rank_count[w] += 1)
    end
    pos = 0
    for w in 1:max_bits
        next_rank_start[w] = pos
        pos += rank_count[w] * (1 << (w - 1))
    end
    for sym in 0:nsyms-1
        w = Int(weights[sym+1])
        w == 0 && continue
        nb1         = max_bits - w + 1
        num_entries = 1 << (w - 1)
        entry = UInt32(nb1 << 24) | UInt32(nb1 << 16) | UInt32(sym)
        start = next_rank_start[w]
        for j in 0:num_entries-1
            dtable[start + j + 1] = entry
        end
        next_rank_start[w] += num_entries
    end

    # Pass 2: upgrade dtable entries to dual-symbol where a second symbol fits.
    for idx in 0:table_size-1
        e1       = dtable[idx + 1]
        sym1     = Int(e1 & 0xFF)
        nb1      = Int((e1 >> 16) & 0xFF)
        rem      = max_bits - nb1
        if rem > 0
            idx2 = (idx << nb1) & (table_size - 1)
            e2   = dtable[idx2 + 1]
            nb2  = Int((e2 >> 16) & 0xFF)
            if nb2 ≤ rem
                sym2 = Int(e2 & 0xFF)
                dtable[idx + 1] = UInt32((nb1 + nb2) << 24) | UInt32(nb1 << 16) |
                                  UInt32(sym2 << 8) | UInt32(sym1)
                continue
            end
        end
        # Leave as single-symbol (already correct from pass 1).
    end

    return HuffmanTable{max_bits}(dtable)
end

# Hot-path variant of _decode_fse_weights: reuses a caller-supplied buffer.
function _decode_fse_weights(br::ForwardBitReader, byte_limit::Int, weights::Vector{UInt8})
    al, dist  = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t         = build_fse_table(dist, al)

    pos_after = byte_pos(br)
    n_remain  = byte_limit - pos_after + 1
    n_remain > 0 || throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    rb = ReverseBitReader(@view br.data[pos_after:pos_after+n_remain-1])

    state1 = fse_init!(rb, t)
    state2 = fse_init!(rb, t)

    empty!(weights)
    while true
        sym1 = fse_peek(t, state1)
        state1 = _fse_update_unchecked(rb, t, state1)
        push!(weights, UInt8(sym1))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(fse_peek(t, state2)))
            break
        end

        sym2 = fse_peek(t, state2)
        state2 = _fse_update_unchecked(rb, t, state2)
        push!(weights, UInt8(sym2))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(fse_peek(t, state1)))
            break
        end
    end

    br.pos   = byte_limit + 1
    br.nbits = 0
    br.bits  = UInt64(0)

    return weights
end

# Hot-path variant of read_huffman_description: reuses state scratch buffers.
function read_huffman_description(data::Vector{UInt8}, pos::Int, state::DecompressState)
    header = Int(data[pos])
    if header < 128
        compressed_size = header
        br = ForwardBitReader(@view data[pos+1:pos+compressed_size])
        weights = _decode_fse_weights(br, compressed_size, state.huf_weights)
        last_sym, last_w, table_log = _infer_last_weight(weights)
        push!(weights, UInt8(last_w))
        ht = build_huffman_table(weights, table_log, state)
        return ht, compressed_size + 1
    else
        nsyms  = header - 127
        nbytes = (nsyms + 1) >> 1
        weights = state.huf_weights
        resize!(weights, nsyms)
        for i in 1:nbytes
            b = data[pos + i]
            lo = b & 0x0f
            hi = (b >> 4) & 0x0f
            idx = (i-1)*2
            if idx + 1 ≤ nsyms;  weights[idx + 1] = hi;  end
            if idx + 2 ≤ nsyms;  weights[idx + 2] = lo;  end
        end
        last_sym, last_w, table_log = _infer_last_weight(weights)
        if last_sym ≤ nsyms
            weights[last_sym] = UInt8(last_w)
        else
            push!(weights, UInt8(last_w))
        end
        ht = build_huffman_table(weights, table_log, state)
        return ht, nbytes + 1
    end
end

# Decode four interleaved Huffman streams into `literals[1:regen_size]`.
# ht::HuffmanTable{L} carries max_bits as a type parameter so that
# safe_n = 57 ÷ L is a compile-time constant; Julia specialises this
# function per distinct L and LLVM unrolls the constant-bound inner loops.
# Called via dynamic dispatch from read_literals — one indirect call per
# outer (refill) iteration, not per symbol.
function _decode_4streams!(data::AbstractVector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int) where L
    s1_start = 7
    s2_start = s1_start + Int(_le16(data, 1))
    s3_start = s2_start + Int(_le16(data, 3))
    s4_start = s3_start + Int(_le16(data, 5))
    s4_end = length(data)

    seg_n = (regen_size + 3) >> 2

    rb1 = ReverseBitReader(@view data[s1_start:s2_start-1])
    rb2 = ReverseBitReader(@view data[s2_start:s3_start-1])
    rb3 = ReverseBitReader(@view data[s3_start:s4_start-1])
    rb4 = ReverseBitReader(@view data[s4_start:s4_end])

    # Dual-symbol interleaved decode.
    # safe_n lookups fit in one refill (≥ 57 bits loaded, ≤ max_bits per lookup).
    # safeendX = endX - 2*safe_n: the last position from which a full batch of
    # safe_n dual-symbol lookups is guaranteed safe.  Each lookup writes pos and
    # pos+1 unconditionally; the guard ensures pos+1 ≤ endX - 1 < endX, so the
    # write never crosses into an adjacent segment.
    # Phase 1: all 4 streams together.
    # Phase 2: fixed pairs (1,2) and (3,4).
    # Phase 3: whichever stream in each pair still has capacity runs single-stream.
    # Phase 4: finish any remaining symbols with single-symbol decode.
    safe_n = 57 ÷ L  # compile-time constant: L is a type parameter
    o1 = 1
    o2 = 1 + seg_n
    o3 = 1 + 2seg_n
    o4 = 1 + 3seg_n
    end1 = seg_n
    end2 = 2seg_n
    end3 = 3seg_n
    end4 = regen_size
    safeend1 = end1 - 2safe_n
    safeend2 = end2 - 2safe_n
    safeend3 = end3 - 2safe_n
    safeend4 = end4 - 2safe_n

    while o1 ≤ safeend1 && o2 ≤ safeend2 && o3 ≤ safeend3 && o4 ≤ safeend4
        refill!(rb1)
        refill!(rb2)
        refill!(rb3)
        refill!(rb4)
        for _ in 1:safe_n
            o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
            o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
        end
    end

    while o1 ≤ safeend1 && o2 ≤ safeend2
        refill!(rb1)
        refill!(rb2)
        for _ in 1:safe_n
            o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
        end
    end

    while o3 ≤ safeend3 && o4 ≤ safeend4
        refill!(rb3)
        refill!(rb4)
        for _ in 1:safe_n
            o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
        end
    end

    if o1 ≤ safeend1
        while o1 ≤ safeend1
            refill!(rb1)
            for _ in 1:safe_n
                o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            end
        end
    elseif o2 ≤ safeend2 # must check because possible to overshoot
        while o2 ≤ safeend2
            refill!(rb2)
            for _ in 1:safe_n
                o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
            end
        end
    end

    if o3 ≤ safeend3
        while o3 ≤ safeend3
            refill!(rb3)
            for _ in 1:safe_n
                o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            end
        end
    elseif o4 ≤ safeend4 # must check because possible to overshoot
        while o4 ≤ safeend4
            refill!(rb4)
            for _ in 1:safe_n
                o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
            end
        end
    end

    while o1 ≤ end1; @inbounds literals[o1] = huffman_decode!(rb1, ht); o1 += 1; end
    while o2 ≤ end2; @inbounds literals[o2] = huffman_decode!(rb2, ht); o2 += 1; end
    while o3 ≤ end3; @inbounds literals[o3] = huffman_decode!(rb3, ht); o3 += 1; end
    while o4 ≤ end4; @inbounds literals[o4] = huffman_decode!(rb4, ht); o4 += 1; end

    return
end

# Read the literals section starting at data[pos].
# Returns (literals::Vector{UInt8}, bytes_consumed::Int).
function read_literals(data::Vector{UInt8}, pos::Int, state::DecompressState)
    b0 = Int(data[pos])
    lit_type = b0 & 0x03
    size_format = (b0 >> 2) & 0x03

    if lit_type == 0 || lit_type == 1   # Raw or RLE
        # Sizes:
        #   size_format == 00 → 5-bit size in bits [7:3] of b0
        #   size_format == 01 → 12-bit size in (b0>>4) | (b1<<4)
        #   size_format == 10 → 20-bit size in (b0>>4) | (b1<<4) | (b2<<12)  (actually 14-bit? no, 20)
        #   size_format == 11 → reserved
        if size_format == 0 || size_format == 2
            regen_size = (b0 >> 3) & 0x1f
            header_size = 1
        elseif size_format == 1
            regen_size = ((b0 >> 4) & 0x0f) | (Int(data[pos+1]) << 4)
            header_size = 2
        else  # size_format == 3
            regen_size = ((b0 >> 4) & 0x0f) | (Int(data[pos+1]) << 4) | (Int(data[pos+2]) << 12)
            header_size = 3
        end

        if lit_type == 0   # Raw
            literals = state.literals_buf
            resize!(literals, regen_size + 15)
            copyto!(literals, 1, data, pos + header_size, regen_size)
            return literals, header_size + regen_size
        else               # RLE
            byte_val = data[pos+header_size]
            literals = state.literals_buf
            resize!(literals, regen_size + 15)
            fill!(view(literals, 1:regen_size), byte_val)
            return literals, header_size + 1
        end
    else   # Compressed (2) or Treeless (3)
        # Header sizes (RFC 8878 §3.1.1.3.2.2):
        #   size_format == 00 → 10-bit sizes, 3 streams NOT used (1 stream), 3-byte header
        #   size_format == 01 → 10-bit sizes, 4-stream, 3-byte header (actually: 10+10 bits = 2.5 bytes → 3 bytes)
        #   size_format == 10 → 14-bit sizes, 4-stream, 4-byte header
        #   size_format == 11 → 18-bit sizes, 4-stream, 5-byte header
        local compressed_size::Int, regen_size::Int, header_size::Int, num_streams::Int
        if size_format == 0
            # 1-stream, header: b0 b1 b2
            # regen_size  = bits [7:2] of b0 combined with b1 [3:0]: 10 bits
            # compressed_size = b1[7:4] | b2: 10 bits
            regen_size      = ((b0 >> 4) & 0x0f) | ((Int(data[pos+1]) & 0x3f) << 4)
            compressed_size = ((Int(data[pos+1]) >> 6) & 0x03) | (Int(data[pos+2]) << 2)
            header_size     = 3
            num_streams     = 1
        elseif size_format == 1
            regen_size      = ((b0 >> 4) & 0x0f) | ((Int(data[pos+1]) & 0x3f) << 4)
            compressed_size = ((Int(data[pos+1]) >> 6) & 0x03) | (Int(data[pos+2]) << 2)
            header_size     = 3
            num_streams     = 4
        elseif size_format == 2
            regen_size      = ((b0 >> 4) & 0x0f) | (Int(data[pos+1]) << 4) | ((Int(data[pos+2]) & 0x03) << 12)
            compressed_size = ((Int(data[pos+2]) >> 2) & 0x3f) | (Int(data[pos+3]) << 6)
            header_size     = 4
            num_streams     = 4
        else  # size_format == 3
            regen_size      = ((b0 >> 4) & 0x0f) | (Int(data[pos+1]) << 4) | ((Int(data[pos+2]) & 0x3f) << 12)
            compressed_size = ((Int(data[pos+2]) >> 6) & 0x03) | (Int(data[pos+3]) << 2) | (Int(data[pos+4]) << 10)
            header_size     = 5
            num_streams     = 4
        end

        payload_start = pos + header_size
        payload_end   = payload_start + compressed_size - 1

        if lit_type == 2   # Compressed: Huffman description included
            ht, hdr_len = read_huffman_description(data, payload_start, state)
            state.huffman = ht
            huf_start = payload_start + hdr_len
        else               # Treeless: reuse previous Huffman table
            state.huffman !== nothing || throw(ArgumentError("zstd: treeless literals but no prior Huffman table"))
            ht = state.huffman
            huf_start = payload_start
        end

        literals = state.literals_buf
        # 16 bytes of slack: 1 for the dual-symbol decode unconditional write,
        # plus 15 for wildcopy16 over-read from literals into out.
        resize!(literals, regen_size + 16)

        if num_streams == 1
            stream_len = payload_end - huf_start + 1
            rb = ReverseBitReader(@view data[huf_start:huf_start+stream_len-1])
            for i in 1:regen_size
                @inbounds literals[i] = UInt8(huffman_decode!(rb, ht))
            end
        else
            # 4 streams: dispatch to _decode_4streams! which specialises on
            # HuffmanTable{L}, making safe_n = 57 ÷ L a compile-time constant.
            _decode_4streams!(data, ht, literals, regen_size, huf_start, payload_end)
        end

        resize!(literals, regen_size + 15)  # keep wildcopy16 over-read slack; trim dual-symbol slack

        return literals, header_size + compressed_size
    end
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
        vstore(vload(Vec{16, UInt8}, src), dst)
        return
    end
    k = 0
    while k + 16 ≤ n
        vstore(vload(Vec{16, UInt8}, src + k), dst + k)
        k += 16
    end
    k < n && vstore(vload(Vec{16, UInt8}, src + n - 16), dst + n - 16)
end

# Execute decoded sequences to produce output bytes.
# Writes starting at wpos in out; returns the next write position.
# When preallocated=true the caller has already resize!'d out to the exact frame size,
# so the total scan and per-block resize! can be skipped entirely.
#
# FUTURE OPTIMISATION — fuse sequence decode and execute:
#
# Currently read_sequences! decodes all sequences into three Int arrays
# (ll_vals, ml_vals, of_vals) and then execute_sequences! replays them.
# This is two passes: the sequence data is written to and read back from
# memory.  Fusing the FSE decode loop directly into the execute loop would
# halve that memory traffic and allow the compiler to interleave FSE state
# updates with literal copy and match copy, improving ILP.
#
# FUTURE OPTIMISATION — wildcopy for short non-overlapping matches:
#
# The non-overlapping match path (offset ≥ ml) calls Base.memcpy (a C FFI
# call) for every match.  For short matches (≤ ~32 bytes), the ccall overhead
# dominates the actual data movement cost.  Replacing short non-overlapping
# match copies with _wildcopy16! (same pattern as literal scatter) would
# eliminate that overhead.  Requires extending the +15 slack on `out` to cover
# over-writes at the match destination, and capping wildcopy to matches that
# cannot overlap (offset ≥ 16 would be sufficient for a 16-byte chunk size).
function execute_sequences!(
        ll_vals::Vector{Int}, ml_vals::Vector{Int}, of_vals::Vector{Int},
        literals::Vector{UInt8}, state::DecompressState,
        out::Vector{UInt8}, wpos::Int, preallocated::Bool)

    n = length(ll_vals)

    if !preallocated
        # Pre-size output: every literal byte + every match byte will be written exactly once.
        # literals has 15 bytes of slack; subtract them so total reflects actual content.
        # Add 15 bytes of slack at the end for _wildcopy16! over-writes.
        total = length(literals) - 15
        @inbounds for i in 1:n
            total += ml_vals[i]
        end
        resize!(out, wpos - 1 + total + 15)
    end

    lit_pos = 1

    @inbounds for i in 1:n
        ll = ll_vals[i]
        ml = ml_vals[i]
        of = of_vals[i]

        # Copy ll literal bytes.  out and literals are distinct arrays so no overlap is possible.
        if ll > 0
            GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), ll)
            wpos    += ll
            lit_pos += ll
        end

        # Determine actual offset from repeat-offset table (RFC 8878 §3.1.1.3.3.5).
        # of is the raw Offset_Value; 1/2/3 are repeat codes, ≥4 is a new offset.
        rep = state.rep
        local offset::Int
        if of > 3
            offset = of - 3
            state.rep = (offset, rep[1], rep[2])
        elseif ll > 0
            # Normal repeat-offset rules
            if of == 1
                offset = rep[1]
                # no rep update
            elseif of == 2
                offset = rep[2]
                state.rep = (rep[2], rep[1], rep[3])
            else  # of == 3
                offset = rep[3]
                state.rep = (rep[3], rep[1], rep[2])
            end
        else
            # LL==0: repeat-offset references shift up by 1
            if of == 1
                offset = rep[2]
                state.rep = (rep[2], rep[1], rep[3])
            elseif of == 2
                offset = rep[3]
                state.rep = (rep[3], rep[1], rep[2])
            else  # of == 3
                offset = rep[1] - 1
                offset > 0 || throw(ArgumentError("zstd: repeat offset - 1 is zero"))
                state.rep = (offset, rep[1], rep[2])
            end
        end

        # Copy match of length ml from offset back in output.
        # The match may reach into the dictionary content prefix.
        # wpos - 1 is the logical end of written output; match_pos is 1-indexed into out.
        dict     = state.dict_content
        dict_len = length(dict)
        match_pos = wpos - offset   # = (wpos - 1) - offset + 1
        if match_pos < 1
            # Offset reaches into dictionary content
            dict_pos = dict_len + match_pos      # 1-indexed into dict
            dict_pos ≥ 1 || throw(ArgumentError("zstd: match offset $offset beyond dictionary and output"))
            for _ in 1:ml
                if dict_pos ≤ dict_len
                    out[wpos] = dict[dict_pos]
                    wpos     += 1
                    dict_pos += 1
                else
                    out[wpos]  = out[match_pos]
                    wpos      += 1
                    match_pos += 1
                end
            end
        else
            if offset ≥ ml
                # Non-overlapping match.  For short copies, _wildcopy16! avoids the
                # libc memcpy FFI call; for larger copies memcpy wins (wider SIMD).
                if ml ≤ 64
                    GC.@preserve out _wildcopy16!(pointer(out, wpos), pointer(out, match_pos), ml)
                else
                    GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), ml)
                end
            elseif offset == 1
                # Single-byte repeat: fill
                @inbounds fill!(view(out, wpos:wpos+ml-1), out[match_pos])
            else
                # Overlapping repeat pattern: copy base pattern once, then
                # keep doubling by copying already-written output.  Each
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

    # Remaining literals after last sequence.
    # Use regen_size (stored in literals length - 15 slack) to get true count.
    rem = length(literals) - 15 - lit_pos + 1
    if rem > 0
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, lit_pos), rem)
        wpos += rem
    end
    return wpos
end

# Read and decode the sequences section.
# data[pos:limit] is the sequences payload.
# Appends decompressed bytes to out.
function read_sequences!(data::Vector{UInt8}, pos::Int, limit::Int,
                         state::DecompressState, literals::Vector{UInt8},
                         out::Vector{UInt8}, wpos::Int, preallocated::Bool)
    # Read sequence count (RFC 8878 §3.1.1.3.3.1)
    b0 = Int(data[pos]);  pos += 1
    local num_seqs::Int
    if b0 < 128
        num_seqs = b0
    elseif b0 < 255
        num_seqs = ((b0 - 128) << 8) | Int(data[pos]);  pos += 1
    else
        num_seqs = Int(data[pos]) + (Int(data[pos+1]) << 8) + 0x7F00;  pos += 2
    end

    if num_seqs == 0
        # No sequences: all literals (literals has 15 bytes of slack)
        lit_len = length(literals) - 15
        if !preallocated
            resize!(out, wpos - 1 + lit_len + 15)
        end
        GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, 1), lit_len)
        return wpos + lit_len
    end

    # Symbol Compression Modes byte (RFC 8878 §3.1.1.3.3.2)
    modes_byte = Int(data[pos]);  pos += 1
    modes_byte & 0x03 == 0 || throw(ArgumentError("zstd: reserved bits set in Symbol_Compression_Modes"))
    ll_mode = (modes_byte >> 6) & 0x03
    of_mode = (modes_byte >> 4) & 0x03
    ml_mode = (modes_byte >> 2) & 0x03

    br = ForwardBitReader(@view data[pos:limit])
    ll_tab = read_fse_table!(br, _LL_TABLE[], state.ll_tab, ll_mode, MAX_LITERALS_LENGTH, 9,
                             state.ll_slot, state)
    of_tab = read_fse_table!(br, _OF_TABLE[], state.of_tab, of_mode, MAX_OFFSET_CODE, 8,
                             state.of_slot, state)
    ml_tab = read_fse_table!(br, _ML_TABLE[], state.ml_tab, ml_mode, MAX_MATCH_LENGTH, 9,
                             state.ml_slot, state)
    state.ll_tab = ll_tab
    state.of_tab = of_tab
    state.ml_tab = ml_tab

    # The bitstream for sequences is a reverse bitstream starting right after
    # the FSE table descriptions.
    seq_start = byte_pos(br)
    seq_len   = limit - seq_start + 1
    seq_len > 0 || throw(ArgumentError("zstd: no data for sequences bitstream"))

    rb = ReverseBitReader(@view data[seq_start:seq_start+seq_len-1])

    # Init FSE states: order is LL, OF, ML (RFC 8878 §3.1.1.3.3.4)
    ll_state = fse_init!(rb, ll_tab)
    of_state = fse_init!(rb, of_tab)
    ml_state = fse_init!(rb, ml_tab)

    resize!(state.ll_vals, num_seqs)
    resize!(state.ml_vals, num_seqs)
    resize!(state.of_vals, num_seqs)
    ll_vals = state.ll_vals
    ml_vals = state.ml_vals
    of_vals = state.of_vals

    for i in 1:num_seqs
        # 1. Peek symbols from all three states (no bits consumed)
        ll_code = fse_peek(ll_tab, ll_state)
        ml_code = fse_peek(ml_tab, ml_state)
        of_code = fse_peek(of_tab, of_state)
        of_code ≤ MAX_OFFSET_CODE ||
            throw(ArgumentError("zstd: offset code $of_code exceeds maximum supported value, $MAX_OFFSET_CODE"))

        # 2+3 batched: precompute all six bit widths, then extract all from
        # the same frozen bit word.  This breaks the 6-deep serial dependency
        # chain (each read_bits! currently gates the next via rb.bits).
        #
        # All widths are known before any bit is consumed, so the CPU can
        # issue the six extractions and the cumulative-offset additions in
        # parallel once the single refill is done.
        of_n  = of_code
        ml_n  = Int(@inbounds MATCH_LENGTH_EXTRA_BITS[ml_code + 1])
        ll_n  = Int(@inbounds LITERALS_LENGTH_EXTRA_BITS[ll_code + 1])

        # State-transition widths (skip on last sequence — states not used again)
        update = i < num_seqs
        ll_nb = update ? _fse_nb_bits(ll_tab, ll_state) : 0
        ml_nb = update ? _fse_nb_bits(ml_tab, ml_state) : 0
        of_nb = update ? _fse_nb_bits(of_tab, of_state) : 0

        total_n = of_n + ml_n + ll_n + ll_nb + ml_nb + of_nb

        if total_n ≤ 57
            # Fast path: a single refill guarantees ≥ 57 bits available.
            rb.nbits < total_n && refill!(rb)
            rb.nbits ≥ total_n || throw(ArgumentError("zstd: unexpected end of sequence bitstream"))

            # Cumulative bit offsets into the frozen snapshot
            c_ml  = of_n
            c_ll  = c_ml + ml_n
            c_llb = c_ll + ll_n
            c_mlb = c_llb + ll_nb
            c_ofb = c_mlb + ml_nb

            # All six extractions read from `bits` independently — ILP-friendly.
            bits     = rb.bits
            of_extra = (of_n  > 0) ? Int(bits              >>> (64 - of_n )) : 0
            ml_extra = (ml_n  > 0) ? Int((bits << c_ml)    >>> (64 - ml_n )) : 0
            ll_extra = (ll_n  > 0) ? Int((bits << c_ll)    >>> (64 - ll_n )) : 0
            ll_bits  = (ll_nb > 0) ? Int((bits << c_llb)   >>> (64 - ll_nb)) : 0
            ml_bits  = (ml_nb > 0) ? Int((bits << c_mlb)   >>> (64 - ml_nb)) : 0
            of_bits  = (of_nb > 0) ? Int((bits << c_ofb)   >>> (64 - of_nb)) : 0

            # Single bulk consume
            rb.bits   = bits << total_n
            rb.nbits -= total_n

        else
            # Slow path: large offset (of_code > ~20); sequential reads.
            of_extra   = (of_n  > 0) ? Int(read_bits!(rb, of_n )) : 0
            ml_extra   = Int(read_bits!(rb, ml_n))
            ll_extra   = Int(read_bits!(rb, ll_n))
            ll_bits    = (ll_nb > 0) ? Int(read_bits!(rb, ll_nb)) : 0
            ml_bits    = (ml_nb > 0) ? Int(read_bits!(rb, ml_nb)) : 0
            of_bits    = (of_nb > 0) ? Int(read_bits!(rb, of_nb)) : 0
        end

        of_val64 = (Int64(1) << of_code) + of_extra
        of_val64 ≤ typemax(Int) || throw(ArgumentError("zstd: offset value $of_val64 exceeds addressable range"))
        of_vals[i] = Int(of_val64)
        ml_vals[i] = Int(@inbounds MATCH_LENGTH_BASELINE[ml_code + 1]) + ml_extra
        ll_vals[i] = Int(@inbounds LITERALS_LENGTH_BASELINE[ll_code + 1]) + ll_extra

        if update
            ll_state = _fse_baseline(ll_tab, ll_state) + ll_bits
            ml_state = _fse_baseline(ml_tab, ml_state) + ml_bits
            of_state = _fse_baseline(of_tab, of_state) + of_bits
        end
    end

    return execute_sequences!(ll_vals, ml_vals, of_vals, literals, state, out, wpos, preallocated)
end

# ============================================================
# Section 9: Block decompression
#   Reference: RFC 8878 §3.1.1
# ============================================================

function decompress_block!(data::Vector{UInt8}, pos::Int, block_size::Int,
                           block_type::Int, state::DecompressState,
                           out::Vector{UInt8}, wpos::Int, preallocated::Bool)
    if block_type == 0   # Raw block
        pos + block_size - 1 ≤ length(data) ||
            throw(ArgumentError("zstd: truncated raw block"))
        if !preallocated
            resize!(out, wpos - 1 + block_size)
        end
        GC.@preserve out data Base.memcpy(pointer(out, wpos), pointer(data, pos), block_size)
        return wpos + block_size
    elseif block_type == 1   # RLE block
        # block_size == regen_size; compressed payload is always 1 byte
        if !preallocated
            resize!(out, wpos - 1 + block_size)
        end
        fill!(view(out, wpos:wpos+block_size-1), data[pos])
        return wpos + block_size
    elseif block_type == 2   # Compressed block
        limit = pos + block_size - 1
        literals, lit_consumed = read_literals(data, pos, state)
        seq_pos = pos + lit_consumed
        if seq_pos ≤ limit
            return read_sequences!(data, seq_pos, limit, state, literals, out, wpos, preallocated)
        else
            # No sequences section: all literals (literals has 15 bytes of slack)
            lit_len = length(literals) - 15
            if !preallocated
                resize!(out, wpos - 1 + lit_len + 15)
            end
            GC.@preserve out literals _wildcopy16!(pointer(out, wpos), pointer(literals, 1), lit_len)
            return wpos + lit_len
        end
    else
        throw(ArgumentError("zstd: reserved block type 3"))
    end
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

function _decompress_frame!(data::Vector{UInt8}, pos::Int, out::Vector{UInt8},
                            dict::Union{ZstdDict, Nothing} = nothing)
    # Magic number (4 bytes, little-endian)
    magic = _read_magic(data, pos)
    magic == ZSTD_MAGIC ||
        throw(ArgumentError("zstd: invalid magic number 0x$(string(magic, base = 16))"))
    pos += 4

    # Frame Header Descriptor (FHD)
    window_size, frame_content_size, content_checksum_flag, pos = _read_frame_header(data, pos, dict)

    # Enforce a maximum window size to prevent memory exhaustion (2 GiB); spec maximum is (1 << 41) + 7 * (1 << 38)
    window_size ≤ Int64(1) << 31 ||
        throw(ArgumentError("zstd: window size $window_size exceeds maximum supported (2 GiB)"))

    state = dict !== nothing ? DecompressState(dict) : DecompressState()
    frame_start = length(out)
    # When FCS is known, resize to the exact frame size upfront so that all
    # per-block writes go directly into pre-allocated space — no per-block
    # resize! or ml_vals total scan needed.  When FCS is unknown, rely on the
    # caller's one-time sizehint! and fall back to per-block resize!.
    preallocated = frame_content_size ≥ 0
    if preallocated
        resize!(out, frame_start + frame_content_size + 15)
    end
    wpos = frame_start + 1

    # Decode blocks
    while true
        length(data) ≥ pos + 2 ||
            throw(ArgumentError("zstd: truncated block header"))
        # Block header is 3 bytes
        bh1, bh2, bh3 = Int(data[pos]), Int(data[pos+1]), Int(data[pos+2])
        pos += 3

        last_block  = bh1 & 0x01
        block_type  = (bh1 >> 1) & 0x03
        # block_size for compressed/raw/RLE: 21-bit field in remaining 21 bits
        block_size  = (bh1 >> 3) | (bh2 << 5) | (bh3 << 13)
        block_size ≤ ZSTD_BLOCKSIZE_MAX ||
            throw(ArgumentError("zstd: block size $block_size exceeds maximum (128 KB)"))

        if block_type == 1   # RLE: 1 compressed byte; block_size = regen size
            wpos = decompress_block!(data, pos, block_size, block_type, state, out, wpos, preallocated)
            pos += 1
        else
            wpos = decompress_block!(data, pos, block_size, block_type, state, out, wpos, preallocated)
            pos += block_size
        end

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

If the frame was compressed with a dictionary, pass it as `dict` — either
a `ZstdDict` returned by [`parse_dictionary`](@ref), or the raw dictionary
bytes (`Vector{UInt8}`).
"""
function inflate_zstd(data::Vector{UInt8}; dict::Union{ZstdDict,Vector{UInt8},Nothing}=nothing, nthreads::Int=Threads.nthreads())
    nthreads ≥ 1 || throw(ArgumentError("zstd: nthreads must be ≥ 1, got $nthreads"))
    isempty(data) && throw(ArgumentError("zstd: empty input"))
    d = dict isa Vector{UInt8} ? parse_dictionary(dict) : dict

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
function inflate_zstd(filename::AbstractString; dict::Union{ZstdDict,Vector{UInt8},Nothing}=nothing, nthreads::Int=Threads.nthreads())
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

If the data was compressed with a dictionary, pass it as `dict` — either
a `ZstdDict` or raw dictionary bytes (`Vector{UInt8}`).
"""
mutable struct InflateZstdStream <: IO
    buf::Vector{UInt8}
    pos::Int
end

function InflateZstdStream(io::IO; dict::Union{ZstdDict,Vector{UInt8},Nothing}=nothing, nthreads::Int=Threads.nthreads())
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
