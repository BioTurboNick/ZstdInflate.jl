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

export inflate_zstd, InflateZstdStream, ZstdDict, parse_dictionary

include("constants.jl")

# Wrapping shift helpers: emit a single shift instruction on x86 without
# Julia's default guards for shift ≥ 64.  Safe only when the caller knows
# n ∈ [0, 63] (the hardware masks the count anyway; `& 63` just proves it
# to LLVM so it drops the overflow branches).
@inline _shl(x::UInt64, n::Int) = x << (n & 63)
@inline _shr(x::UInt64, n::Int) = x >>> (n & 63)


# ============================================================
# Section 2: xxHash-64  (RFC 8878 §3.1.5 — content checksum)
#   Reference algorithm: https://cyan4973.github.io/xxHash/
# ============================================================

const XXH_P1 = 0x9E3779B185EBCA87
const XXH_P2 = 0xC2B2AE3D27D4EB4F
const XXH_P3 = 0x165667B19E3779F9
const XXH_P4 = 0x85EBCA77C2B2AE63
const XXH_P5 = 0x27D4EB2F165667C5

@inline xxh_rotl64(x::Union{UInt64, Vec{4, UInt64}}, r) = (x << r) | (x >>> (64 - r))

@inline function xxh_round(acc::T, lane::T) where T <: Union{UInt64, Vec{4, UInt64}}
    acc += lane * XXH_P2
    acc  = xxh_rotl64(acc, 31)
    acc *= XXH_P1
    return acc
end

@inline function xxh_merge(acc::T, val::T) where T <: Union{UInt64, Vec{4, UInt64}}
    acc ⊻= xxh_round(T(0), val)
    acc  = acc * XXH_P1 + XXH_P4
    return acc
end

# Little-endian loads
@inline _le64(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt64}(pointer(d, i))))
@inline _le32(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt32}(pointer(d, i))))
@inline _le16(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt16}(pointer(d, i))))

function xxhash64(data::AbstractVector{UInt8}, seed::UInt64 = UInt64(0))
    len = length(data)
    pos = 1
    local h64::UInt64

    if len ≥ 32
        rdata64 = reinterpret(UInt64, @view data[1:end - (end % 8)])
        v1 = seed + XXH_P1 + XXH_P2
        v2 = seed + XXH_P2
        v3 = seed
        v4 = seed - XXH_P1
        while 8(pos - 1) + 32 ≤ len
            vs = Vec{4, UInt64}((v1, v2, v3, v4))
            vdata = vload(Vec{4, UInt64}, (@view rdata64[pos:pos + 3]), 1)
            vs = xxh_round(vs, ltoh(vdata))
            v1, v2, v3, v4 = NTuple{4, UInt64}(vs)
            pos += 4
        end
        h64  = xxh_rotl64(v1, 1) + xxh_rotl64(v2, 7) + xxh_rotl64(v3, 12) + xxh_rotl64(v4, 18)
        h64  = xxh_merge(h64, v1)
        h64  = xxh_merge(h64, v2)
        h64  = xxh_merge(h64, v3)
        h64  = xxh_merge(h64, v4)
    else
        h64 = seed + XXH_P5
    end

    h64 += UInt64(len)

    while 8(pos - 1) + 8 ≤ len
        rdata64 = reinterpret(UInt64, @view data[1:end - (end % 8)])
        h64 ⊻= xxh_round(UInt64(0), ltoh(rdata64[pos])); pos += 1
        h64  = xxh_rotl64(h64, 27) * XXH_P1 + XXH_P4
    end
    pos = 2 * (pos - 1) + 1
    if 4(pos - 1) + 4 ≤ len
        rdata32 = reinterpret(UInt32, @view data[1:end - (end % 4)])
        h64 ⊻= UInt64(ltoh(rdata32[pos])) * XXH_P1; pos += 1
        h64  = xxh_rotl64(h64, 23) * XXH_P2 + XXH_P3
    end
    pos = 4 * (pos - 1) + 1
    while pos ≤ len
        h64 ⊻= UInt64(data[pos]) * XXH_P5
        h64  = xxh_rotl64(h64, 11) * XXH_P1; pos += 1
    end

    h64 ⊻= h64 >>> 33;  h64 *= XXH_P2
    h64 ⊻= h64 >>> 29;  h64 *= XXH_P3
    h64 ⊻= h64 >>> 32
    return h64
end

# ============================================================
# Section 3a: Forward bit reader  (LSB-first)
#   Used for: frame/block headers, literals headers, FSE table
#   descriptions, and sequences section header.
# ============================================================

mutable struct ForwardBitReader
    data::Vector{UInt8}
    pos::Int       # next byte to load into the buffer (1-indexed)
    limit::Int     # one past the last valid byte (exclusive, 1-indexed)
    bits::UInt64   # bit buffer; bit 0 is the next bit to deliver
    nbits::Int     # number of valid bits currently in buffer
end

function ForwardBitReader(data::Vector{UInt8}, byte_offset::Int, byte_limit::Int)
    ForwardBitReader(data, byte_offset, byte_limit, UInt64(0), 0)
end

function _fbr_refill!(br::ForwardBitReader)
    while br.nbits ≤ 56 && br.pos < br.limit
        br.bits  |= UInt64(br.data[br.pos]) << br.nbits
        br.nbits += 8
        br.pos   += 1
    end
end

@inline function read_bits!(br::ForwardBitReader, n::Int)
    n == 0 && return UInt64(0)
    br.nbits < n && _fbr_refill!(br)
    br.nbits ≥ n || throw(ArgumentError("zstd: unexpected end of bitstream (need $n bits, have $(br.nbits))"))
    mask = (n < 64) ? (_shl(UInt64(1), n) - UInt64(1)) : typemax(UInt64)
    val      = br.bits & mask
    br.bits  = (n < 64) ? _shr(br.bits, n) : UInt64(0)
    br.nbits -= n
    return val
end

# Discard any sub-byte leftover so the reader is byte-aligned.
function align_to_byte!(br::ForwardBitReader)
    discard  = br.nbits & 7
    discard > 0 && (br.bits >>>= discard)
    br.nbits -= discard
end

# Byte position of the next unread byte (after draining the bit buffer).
byte_pos(br::ForwardBitReader) = br.pos - (br.nbits >> 3)

# ============================================================
# Section 3b: Reverse bit reader  (MSB-first, reads backward)
#   Used for: FSE sequences bitstream, Huffman literal streams.
#   The stream ends with a sentinel byte whose highest set bit
#   marks the start of valid data.
# ============================================================

mutable struct ReverseBitReader
    data::Vector{UInt8}
    start::Int     # first valid byte of this slice (1-indexed)
    pos::Int       # next byte to load (decreasing, 1-indexed)
    bits::UInt64   # bit buffer; bit 63 is the next bit to deliver
    nbits::Int     # number of valid bits currently in buffer
end

function ReverseBitReader(data::Vector{UInt8}, byte_offset::Int, byte_len::Int)
    byte_len  > 0 || throw(ArgumentError("zstd: empty reverse bitstream"))
    last       = data[byte_offset + byte_len - 1]
    last != 0  || throw(ArgumentError("zstd: reverse bitstream sentinel byte is zero"))

    # Position of the sentinel bit (0 = LSB, 7 = MSB)
    sentinel   = 63 - leading_zeros(UInt64(last))   # highest set bit of last byte

    # Valid data bits sit below the sentinel: bits [0, sentinel-1]
    valid_n    = sentinel                            # may be 0 if last == 1
    valid_bits = UInt64(last) & ((valid_n > 0) ? ((UInt64(1) << valid_n) - UInt64(1)) : UInt64(0))

    # Pack into the MSB region of the 64-bit container
    bits  = (valid_n > 0) ? (valid_bits << (64 - valid_n)) : UInt64(0)
    nbits = valid_n

    rb = ReverseBitReader(data, byte_offset, byte_offset + byte_len - 2, bits, nbits)
    _rbr_refill!(rb)
    return rb
end

@inline function _load64_le(data::Vector{UInt8}, i::Int)
    # Load 8 bytes from data[i:i+7] as a little-endian UInt64.
    # On little-endian x86, data[i+7] (highest address) lands in the MSB,
    # which is exactly right for a reverse bit reader where data[pos] (the
    # highest-addressed unread byte) should be the most significant.
    unsafe_load(Ptr{UInt64}(pointer(data, i)))
end

function _rbr_refill!(rb::ReverseBitReader)
    n0 = rb.nbits
    n0 > 56 && return
    pos   = rb.pos
    avail = pos - rb.start + 1
    avail ≤ 0 && return

    k = min(((57 - n0) + 7) >> 3, avail)

    if pos ≥ 8
        # Fast path: single 8-byte load.  data[pos-7..pos] is read as big-endian
        # so data[pos] (the next byte to consume) lands in the MSB.  Shifting
        # right by n0 positions the new bytes just below the existing valid bits.
        # We mask off extra loaded bytes beyond k so they don't contaminate the
        # lower bit region (which must stay zero for the next refill's OR).
        raw = _load64_le(rb.data, pos - 7)
        # Zero out everything below the k bytes we need: keep top 8k bits of raw
        cleaned = raw & _shl(typemax(UInt64), 64 - 8k)
        rb.bits |= _shr(cleaned, n0)
    else
        # Near start of data (pos < 8): byte-by-byte fallback.
        # Happens at most once per stream so performance is not critical.
        s = 56 - n0
        a = rb.bits
        b = UInt64(0)
        @inbounds begin
            k ≥ 1 && (a |= _shl(UInt64(rb.data[pos    ]), s      ))
            k ≥ 2 && (b |= _shl(UInt64(rb.data[pos - 1]), s -  8 ))
            k ≥ 3 && (a |= _shl(UInt64(rb.data[pos - 2]), s - 16 ))
            k ≥ 4 && (b |= _shl(UInt64(rb.data[pos - 3]), s - 24 ))
            k ≥ 5 && (a |= _shl(UInt64(rb.data[pos - 4]), s - 32 ))
            k ≥ 6 && (b |= _shl(UInt64(rb.data[pos - 5]), s - 40 ))
            k ≥ 7 && (a |= _shl(UInt64(rb.data[pos - 6]), s - 48 ))
        end
        rb.bits = a | b
    end

    rb.nbits = n0 + 8k
    rb.pos   = pos - k

    # FUTURE OPTIMISATION — "consumed counter" model (zstd reference style):
    #
    # Switching to a "bits_consumed" counter would make refills O(1):
    #
    #   ptr          -= bits_consumed >> 3   # skip fully-consumed bytes
    #   bitContainer  = load64(ptr)
    #   bits_consumed &= 7                   # keep the partial-byte remainder
    #
    # This would give 60–64 valid bits per refill instead of 57, increasing
    # safe_n and reducing refill frequency.  The change requires updating
    # ReverseBitReader (add bits_consumed field, remove nbits), and adjusting
    # every consumer to use `(rb.bits << rb.bits_consumed) >>> (64 - n)`.
end

@inline function read_bits_r!(rb::ReverseBitReader, n::Int)
    n == 0 && return UInt64(0)
    rb.nbits < n && _rbr_refill!(rb)
    rb.nbits ≥ n || throw(ArgumentError("zstd: unexpected end of reverse bitstream"))
    val      = _shr(rb.bits, 64 - n)
    rb.bits  = (n < 64) ? _shl(rb.bits, n) : UInt64(0)
    rb.nbits -= n
    return val
end

# Peek without consuming.
@inline function peek_bits_r!(rb::ReverseBitReader, n::Int)
    rb.nbits < n && _rbr_refill!(rb)
    rb.nbits ≥ n || throw(ArgumentError("zstd: unexpected end of reverse bitstream (peek)"))
    return _shr(rb.bits, 64 - n)
end

@inline function consume_bits_r!(rb::ReverseBitReader, n::Int)
    # RFC 8878: Huffman code lengths are bounded by HUFTABLE_LOG_MAX (11),
    # so n is always in [1, 11] — never ≥ 64.  Catch bugs if that ever changes.
    n < 64 || throw(ArgumentError("zstd: consume_bits_r! shift $n ≥ 64"))
    rb.bits  = _shl(rb.bits, n)
    rb.nbits -= n
end

# Read n bits without checking for underflow (allows nbits to go negative).
# Used by the interleaved FSE weight decoder tail loop where overflow is
# detected after the read rather than before.
@inline function _read_bits_r_unchecked!(rb::ReverseBitReader, n::Int)
    n == 0 && return UInt64(0)
    _rbr_refill!(rb)
    val      = _shr(rb.bits, 64 - n)
    rb.bits  = (n < 64) ? _shl(rb.bits, n) : UInt64(0)
    rb.nbits -= n
    return val
end

# Check whether previous unchecked reads consumed more bits than available.
@inline _rbr_overflowed(rb::ReverseBitReader) = rb.nbits < 0

# ============================================================
# Section 4: FSE decode table
#   Reference: RFC 8878 §4.1
# ============================================================

struct FSETable
    accuracy_log::Int
    symbols  ::Vector{UInt8}
    nb_bits  ::Vector{UInt8}
    baselines::Vector{UInt32}
end

# Trivial FSE table for RLE mode: every state emits the same symbol with
# zero extra bits.  Stored as a single byte — no heap allocation.
struct RLEFSETable
    symbol::UInt8
end

# floor(log2(n)) for n ≥ 1
@inline _flog2(n::Int) = 63 - leading_zeros(UInt64(n))

# Build an FSE decode table from a normalized probability distribution.
# norm[i+1] is the probability of symbol i; -1 means "probability 1/tableSize".
# Hot-path: caller supplies pre-allocated backing arrays; they are resized and filled in-place.
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int,
                          syms::Vector{UInt8}, nb::Vector{UInt8},
                          base::Vector{UInt32}, occ::Vector{Int})
    table_size = 1 << accuracy_log
    resize!(syms, table_size)
    resize!(nb,   table_size)
    resize!(base, table_size)
    resize!(occ,  length(norm))
    fill!(occ, 0)

    # --- Spread: place -1 symbols at the end, others via step pattern ---
    high = table_size - 1
    for s in 0:length(norm)-1
        norm[s+1] == -1 || continue
        syms[high+1] = UInt8(s)
        high -= 1
    end

    step = (table_size >> 1) + (table_size >> 3) + 3
    mask = table_size - 1
    pos  = 0
    for s in 0:length(norm)-1
        c = Int(norm[s+1])
        c > 0 || continue
        for _ in 1:c
            syms[pos+1] = UInt8(s)
            pos = (pos + step) & mask
            while pos > high
                pos = (pos + step) & mask
            end
        end
    end

    # --- Build per-state decode entries ---
    for x in 0:table_size-1
        s  = Int(syms[x+1])
        c  = Int(norm[s+1]);  c == -1 && (c = 1)
        i  = occ[s+1];        occ[s+1] += 1
        ci = c + i
        n  = accuracy_log - _flog2(ci)
        nb[x+1]   = UInt8(n)
        base[x+1] = UInt32((ci << n) - table_size)
    end

    return FSETable(accuracy_log, syms, nb, base)
end

# Cold-path: allocates its own backing arrays (used by __init__, parse_dictionary, etc.)
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int)
    table_size = 1 << accuracy_log
    return build_fse_table(norm, accuracy_log,
                           Vector{UInt8}(undef, table_size),
                           Vector{UInt8}(undef, table_size),
                           Vector{UInt32}(undef, table_size),
                           Vector{Int}(undef, length(norm)))
end

# Read an FSE normalized distribution from the forward bitstream.
# Returns (accuracy_log, norm_counts).
# Implements the exact reference zstd sliding-threshold algorithm.
# Hot-path: caller supplies a reusable norm buffer (emptied on entry).
function read_fse_dist!(br::ForwardBitReader, max_sym::Int, norm::Vector{Int16})
    accuracy_log = Int(read_bits!(br, 4)) + 5
    table_size   = 1 << accuracy_log

    empty!(norm)
    remaining = table_size + 1   # reference zstd initialises to tableSize+1
    threshold = table_size
    nbits     = accuracy_log + 1

    while remaining > 1
        br.nbits < nbits && _fbr_refill!(br)

        # max = number of values encodable in the short (nbits-1 bit) path
        max_val = (2 * threshold - 1) - remaining
        low     = Int(br.bits & UInt64(threshold - 1))   # peek lower nbits-1 bits

        local count::Int
        if low < max_val
            # Short path: value from nbits-1 bits
            count = low
            br.bits  >>>= (nbits - 1)
            br.nbits  -= (nbits - 1)
        else
            # Long path: value from nbits bits
            count = Int(br.bits & UInt64(2 * threshold - 1))
            if count ≥ threshold
                count -= max_val
            end
            br.bits  >>>= nbits
            br.nbits  -= nbits
        end

        count -= 1   # "extra accuracy" offset: value 0 → count -1 (low-prob)

        push!(norm, Int16(count))

        if count == 0
            # Zero-run: chained 2-bit repeat count (RFC 8878 §4.1.1)
            while true
                br.nbits < 2 && _fbr_refill!(br)
                r = Int(br.bits & 3)
                br.bits  >>>= 2
                br.nbits  -= 2
                for _ in 1:r
                    push!(norm, Int16(0))
                end
                r < 3 && break
            end
        else
            remaining -= count < 0 ? -count : count
        end

        # Shrink threshold/nbits as remaining decreases.
        while remaining < threshold
            nbits     -= 1
            threshold >>= 1
        end
    end

    align_to_byte!(br)
    return accuracy_log, norm
end

# Cold-path: allocates its own norm buffer (used by parse_dictionary, _decode_fse_weights, etc.)
function read_fse_dist!(br::ForwardBitReader, max_sym::Int)
    norm = Int16[]
    sizehint!(norm, max_sym + 1)
    return read_fse_dist!(br, max_sym, norm)
end

# ------- Predefined tables (built once at module init) -------

const _LL_TABLE = Ref{FSETable}()
const _ML_TABLE = Ref{FSETable}()
const _OF_TABLE = Ref{FSETable}()

function __init__()
    _LL_TABLE[] = build_fse_table(LITERALS_LENGTH_DEFAULT_DIST, LITERALS_LENGTH_ACCURACY_LOG)
    _ML_TABLE[] = build_fse_table(MATCH_LENGTH_DEFAULT_DIST, MATCH_LENGTH_ACCURACY_LOG)
    _OF_TABLE[] = build_fse_table(OFFSET_DEFAULT_DIST, OFFSET_ACCURACY_LOG)
end

# ------- FSE state machine helpers -------

@inline fse_init!(rb::ReverseBitReader, t::FSETable) =
    Int(read_bits_r!(rb, t.accuracy_log))

@inline fse_peek(t::FSETable, state::Int) = Int(t.symbols[state+1])

@inline function fse_update!(rb::ReverseBitReader, t::FSETable, state::Int)
    nb   = Int(t.nb_bits[state+1])
    bits = Int(read_bits_r!(rb, nb))
    return Int(t.baselines[state+1]) + bits
end

# RLE variants — accuracy_log is always 1, both states emit the same symbol,
# and transitions consume 0 bits.
@inline fse_init!(rb::ReverseBitReader, t::RLEFSETable) =
    (read_bits_r!(rb, 1); 0)   # consume the 1-bit state init, state is always 0

@inline fse_peek(t::RLEFSETable, state::Int) = Int(t.symbol)

@inline fse_update!(rb::ReverseBitReader, t::RLEFSETable, state::Int) = 0

# Helpers for the batched sequence reader: retrieve the transition width and
# baseline for the current state without consuming any bits.
@inline _fse_nb_bits(t::FSETable,    state::Int) = Int(@inbounds t.nb_bits[state + 1])
@inline _fse_nb_bits(t::RLEFSETable, state::Int) = 0

@inline _fse_baseline(t::FSETable,    state::Int) = Int(@inbounds t.baselines[state + 1])
@inline _fse_baseline(t::RLEFSETable, state::Int) = 0

# Update without checking for underflow (allows overflow detection after).
@inline function _fse_update_unchecked(rb::ReverseBitReader, t::FSETable, state::Int)
    nb   = Int(t.nb_bits[state+1])
    bits = Int(_read_bits_r_unchecked!(rb, nb))
    return Int(t.baselines[state+1]) + bits
end

# Read an FSE table according to the given mode byte bits.
# mode: 0=predefined, 1=RLE, 2=FSE_Compressed, 3=Repeat
# Returns the FSETable and advances br.
function read_fse_table!(br::ForwardBitReader, default::FSETable,
                         prev::Union{FSETable,RLEFSETable,Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         syms::Vector{UInt8}, nb::Vector{UInt8},
                         base::Vector{UInt32}, occ::Vector{Int},
                         norm::Vector{Int16})
    if mode == 0
        return default
    elseif mode == 1
        sym = read_bits!(br, 8)   # RLE: single symbol
        return RLEFSETable(UInt8(sym))
    elseif mode == 2
        al, dist = read_fse_dist!(br, max_sym, norm)
        al ≤ max_al || throw(ArgumentError("zstd: accuracy log $al exceeds maximum $max_al"))
        length(dist) ≤ max_sym + 1 || throw(ArgumentError("zstd: FSE distribution has $(length(dist)) symbols, maximum is $(max_sym + 1)"))
        return build_fse_table(dist, al, syms, nb, base, occ)
    else   # mode == 3: repeat previous table (RFC 8878 §3.1.1.3.3.2)
        prev !== nothing || throw(ArgumentError("zstd: repeat mode but no previous FSE table"))
        return prev
    end
end

# ============================================================
# Section 5: Huffman decode table
#   Reference: RFC 8878 §4.2
# ============================================================

struct HuffmanTable{L}
    # Packed dual-symbol decode table indexed by the top L bits of the peeked code word.
    # Each UInt32 entry encodes up to two symbols:
    #   bits [31:24]  nb_total — total bits consumed (nb1 + nb2; ≤ L)
    #   bits [23:16]  nb1      — bits consumed by sym1 alone (needed for single-symbol path)
    #   bits [15: 8]  sym2     — second symbol (valid only when nb_total > nb1)
    #   bits [ 7: 0]  sym1     — first symbol (always valid)
    # nb_total == nb1 indicates a single-symbol entry; nb_total > nb1 is a two-symbol entry.
    decode_table::Vector{UInt32}
end

# Build a dual-symbol Huffman decode table from a weight array.
# weights[i+1] = weight for symbol i (0 = absent; weight w ≥ 1 means code length
# max_bits - w + 1, probability 2^(w-1)).
# Each entry is a UInt32: (nb_total<<24)|(nb1<<16)|(sym2<<8)|sym1.
function build_huffman_table(weights::Vector{UInt8}, max_bits::Int)
    nsyms = length(weights)
    max_bits > 0 ||
        throw(ArgumentError("zstd: all-zero Huffman weights"))
    max_bits ≤ HUFTABLE_LOG_MAX ||
        throw(ArgumentError("zstd: Huffman table log $max_bits exceeds maximum ($HUFTABLE_LOG_MAX)"))

    table_size = 1 << max_bits
    # Pass 1: build a temporary single-symbol table (sym, nb1) to use as input
    # for the dual-symbol pass below.
    tmp = fill(UInt16(max_bits), table_size)  # (sym<<8)|nb1

    rank_count      = zeros(Int, max_bits + 1)
    next_rank_start = zeros(Int, max_bits + 1)
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
        entry       = UInt16((sym << 8) | nb1)
        start       = next_rank_start[w]
        for j in 0:num_entries-1
            tmp[start + j + 1] = entry
        end
        next_rank_start[w] += num_entries
    end

    # Pass 2: build the full max_bits-wide dual-symbol decode table.
    dtable = Vector{UInt32}(undef, table_size)
    for idx in 0:table_size-1
        e1   = Int(tmp[idx + 1])
        sym1 = (e1 >> 8) & 0xFF
        nb1  = e1 & 0xFF
        rem  = max_bits - nb1   # bits remaining after sym1
        if rem > 0
            idx2 = (idx << nb1) & (table_size - 1)
            e2   = Int(tmp[idx2 + 1])
            sym2 = (e2 >> 8) & 0xFF
            nb2  = e2 & 0xFF
            if nb2 ≤ rem
                dtable[idx + 1] = UInt32((nb1 + nb2) << 24) | UInt32(nb1 << 16) |
                                  UInt32(sym2 << 8) | UInt32(sym1)
                continue
            end
        end
        dtable[idx + 1] = UInt32(nb1 << 24) | UInt32(nb1 << 16) | UInt32(sym1)
    end

    return HuffmanTable{max_bits}(dtable)
end

# Decode one Huffman symbol — caller must ensure nbits ≥ max_bits (no refill).
@inline function _huffman_decode_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    idx      = _shr(rb.bits, 64 - L) % Int
    e        = @inbounds ht.decode_table[idx + 1]
    nb1      = Int((e >> 16) & 0xFF)
    rb.bits  = _shl(rb.bits, nb1)
    rb.nbits -= nb1
    return Int(e & 0xFF)
end

# Decode one Huffman symbol from the reverse bit reader.
@inline function huffman_decode!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    rb.nbits < L && _rbr_refill!(rb)
    _huffman_decode_nocheck!(rb, ht)
end

# Decode up to 2 Huffman symbols — caller must ensure nbits ≥ max_bits (no refill).
# Writes sym1 and sym2 to out[pos] and out[pos+1] unconditionally (caller must
# ensure out has one byte of slack past regen_size for the last-slot case).
# Returns the number of symbols decoded (1 or 2).
@inline function _huffman_decode2_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L},
                                           out::Vector{UInt8}, pos::Int) where L
    idx      = _shr(rb.bits, 64 - L) % Int
    e        = @inbounds ht.decode_table[idx + 1]
    nb_total = Int(e >> 24)
    nb1      = Int((e >> 16) & 0xFF)
    @inbounds out[pos]   = UInt8(e & 0xFF)
    @inbounds out[pos+1] = UInt8((e >> 8) & 0xFF)
    rb.bits  = _shl(rb.bits, nb_total)
    rb.nbits  -= nb_total
    return nb_total > nb1 ? 2 : 1
end


# ============================================================
# Section 6: Huffman tree description
#   Reference: RFC 8878 §4.2.1
# ============================================================

# Infer the weight of the last symbol given the weight array so far.
# Returns the (last_sym, last_weight) pair, where last_sym is the index
# of the symbol whose weight we must fill in.
function _infer_last_weight(weights::Vector{UInt8})
    # RFC 8878 §4.2.1: sum of 2^(w-1) for all present symbols must be a
    # power of two (call it P). The last (highest) symbol gets w = log2(P - sum) + 1.
    # Returns (last_sym_index, last_weight, table_log).
    total = UInt64(0)
    for w in weights
        w == 0 && continue
        total += UInt64(1) << (Int(w) - 1)
    end
    total == 0 && return (length(weights), 1, 1)  # single symbol edge case
    # tableLog = floor(log2(total)) + 1  (reference: BIT_highbit32(weightTotal) + 1)
    table_log = _flog2(Int(total)) + 1
    p = UInt64(1) << table_log
    p > total || (table_log += 1; p <<= 1)   # ensure p > total
    rest = Int(p - total)
    rest > 0 && (rest & (rest - 1)) == 0 || throw(ArgumentError("zstd: invalid Huffman weight sum"))
    last_w = _flog2(rest) + 1
    return (length(weights) + 1, last_w, table_log)
end

# Decode FSE-compressed Huffman weights.
# Uses two interleaved FSE states that are updated alternately.
# RFC 8878 §4.2.1.1
function _decode_fse_weights(br::ForwardBitReader, byte_limit::Int)
    # Read the FSE table for weights
    al, dist  = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t         = build_fse_table(dist, al)

    # The remainder of the weight description is a reverse bitstream
    pos_after = byte_pos(br)
    n_remain  = byte_limit - pos_after + 1
    n_remain > 0 || throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    rb = ReverseBitReader(br.data, pos_after, n_remain)

    # Init two interleaved states
    state1 = fse_init!(rb, t)
    state2 = fse_init!(rb, t)

    weights = UInt8[]
    sizehint!(weights, 256)
    # Interleaved FSE decode matching the reference zstd tail loop.
    # Each iteration: decode symbol (peek + update), emit it, then
    # check if the stream overflowed.  After overflow, emit the OTHER
    # state's pending symbol and break.
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

    # Advance the forward reader past the weight data
    br.pos   = byte_limit + 1
    br.nbits = 0
    br.bits  = UInt64(0)

    return weights
end

# Read the Huffman tree description and return a HuffmanTable.
# data[pos] is the first byte of the description.
# Returns (HuffmanTable, bytes_consumed).
function read_huffman_description(data::Vector{UInt8}, pos::Int)
    header = Int(data[pos])
    if header < 128
        # FSE-compressed weights: header = compressed_size
        compressed_size = header
        # Forward bit reader over the compressed bytes
        br = ForwardBitReader(data, pos + 1, pos + compressed_size + 1)
        weights = _decode_fse_weights(br, pos + compressed_size)
        last_sym, last_w, table_log = _infer_last_weight(weights)
        push!(weights, UInt8(last_w))
        ht = build_huffman_table(weights, table_log)
        return ht, compressed_size + 1
    else
        # Direct representation: (header - 127) weight nibbles follow
        nsyms = header - 127
        # Each byte holds two nibbles; ceil(nsyms/2) bytes
        nbytes = (nsyms + 1) >> 1
        weights = Vector{UInt8}(undef, nsyms)
        for i in 1:nbytes
            b = data[pos + i]
            lo = b & 0x0f
            hi = (b >> 4) & 0x0f
            idx = (i-1)*2
            if idx + 1 ≤ nsyms
                weights[idx + 1] = hi    # high nibble = first weight
            end
            if idx + 2 ≤ nsyms
                weights[idx + 2] = lo    # low nibble = second weight
            end
        end
        last_sym, last_w, table_log = _infer_last_weight(weights)
        if last_sym ≤ nsyms
            weights[last_sym] = UInt8(last_w)
        else
            push!(weights, UInt8(last_w))
        end
        ht = build_huffman_table(weights, table_log)
        return ht, nbytes + 1
    end
end


# ============================================================
# Section 7: Dictionary support
#   Reference: RFC 8878 §5
# ============================================================

struct ZstdDict
    id      ::UInt32
    huffman ::Union{HuffmanTable,Nothing}
    of_tab  ::Union{FSETable,Nothing}
    ml_tab  ::Union{FSETable,Nothing}
    ll_tab  ::Union{FSETable,Nothing}
    rep     ::NTuple{3,Int}
    content ::Vector{UInt8}
end

"""
    parse_dictionary(raw::Vector{UInt8}) -> ZstdDict

Parse a Zstandard dictionary (RFC 8878 §5).  Accepts both structured
dictionaries (magic `0xEC30A437`) and raw content dictionaries.
"""
function parse_dictionary(raw::Vector{UInt8})
    length(raw) ≥ 8 || throw(ArgumentError("zstd: dictionary too short"))
    magic = UInt32(raw[1]) | (UInt32(raw[2]) << 8) |
            (UInt32(raw[3]) << 16) | (UInt32(raw[4]) << 24)
    if magic != ZSTD_DICT_MAGIC
        # Raw content dictionary: no entropy tables, default repeat offsets
        return ZstdDict(UInt32(0), nothing, nothing, nothing, nothing, INIT_REPEAT_OFFSETS, raw)
    end

    # Structured dictionary
    dict_id = UInt32(raw[5]) | (UInt32(raw[6]) << 8) |
              (UInt32(raw[7]) << 16) | (UInt32(raw[8]) << 24)
    pos = 9

    # 1. Huffman table for literals
    ht, hdr_len = read_huffman_description(raw, pos)
    pos += hdr_len

    # 2. FSE table for offsets
    br = ForwardBitReader(raw, pos, length(raw) + 1)
    of_al, of_dist = read_fse_dist!(br, MAX_OFFSET_CODE)
    of_tab = build_fse_table(of_dist, of_al)

    # 3. FSE table for match lengths
    ml_al, ml_dist = read_fse_dist!(br, MAX_MATCH_LENGTH)
    ml_tab = build_fse_table(ml_dist, ml_al)

    # 4. FSE table for literals lengths
    ll_al, ll_dist = read_fse_dist!(br, MAX_LITERALS_LENGTH)
    ll_tab = build_fse_table(ll_dist, ll_al)

    pos = byte_pos(br)

    # 5. Three repeat offsets, 4 bytes LE each
    length(raw) ≥ pos + 11 || throw(ArgumentError("zstd: dictionary truncated (repeat offsets)"))
    rep1 = Int(_le32(raw, pos));     pos += 4
    rep2 = Int(_le32(raw, pos));     pos += 4
    rep3 = Int(_le32(raw, pos));     pos += 4

    content = raw[pos:end]
    return ZstdDict(dict_id, ht, of_tab, ml_tab, ll_tab, (rep1, rep2, rep3), content)
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

    rb = ReverseBitReader(br.data, pos_after, n_remain)

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
        br = ForwardBitReader(data, pos + 1, pos + compressed_size + 1)
        weights = _decode_fse_weights(br, pos + compressed_size, state.huf_weights)
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
function _decode_4streams!(data::Vector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int,
                            huf_start::Int, payload_end::Int) where L
    j1 = Int(_le16(data, huf_start))
    j2 = Int(_le16(data, huf_start + 2))
    j3 = Int(_le16(data, huf_start + 4))
    s1_start = huf_start + 6
    s2_start = s1_start + j1
    s3_start = s2_start + j2
    s4_start = s3_start + j3
    s4_end   = payload_end

    seg_n = (regen_size + 3) >> 2

    rb1 = ReverseBitReader(data, s1_start, j1)
    rb2 = ReverseBitReader(data, s2_start, j2)
    rb3 = ReverseBitReader(data, s3_start, j3)
    rb4 = ReverseBitReader(data, s4_start, s4_end - s4_start + 1)

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
    o1 = 1; o2 = 1 + seg_n; o3 = 1 + 2seg_n; o4 = 1 + 3seg_n
    end1 = seg_n; end2 = 2seg_n; end3 = 3seg_n; end4 = regen_size
    safeend1 = end1 - 2safe_n; safeend2 = end2 - 2safe_n
    safeend3 = end3 - 2safe_n; safeend4 = end4 - 2safe_n

    while o1 ≤ safeend1 && o2 ≤ safeend2 &&
          o3 ≤ safeend3 && o4 ≤ safeend4
        _rbr_refill!(rb1)
        _rbr_refill!(rb2)
        _rbr_refill!(rb3)
        _rbr_refill!(rb4)
        for _ in 1:safe_n
            o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
            o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
        end
    end

    while o1 ≤ safeend1 && o2 ≤ safeend2
        _rbr_refill!(rb1)
        _rbr_refill!(rb2)
        for _ in 1:safe_n
            o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
        end
    end

    while o3 ≤ safeend3 && o4 ≤ safeend4
        _rbr_refill!(rb3)
        _rbr_refill!(rb4)
        for _ in 1:safe_n
            o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
        end
    end

    if o1 ≤ safeend1
        while o1 ≤ safeend1
            _rbr_refill!(rb1)
            for _ in 1:safe_n
                o1 += _huffman_decode2_nocheck!(rb1, ht, literals, o1)
            end
        end
    elseif o2 ≤ safeend2 # must check because possible to overshoot
        while o2 ≤ safeend2
            _rbr_refill!(rb2)
            for _ in 1:safe_n
                o2 += _huffman_decode2_nocheck!(rb2, ht, literals, o2)
            end
        end
    end

    if o3 ≤ safeend3
        while o3 ≤ safeend3
            _rbr_refill!(rb3)
            for _ in 1:safe_n
                o3 += _huffman_decode2_nocheck!(rb3, ht, literals, o3)
            end
        end
    elseif o4 ≤ safeend4 # must check because possible to overshoot
        while o4 ≤ safeend4
            _rbr_refill!(rb4)
            for _ in 1:safe_n
                o4 += _huffman_decode2_nocheck!(rb4, ht, literals, o4)
            end
        end
    end

    while o1 ≤ end1; @inbounds literals[o1] = huffman_decode!(rb1, ht); o1 += 1; end
    while o2 ≤ end2; @inbounds literals[o2] = huffman_decode!(rb2, ht); o2 += 1; end
    while o3 ≤ end3; @inbounds literals[o3] = huffman_decode!(rb3, ht); o3 += 1; end
    while o4 ≤ end4; @inbounds literals[o4] = huffman_decode!(rb4, ht); o4 += 1; end
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
            rb = ReverseBitReader(data, huf_start, stream_len)
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
                # Non-overlapping match.
                GC.@preserve out Base.memcpy(pointer(out, wpos), pointer(out, match_pos), ml)
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

    br = ForwardBitReader(data, pos, limit + 1)
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

    rb = ReverseBitReader(data, seq_start, seq_len)

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
        # chain (each read_bits_r! currently gates the next via rb.bits).
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
            rb.nbits < total_n && _rbr_refill!(rb)
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
            of_extra   = (of_n  > 0) ? Int(read_bits_r!(rb, of_n )) : 0
            ml_extra   = Int(read_bits_r!(rb, ml_n))
            ll_extra   = Int(read_bits_r!(rb, ll_n))
            ll_bits    = (ll_nb > 0) ? Int(read_bits_r!(rb, ll_nb)) : 0
            ml_bits    = (ml_nb > 0) ? Int(read_bits_r!(rb, ml_nb)) : 0
            of_bits    = (of_nb > 0) ? Int(read_bits_r!(rb, of_nb)) : 0
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
