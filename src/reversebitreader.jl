# ============================================================
# Reverse bit reader
#   Used for encoded data bitstreams (literals, sequences)
# ============================================================

mutable struct ReverseBitReader{T <: AbstractVector{UInt8}}
    data::T
    pos::Int     # next byte to load (decreasing, 1-indexed)
    bits::UInt64 # bit buffer; MSB is the next bit to deliver
    nbits::Int   # number of valid bits currently in buffer
end

# Expects the last byte to contain at least one set bit, which indicates the end of the bitstream.
function ReverseBitReader(data::T) where T <: AbstractVector{UInt8}
    isempty(data) &&
        throw(ArgumentError("zstd: empty reverse bitstream"))
    lastbyte = data[end]
    lastbyte != 0 ||
        throw(ArgumentError("zstd: reverse bitstream sentinel byte is zero"))

    # Read and clear sentinel bit
    nbits = 7 - leading_zeros(lastbyte)
    valid_bits = lastbyte ⊻ (UInt8(1) << nbits)

    # Pack into the MSB region of the 64-bit container
    bits = UInt64(valid_bits) << (64 - nbits)

    rb = ReverseBitReader{T}(data, length(data) - 1, bits, nbits)
    if length(rb.data) ≤ 8
        _refill_bytewise!(rb)
    else
        refill!(rb)
    end
    return rb
end

function refill!(rb::ReverseBitReader)
    length(rb.data) > 8 || # Streams this short are already depleted after initialization
        return
    navail = rb.pos
    (navail > 0 && rb.nbits < 57) ||
        return

    nread = min((64 - rb.nbits) >> 3, navail)

    if rb.pos ≥ 8
        # Load 8 bytes, zero out the lower bits that aren't needed yet, and shift into position
        raw = _le64(rb.data, rb.pos - 7)
        readmask = _shl(typemax(UInt64), 64 - 8nread)
        rb.bits |= _shr(raw & readmask, rb.nbits)
    else
        # At start of data; load 8 bytes with zero padding, then shift into position
        raw = _le64(rb.data, 1)
        readmask = _shl(typemax(UInt64), 64 - 8nread)
        readmask = _shr(readmask, 64 - 8rb.pos)
        loaded = _shl(raw & readmask, 64 - 8rb.pos)
        rb.bits |= _shr(loaded, rb.nbits)
    end

    rb.nbits += 8nread
    rb.pos -= nread
    return
end

function _refill_bytewise!(rb::ReverseBitReader)
    navail = rb.pos
    (navail > 0 && rb.nbits < 57) ||
        return

    nread = min((64 - rb.nbits) >> 3, navail)

    s = 56 - rb.nbits
    for i in 0:nread - 1
        rb.bits |= _shl(UInt64(rb.data[rb.pos - i]), s - 8i)
    end
    rb.nbits += 8nread
    rb.pos -= nread
    return
end

@inline function read_bits!(rb::ReverseBitReader, n::Int)
    n == 0 &&
        return UInt64(0)
    rb.nbits < n &&
        refill!(rb)
    rb.nbits ≥ n ||
        throw(ArgumentError("zstd: unexpected end of reverse bitstream"))
    val = _shr(rb.bits, 64 - n)
    rb.bits <<= n
    rb.nbits -= n
    return val
end

# Read n bits without checking for underflow (allows nbits to go negative).
# Used by the interleaved FSE weight decoder tail loop where overflow is
# detected after the read rather than before.
@inline function _read_bits_unchecked!(rb::ReverseBitReader, n::Int)
    n == 0 &&
        return UInt64(0)
    refill!(rb)
    val = _shr(rb.bits, 64 - n)
    rb.bits <<= n
    rb.nbits -= n
    return val
end

# Check whether previous unchecked reads consumed more bits than available.
@inline _rbr_overflowed(rb::ReverseBitReader) = rb.nbits < 0

# ============================================================
# 4-stream SIMD reverse bit reader
#   bits/nbits/pos stored as NTuple{4,...} so they can be
#   loaded into Vec{4,...} and updated in parallel.
# ============================================================

# Inner refill logic for one stream. Returns (new_bits, new_nbits, new_pos).
# Mirrors refill! / _refill_bytewise! but is stateless (no mutation).
@inline function _refill_stream(data::AbstractVector{UInt8},
                                 bits::UInt64, nbits::Int, pos::Int)
    navail = pos
    (navail > 0 && nbits < 57) || return (bits, nbits, pos)
    nread = min((64 - nbits) >> 3, navail)
    if length(data) > 8
        if pos ≥ 8
            raw      = _le64(data, pos - 7)
            readmask = _shl(typemax(UInt64), 64 - 8nread)
            bits    |= _shr(raw & readmask, nbits)
        else
            raw      = _le64(data, 1)
            readmask = _shl(typemax(UInt64), 64 - 8nread)
            readmask = _shr(readmask, 64 - 8pos)
            loaded   = _shl(raw & readmask, 64 - 8pos)
            bits    |= _shr(loaded, nbits)
        end
    else
        s = 56 - nbits
        for i in 0:nread - 1
            bits |= _shl(UInt64(data[pos - i]), s - 8i)
        end
    end
    return (bits, nbits + 8nread, pos - nread)
end

# Groups four reverse bit streams whose hot-path state (bits/nbits/pos) is
# stored as NTuple{4,...} so a single Vec load covers all four lanes.
mutable struct ReverseBitReader4X{T <: AbstractVector{UInt8}}
    data ::NTuple{4, T}
    bits ::NTuple{4, UInt64}
    nbits::NTuple{4, Int64}
    pos  ::NTuple{4, Int64}
end

# Construct from four data slices, delegating per-stream init (sentinel handling,
# initial refill) to the existing ReverseBitReader constructor.
function ReverseBitReader4X(d1::T, d2::T, d3::T, d4::T) where T <: AbstractVector{UInt8}
    rb1 = ReverseBitReader(d1)
    rb2 = ReverseBitReader(d2)
    rb3 = ReverseBitReader(d3)
    rb4 = ReverseBitReader(d4)
    bits  = (rb1.bits,         rb2.bits,         rb3.bits,         rb4.bits)
    nbits = (Int64(rb1.nbits), Int64(rb2.nbits), Int64(rb3.nbits), Int64(rb4.nbits))
    pos   = (Int64(rb1.pos),   Int64(rb2.pos),   Int64(rb3.pos),   Int64(rb4.pos))
    ReverseBitReader4X{T}((d1, d2, d3, d4), bits, nbits, pos)
end

# Refill all four streams.
#
# Fast path (all pos ≥ 8): the four 8-byte loads are scalar (different addresses),
# but the OR into bits is done with Vec{4,UInt64} ops.
#
# When pos ≥ 8, nread = (64 - nbits) >> 3 ≤ 8 ≤ pos, so min(nread, navail) = nread.
#
# When nbits ≥ 57, nread = 0. The readmask formula
#   ~((UInt64(1) << ((8 - nread) * 8)) - 1)
# uses plain Julia `<<` (not `_shl`), which returns 0 for shift ≥ 64. So:
#   nread = 0  →  shift = 64  →  1 << 64 = 0  →  ~(0 - 1) = ~typemax = 0
# Readmask = 0 makes the OR a no-op, and nread = 0 makes the nbits/pos deltas zero.
# No explicit branch on nbits < 57 is needed.
#
# Slow path (any pos < 8): scalar fallback via _refill_stream.
# Compute a mask with the top 8*nread bits set (0 when nread=0).
# Scalar path: _shl(typemax, 64-nread*8) is wrong for nread=0 (64 & 63 = 0 → typemax),
# so use ifelse to zero it out branchlessly (CMOV, no branch).
# Vec path: vpsllvq returns 0 for shift=64 natively, so the formula is always correct.
@inline _readmask(nread::Int) =
    ifelse(nread > 0, _shl(typemax(UInt64), 64 - nread * 8), UInt64(0))
@inline _readmask(nread::Union{Vec{4, Int64}, Vec{4, UInt64},
                               Vec{2, Int64}, Vec{2, UInt64}}) =
    ~((UInt64(1) << ((8 - nread) * 8)) - UInt64(1))


function refill_unchecked!(rb::ReverseBitReader4X)
    nread    = (64 .- rb.nbits) .>>> 3             # logical shift: value always ≥ 0, avoids vpsrad emulation
    raw      = _le64.(rb.data, rb.pos .- 7)        # NTuple{4,UInt64}, 8-byte loads
    readmask = _readmask.(nread)                    # NTuple{4,UInt64}, top 8*nread bits
    rb.bits  = rb.bits .| ((raw .& readmask) .>>> (rb.nbits .& Int64(63)))
    rb.nbits = rb.nbits .+ 8 .* nread
    rb.pos   = rb.pos .- nread
    return
end

# Extract one stream as an individual ReverseBitReader (used for tail phases).
@inline function _extract_stream(rb4x::ReverseBitReader4X{T}, ::Val{I}) where {T, I}
    data = I == 1 ? rb4x.data[1] : I == 2 ? rb4x.data[2] : I == 3 ? rb4x.data[3] : rb4x.data[4]
    ReverseBitReader{T}(data, Int(rb4x.pos[I]), rb4x.bits[I], Int(rb4x.nbits[I]))
end

# ============================================================
# 2-stream SIMD reverse bit reader
# ============================================================

mutable struct ReverseBitReader2X{T <: AbstractVector{UInt8}}
    data ::NTuple{2, T}
    bits ::NTuple{2, UInt64}
    nbits::NTuple{2, Int64}
    pos  ::NTuple{2, Int64}
end

# Construct from two streams of a ReverseBitReader4X (no re-init needed;
# bits/nbits/pos are already populated by the 4X reader).
@inline function ReverseBitReader2X(rb4x::ReverseBitReader4X{T}, ::Val{A}, ::Val{B}) where {T, A, B}
    ReverseBitReader2X{T}(
        (rb4x.data[A],         rb4x.data[B]),
        (rb4x.bits[A],         rb4x.bits[B]),
        (rb4x.nbits[A],        rb4x.nbits[B]),
        (rb4x.pos[A],          rb4x.pos[B]),
    )
end

function refill_unchecked!(rb::ReverseBitReader2X)
    nread    = (64 .- rb.nbits) .>>> 3
    raw      = _le64.(rb.data, rb.pos .- 7)
    readmask = _readmask.(nread)
    rb.bits  = rb.bits .| ((raw .& readmask) .>>> (rb.nbits .& Int64(63)))
    rb.nbits = rb.nbits .+ 8 .* nread
    rb.pos   = rb.pos .- nread
    return
end

@inline function _extract_stream(rb2x::ReverseBitReader2X{T}, ::Val{I}) where {T, I}
    ReverseBitReader{T}(rb2x.data[I], Int(rb2x.pos[I]), rb2x.bits[I], Int(rb2x.nbits[I]))
end

# Construct a ReverseBitReader2X from two individual ReverseBitReaders
# (e.g. to cross-pair streams after a 2X phase).
@inline function ReverseBitReader2X(rba::ReverseBitReader{T}, rbb::ReverseBitReader{T}) where T
    ReverseBitReader2X{T}(
        (rba.data,         rbb.data),
        (rba.bits,         rbb.bits),
        (Int64(rba.nbits), Int64(rbb.nbits)),
        (Int64(rba.pos),   Int64(rbb.pos)),
    )
end
