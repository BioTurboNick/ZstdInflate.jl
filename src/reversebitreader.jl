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
    _first_refill!(rb)
    return rb
end

function _first_refill!(rb::ReverseBitReader)
    if length(rb.data) ≤ 8
        _refill_bytewise!(rb)
    else
        refill!(rb)
    end
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
        mask = _shl(typemax(UInt64), 64 - 8nread)
        rb.bits |= _shr(raw & mask, rb.nbits)
    else
        # At start of data; load 8 bytes with zero padding, then shift into position
        raw = _le64(rb.data, 1)
        mask = _shl(typemax(UInt64), 64 - 8nread)
        mask = _shr(mask, 64 - 8rb.pos)
        loaded = _shl(raw & mask, 64 - 8rb.pos)
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

@inline function read_bits_r!(rb::ReverseBitReader, n::Int)
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
@inline function _read_bits_r_unchecked!(rb::ReverseBitReader, n::Int)
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
