# ============================================================
# Reverse bit reader
#   Used for encoded data bitstreams (literals, sequences)
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
        raw = _le64(rb.data, pos - 7)
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
    n == 0 &&
        return UInt64(0)
    rb.nbits < n &&
        _rbr_refill!(rb)
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
    _rbr_refill!(rb)
    val = _shr(rb.bits, 64 - n)
    rb.bits <<= n
    rb.nbits -= n
    return val
end

# Check whether previous unchecked reads consumed more bits than available.
@inline _rbr_overflowed(rb::ReverseBitReader) = rb.nbits < 0
