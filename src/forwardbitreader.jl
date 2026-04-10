# ============================================================
# Forward bit reader
#   Used for non-data bitstreams
# ============================================================

mutable struct ForwardBitReader
    data::Vector{UInt8}
    pos::Int     # next byte to load into the buffer (1-indexed)
    limit::Int   # one past the last valid byte (exclusive, 1-indexed)
    bits::UInt64 # bit buffer; bit 0 is the next bit to deliver
    nbits::Int   # number of valid bits currently in buffer
end

function ForwardBitReader(data::Vector{UInt8}, byte_offset::Int, byte_limit::Int)
    ForwardBitReader(data, byte_offset, byte_limit, UInt64(0), 0)
end

function _fbr_refill!(br::ForwardBitReader)
    while br.nbits ≤ 56 && br.pos < br.limit
        br.bits  |= _shl(UInt64(br.data[br.pos]), br.nbits)
        br.nbits += 8
        br.pos   += 1
    end
end

@inline function read_bits!(br::ForwardBitReader, n::Int)
    n > 0 ||
        return UInt64(0)
    br.nbits < n &&
        _fbr_refill!(br)
    br.nbits ≥ n ||
        throw(ArgumentError("zstd: unexpected end of bitstream (need $n bits, have $(br.nbits))"))
    readmask = (UInt64(1) << n) - UInt64(1)
    val = br.bits & readmask
    br.bits >>>= n
    br.nbits -= n
    return val
end

# Discard any sub-byte leftover so the reader is byte-aligned.
function align_to_byte!(br::ForwardBitReader)
    discard = br.nbits & 7
    discard > 0 &&
        (br.bits >>>= discard)
    br.nbits -= discard
    return nothing
end

# Byte position of the next unread byte (after draining the bit buffer).
byte_pos(br::ForwardBitReader) = br.pos - (br.nbits >> 3)
