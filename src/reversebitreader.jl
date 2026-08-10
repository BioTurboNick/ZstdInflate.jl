# ============================================================
# Reverse bit reader
#   Used for encoded data bitstreams (literals, sequences)
# ============================================================

# Concrete type of every `@view data[a:b]` slice these readers are built
# from (frame bytes are always a plain Vector{UInt8}, and nested views over a
# UnitRange flatten to this same type rather than nesting SubArrays). Scratch
# fields reused across calls (DecompressState) need a concrete, matching
# type parameter to reinitialize in place without reallocating.
const RBRView = SubArray{UInt8, 1, Vector{UInt8}, Tuple{UnitRange{Int64}}, true}

mutable struct ReverseBitReader{T <: AbstractVector{UInt8}}
    data::T
    pos::Int     # next byte to load (decreasing, 1-indexed)
    bits::UInt64 # bit buffer; MSB is the next bit to deliver
    nbits::Int   # number of valid bits currently in buffer
end

# Parse the sentinel bit at the end of a reverse bitstream, returning the
# pre-initial-refill (pos, bits, nbits). Shared by the allocating constructor
# and `reinit!` below.
@inline function _rbr_sentinel(data::AbstractVector{UInt8})
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
    return length(data) - 1, bits, nbits
end

# Full initial state (post-first-refill) for a fresh reverse bitstream, with
# no reader object involved. `_refill_stream` already unifies the fast
# (pos ≥ 8), padded (pos < 8), and bytewise (length(data) ≤ 8) cases that
# `refill!`/`_refill_bytewise!` otherwise dispatch between, so this is a
# drop-in replacement for "construct + do the initial refill" that never
# needs a temporary reader to get there.
@inline function _rbr_init_state(data::AbstractVector{UInt8})
    pos, bits, nbits = _rbr_sentinel(data)
    bits, nbits, pos = _refill_stream(data, bits, nbits, pos)
    return pos, bits, nbits
end

# Expects the last byte to contain at least one set bit, which indicates the end of the bitstream.
function ReverseBitReader(data::T) where T <: AbstractVector{UInt8}
    pos, bits, nbits = _rbr_init_state(data)
    return ReverseBitReader{T}(data, pos, bits, nbits)
end

# Reinitialize an existing reader for a new bitstream in place, instead of
# allocating a fresh one. `data` must share `rb`'s existing type parameter.
function reinit!(rb::ReverseBitReader{T}, data::T) where T <: AbstractVector{UInt8}
    pos, bits, nbits = _rbr_init_state(data)
    rb.data = data
    rb.pos = pos
    rb.bits = bits
    rb.nbits = nbits
    return rb
end

@inline function Base.peek(rb::ReverseBitReader, ::Val{L}) where L
    _shr(rb.bits, Int64(64 - L))
end

@inline function Base.skip(rb::ReverseBitReader, nbits_offset::Int)
    n = Int(nbits_offset)
    rb.bits = _shl(rb.bits, n)
    rb.nbits = rb.nbits - n
    return
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

@inline function Base.read(rb::ReverseBitReader, n::Int)
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
mutable struct ReverseBitReaderX{X, T <: AbstractVector{UInt8}}
    data ::NTuple{X, T}
    bits ::NTuple{X, UInt64}
    nbits::NTuple{X, Int64}
    pos  ::NTuple{X, Int64}
end

# Construct from four data slices. Uses `_rbr_init_state` directly rather than
# building and discarding four temporary `ReverseBitReader`s.
function ReverseBitReaderX(d1::T, d2::T, d3::T, d4::T) where T <: AbstractVector{UInt8}
    p1, b1, n1 = _rbr_init_state(d1)
    p2, b2, n2 = _rbr_init_state(d2)
    p3, b3, n3 = _rbr_init_state(d3)
    p4, b4, n4 = _rbr_init_state(d4)
    ReverseBitReaderX{4, T}((d1, d2, d3, d4),
                            (b1, b2, b3, b4),
                            (Int64(n1), Int64(n2), Int64(n3), Int64(n4)),
                            (Int64(p1), Int64(p2), Int64(p3), Int64(p4)))
end

# Reinitialize an existing 4-lane reader in place for four new data slices.
function reinit!(rb::ReverseBitReaderX{4, T}, d1::T, d2::T, d3::T, d4::T) where T <: AbstractVector{UInt8}
    p1, b1, n1 = _rbr_init_state(d1)
    p2, b2, n2 = _rbr_init_state(d2)
    p3, b3, n3 = _rbr_init_state(d3)
    p4, b4, n4 = _rbr_init_state(d4)
    rb.data  = (d1, d2, d3, d4)
    rb.bits  = (b1, b2, b3, b4)
    rb.nbits = (Int64(n1), Int64(n2), Int64(n3), Int64(n4))
    rb.pos   = (Int64(p1), Int64(p2), Int64(p3), Int64(p4))
    return rb
end

function ReverseBitReaderX(d1::T, d2::T) where T <: AbstractVector{UInt8}
    p1, b1, n1 = _rbr_init_state(d1)
    p2, b2, n2 = _rbr_init_state(d2)
    ReverseBitReaderX{2, T}((d1, d2), (b1, b2), (Int64(n1), Int64(n2)), (Int64(p1), Int64(p2)))
end

# Reinitialize an existing 2-lane reader in place for two new data slices.
function reinit!(rb::ReverseBitReaderX{2, T}, d1::T, d2::T) where T <: AbstractVector{UInt8}
    p1, b1, n1 = _rbr_init_state(d1)
    p2, b2, n2 = _rbr_init_state(d2)
    rb.data  = (d1, d2)
    rb.bits  = (b1, b2)
    rb.nbits = (Int64(n1), Int64(n2))
    rb.pos   = (Int64(p1), Int64(p2))
    return rb
end

function ReverseBitReaderX(rb1::ReverseBitReader{T}, rb2::ReverseBitReader{T}) where T <: AbstractVector{UInt8}
    bits  = (rb1.bits,         rb2.bits)
    nbits = (Int64(rb1.nbits), Int64(rb2.nbits))
    pos   = (Int64(rb1.pos),   Int64(rb2.pos))
    ReverseBitReaderX{2, T}((rb1.data, rb2.data), bits, nbits, pos)
end

# Reinitialize an existing 2-lane reader in place from two already-live
# single readers (repackaging, no new initial refill).
function reinit!(rb::ReverseBitReaderX{2, T}, rb1::ReverseBitReader{T}, rb2::ReverseBitReader{T}) where T <: AbstractVector{UInt8}
    rb.data  = (rb1.data, rb2.data)
    rb.bits  = (rb1.bits, rb2.bits)
    rb.nbits = (Int64(rb1.nbits), Int64(rb2.nbits))
    rb.pos   = (Int64(rb1.pos), Int64(rb2.pos))
    return rb
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

# ============================================================
# Sentinel-bit bit-position encoding
#
# Borrowed from libzstd, which uses it in both huf_decompress_amd64.S and the
# plain-C fast loop it falls back to. Its own description:
#
#   "bits[] is the bit container. It is read from the MSB down to the LSB. It
#    is shifted left as it is read, and zeros are shifted in. After the lowest
#    valid bit a 1 is set, so that CountTrailingZeros(bits[]) can be used to
#    count how many bits we've consumed."
#
# The point is register pressure. Tracking a separate `nbits` per stream costs
# one live register per stream and one `sub` per symbol; encoding the same
# information as the position of a sentinel bit *inside* the container costs
# neither, and is recovered with a single `tzcnt` once per refill rather than
# once per symbol.
#
# It also makes the per-symbol shift count free of masking: with `nbits` gone,
# the consumed-bit count is only ever fed to `_shl`, which masks to 6 bits, so
# adjacent packed fields in the table entry can be left in the high bits
# instead of being masked off.
#
# Invariants, with `s = trailing_zeros(bits)`:
#   * valid bits remaining = 63 - s   (bit 63 is the next bit to deliver)
#   * consuming n bits is `bits <<= n`, which moves the sentinel up by n
#   * at most 56 bits may be consumed between refills, so the sentinel can
#     never be shifted out of the container

# Sentinel form tracks a different byte cursor from the conventional readers.
# `pos` there is the highest byte *not yet loaded*; `P` here is the byte
# *containing the next bit to deliver*, which is what the refill steps back
# from. `_sent_byte_pos` converts one to the other: the lowest buffered bit is
# the LSB of byte pos+1, so the next bit to deliver sits (nbits - 1) bits above
# it.
@inline _sent_byte_pos(nbits::Int, pos::Int) = pos + 1 + ((nbits - 1) >> 3)

# Enter sentinel form. Requires nbits ≥ 1 and `_sent_byte_pos(nbits, pos) ≥ 8`;
# callers check both. The container is re-read from memory rather than derived
# from the conventional one, which keeps the bit accounting in one place.
@inline function _sent_enter(data::AbstractVector{UInt8}, nbits::Int, pos::Int)
    P = _sent_byte_pos(nbits, pos)
    return (_le64(data, P - 7) | UInt64(1)) << (7 - ((nbits - 1) & 7)), P
end

# Leave sentinel form, returning a conventional (bits, nbits, pos) triple that
# buffers only what is left of the byte holding the next bit. The caller's own
# refill tops it back up. Re-deriving from memory like this sidesteps the fact
# that a sentinel container's valid region generally does not end on a byte
# boundary, which the conventional encoding cannot represent.
@inline function _sent_leave(data::AbstractVector{UInt8}, bits::UInt64, P::Int)
    consumed = trailing_zeros(bits)
    Pc = P - (consumed >> 3)
    r  = consumed & 7
    v  = UInt64(@inbounds(data[Pc]) & (0xff >> r))
    return v << (56 + r), 8 - r, Pc - 1
end

# Refill in sentinel form: the sentinel's position gives both how many whole
# bytes to step back and how many leftover bits of the boundary byte to
# discard, so the container is replaced outright rather than merged into.
# Requires `P ≥ 15`: the step back is at most 7 bytes and the load then needs 8
# bytes at or below the new position.
@inline function _sent_refill(data::AbstractVector{UInt8}, bits::UInt64, P::Int)
    consumed = trailing_zeros(bits)
    P -= consumed >> 3
    return (_le64(data, P - 7) | UInt64(1)) << (consumed & 7), P
end

function refill_unchecked!(rb::ReverseBitReaderX{X}) where X
    # Any stream within 8 bytes of its start must take the scalar path
    if any(Tuple(rb.pos) .< 8)
        r = ntuple(i -> _refill_stream(rb.data[i], rb.bits[i], Int(rb.nbits[i]), Int(rb.pos[i])), Val(X))
        rb.bits = ntuple(i -> r[i][1], Val(X))
        rb.nbits = ntuple(i -> Int64(r[i][2]), Val(X))
        rb.pos  = ntuple(i -> Int64(r[i][3]), Val(X))
        return
    end

    nread    = (64 .- rb.nbits) .>>> 3             # logical shift: value always ≥ 0, avoids vpsrad emulation
    raw      = _le64.(rb.data, rb.pos .- 7)        # NTuple{X,UInt64}, 8-byte loads
    readmask = _readmask.(nread)                   # NTuple{X,UInt64}, top 8*nread bits
    rb.bits  = rb.bits .| ((raw .& readmask) .>>> (rb.nbits .& Int64(63)))
    rb.nbits = rb.nbits .+ 8 .* nread
    rb.pos   = rb.pos .- nread
    return
end

@inline function Base.peek(rb::ReverseBitReaderX, ::Val{L}) where L
    _shr.(rb.bits, Int64(64 - L))
end

@inline function Base.skip(rb::ReverseBitReaderX{X}, nbits_offset::NTuple{X, Int}) where X
    rb.bits  = _shl.(rb.bits, Int64.(nbits_offset))
    rb.nbits = rb.nbits .- nbits_offset
    return
end

# Extract one stream into an existing ReverseBitReader (used for tail phases).
# Writes into `dst` in place instead of allocating a fresh reader.
@inline function _extract_stream!(dst::ReverseBitReader{T}, rb2x::ReverseBitReaderX{2, T}, ::Val{I}) where {T, I}
    dst.data  = rb2x.data[I]
    dst.pos   = Int(rb2x.pos[I])
    dst.bits  = rb2x.bits[I]
    dst.nbits = Int(rb2x.nbits[I])
    return dst
end

@inline function _extract_stream!(dst::ReverseBitReader{T}, rb4x::ReverseBitReaderX{4, T}, ::Val{I}) where {T, I}
    data = I == 1 ? rb4x.data[1] : I == 2 ? rb4x.data[2] : I == 3 ? rb4x.data[3] : rb4x.data[4]
    dst.data  = data
    dst.pos   = Int(rb4x.pos[I])
    dst.bits  = rb4x.bits[I]
    dst.nbits = Int(rb4x.nbits[I])
    return dst
end

# ============================================================
# Scratch pool for _decode_4streams! (huffman.jl)
#   Bundles every ReverseBitReader/ReverseBitReaderX that function needs so
#   DecompressState can hold and reuse them across calls instead of
#   allocating a fresh set (3 X-readers + 8 single readers) per literals
#   section.
# ============================================================

mutable struct Huffman4StreamScratch{T}
    rb4x  ::ReverseBitReaderX{4, T}
    rbA   ::ReverseBitReaderX{2, T}
    rbB   ::ReverseBitReaderX{2, T}
    s1    ::ReverseBitReader{T}
    s2    ::ReverseBitReader{T}
    s3    ::ReverseBitReader{T}
    s4    ::ReverseBitReader{T}
    ra_ia ::ReverseBitReader{T}
    ra_ib ::ReverseBitReader{T}
    rb_2b ::ReverseBitReader{T}
    rb_ic2::ReverseBitReader{T}
end

# Placeholder-initialize a scratch pool from a dummy data slice. The dummy
# 1-byte stream (0x01, i.e. sentinel-only, zero payload bits) is a
# valid-but-empty reverse bitstream, just enough to satisfy the reader
# constructors; every field is overwritten by `reinit!`/`_extract_stream!`
# before real use.
function Huffman4StreamScratch(dummy::T) where T <: AbstractVector{UInt8}
    Huffman4StreamScratch{T}(
        ReverseBitReaderX(dummy, dummy, dummy, dummy),
        ReverseBitReaderX(dummy, dummy),
        ReverseBitReaderX(dummy, dummy),
        ReverseBitReader(dummy), ReverseBitReader(dummy),
        ReverseBitReader(dummy), ReverseBitReader(dummy),
        ReverseBitReader(dummy), ReverseBitReader(dummy),
        ReverseBitReader(dummy), ReverseBitReader(dummy),
    )
end
