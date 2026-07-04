# Julia's default shift guards against shifts of ≥64 bits, which requires a branch and so
# becomes slow in tight loops. The shifts in Zstd are guaranteed to be <64 bits, so we can
# prove to the compiler that the branch is not necessary by masking the shift count. Thus,
# the compiler can emit a single shift instruction without the guard.
@inline _shl(x::UInt64,         n::Int64)                 = x << (n & 63)
@inline _shl(x::Vec{X, UInt64}, n::Int64)         where X = x << (n & Int64(63))
@inline _shl(x::Vec{X, UInt64}, n::Vec{X, Int64}) where X = x << (n & 63)
@inline _shr(x::UInt64,         n::Int64)                 = x >>> (n & 63)
@inline _shr(x::Vec{X, UInt64}, n::Int64)         where X = x >>> (n & Int64(63))
@inline _shr(x::Vec{X, UInt64}, n::Vec{X, Int64}) where X = x >>> (n & 63)

# Little-endian loads
@inline _le64(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt64}(pointer(d, i))))
@inline _le32(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt32}(pointer(d, i))))
@inline _le16(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt16}(pointer(d, i))))

# floor(log2(n)) for n ≥ 1
@inline _flog2(n::Int) = 63 - leading_zeros(UInt64(n))


splitbyte(b::UInt8) = b & 0x0F, b >> 4

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
