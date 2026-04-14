# Julia's default shift guards against shifts of ≥64 bits, which requires a branch and so
# becomes slow in tight loops. The shifts in Zstd are guaranteed to be <64 bits, so we can
# prove to the compiler that the branch is not necessary by masking the shift count. Thus,
# the compiler can emit a single shift instruction without the guard.
@inline _shl(x::UInt64,          n::Int64)               = x << (n & 63)
@inline _shl(x::Vec{X, UInt64}, n::Int64)          where X = x << (n & Int64(63))
@inline _shl(x::Vec{X, UInt64}, n::Vec{X, Int64})  where X = x << (n & 63)
@inline _shr(x::UInt64,          n::Int64)               = x >>> (n & 63)
@inline _shr(x::Vec{X, UInt64}, n::Int64)          where X = x >>> (n & Int64(63))
@inline _shr(x::Vec{X, UInt64}, n::Vec{X, Int64})  where X = x >>> (n & 63)

# Little-endian loads
@inline _le64(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt64}(pointer(d, i))))
@inline _le32(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt32}(pointer(d, i))))
@inline _le16(d, i) = GC.@preserve d ltoh(unsafe_load(Ptr{UInt16}(pointer(d, i))))
