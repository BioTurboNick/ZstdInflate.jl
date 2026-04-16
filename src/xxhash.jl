# ============================================================
#  xxHash XXH64
#   https://xxhash.com/
#   Reference algorithm: https://cyan4973.github.io/xxHash/
# ============================================================

const XXH_P1 = 0x9E3779B185EBCA87
const XXH_P2 = 0xC2B2AE3D27D4EB4F
const XXH_P3 = 0x165667B19E3779F9
const XXH_P4 = 0x85EBCA77C2B2AE63
const XXH_P5 = 0x27D4EB2F165667C5

@inline xxh_rotl64(x::Union{UInt64, Vec{4, UInt64}}, r) = _shl(x, r) | _shr(x, (64 - r))

@inline function xxh_round(acc::T, lane::T) where T <: Union{UInt64, Vec{4, UInt64}}
    acc += lane * XXH_P2
    return xxh_rotl64(acc, 31) * XXH_P1
end

@inline function xxh_merge(acc::T, val::T) where T <: Union{UInt64, Vec{4, UInt64}}
    acc ⊻= xxh_round(T(0), val)
    return acc * XXH_P1 + XXH_P4
end

function xxhash64(data::AbstractVector{UInt8}, seed::UInt64 = UInt64(0))
    len = length(data)
    pos = 1

    if len ≥ 32
        rdata64 = reinterpret(UInt64, @view data[1:end - (end % 8)])
        vs = Vec{4, UInt64}((
            seed + XXH_P1 + XXH_P2,
            seed + XXH_P2,
            seed,
            seed - XXH_P1))
        while 8(pos - 1) + 32 ≤ len
            vdata = @inbounds vloada(Vec{4, UInt64}, (@view rdata64[pos:pos + 3]), 1)
            vs = xxh_round(vs, ltoh(vdata))
            pos += 4
        end
        h64 = sum(xxh_rotl64(vs, Vec{4, Int64}((1, 7, 12, 18))))
        h64 = xxh_merge(h64, vs[1])
        h64 = xxh_merge(h64, vs[2])
        h64 = xxh_merge(h64, vs[3])
        h64 = xxh_merge(h64, vs[4])
    else
        h64 = seed + XXH_P5
    end

    h64 += UInt64(len)

    while 8(pos - 1) + 8 ≤ len
        rdata64 = reinterpret(UInt64, @view data[1:end - (end % 8)])
        h64 ⊻= xxh_round(UInt64(0), ltoh(rdata64[pos]))
        h64 = xxh_rotl64(h64, 27) * XXH_P1 + XXH_P4
        pos += 1
    end
    pos = 2 * (pos - 1) + 1
    if 4(pos - 1) + 4 ≤ len
        rdata32 = reinterpret(UInt32, @view data[1:end - (end % 4)])
        h64 ⊻= UInt64(ltoh(rdata32[pos])) * XXH_P1
        h64 = xxh_rotl64(h64, 23) * XXH_P2 + XXH_P3
        pos += 1
    end
    pos = 4 * (pos - 1) + 1
    while pos ≤ len
        h64 ⊻= UInt64(data[pos]) * XXH_P5
        h64 = xxh_rotl64(h64, 11) * XXH_P1
        pos += 1
    end

    h64 ⊻= h64 >>> 33;
    h64 *= XXH_P2
    h64 ⊻= h64 >>> 29;
    h64 *= XXH_P3
    h64 ⊻= h64 >>> 32
    return h64
end
