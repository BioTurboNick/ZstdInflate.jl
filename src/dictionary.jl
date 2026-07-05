# ============================================================
# Dictionary support
#   Reference: RFC 8878 §5
# ============================================================

"""
    ZstdDict

A parsed Zstandard dictionary (RFC 8878 §5), holding the dictionary ID,
pre-built entropy tables, initial repeat offsets, and the content prefix that
frame matches may reference.

Construct one from raw dictionary bytes with `Base.parse(ZstdDict, bytes)`,
then pass it via the `dict` keyword of [`inflate_zstd`](@ref) or
[`InflateZstdStream`](@ref).
"""
struct ZstdDict
    id      ::UInt32
    huffman ::Union{HuffmanTable, Nothing}
    of_tab  ::Union{FSEDistTable, Nothing}
    ml_tab  ::Union{FSEDistTable, Nothing}
    ll_tab  ::Union{FSEDistTable, Nothing}
    rep     ::NTuple{3, Int}
    content ::Vector{UInt8}
end

"""
    Base.parse(ZstdDict, raw::Vector{UInt8}; raw_content::Bool=false) -> ZstdDict

Parse a Zstandard dictionary (RFC 8878 §5).

When `raw_content=false` (the default), the magic number `0xEC30A437` is
checked: if present the dictionary is parsed as a structured dictionary
(with entropy tables and repeat offsets); if absent it is treated as a raw
content dictionary.

When `raw_content=true`, the bytes are read as only raw content.
"""
function Base.parse(::Type{ZstdDict}, data::Vector{UInt8}; raw_content::Bool = false)
    length(data) ≥ 8 ||
        throw(ArgumentError("zstd: dictionary too short; must be at least 8 bytes"))

    (raw_content || _le32(data, 1) != ZSTD_DICT_MAGIC) &&
        return ZstdDict(UInt32(0), nothing, nothing, nothing, nothing, INIT_REPEAT_OFFSETS, data)

    dict_id = _le32(data, 5)
    pos = 9

    # 1. Huffman table for literals
    ht, hdr_len = read_huffman_description(@view data[9:end])
    pos += hdr_len

    # 2.–4. FSE tables for offsets, match lengths, and literals lengths.
    # Validate accuracy logs and symbol counts exactly like block-supplied
    # tables (read_distribution_table!): sequence decoding indexes the
    # baseline/extra-bits tables with these symbols under @inbounds, so an
    # unvalidated dictionary table would allow out-of-bounds reads.
    br = ForwardBitReader(@view data[pos:end])
    of_al, of_dist = read_fse_dist!(br, MAX_OFFSET_CODE)
    of_al ≤ 8 && length(of_dist) ≤ MAX_OFFSET_CODE + 1 ||
        throw(ArgumentError("zstd: invalid offset code table in dictionary"))
    of_tab = build_fse_table(of_dist, of_al)

    ml_al, ml_dist = read_fse_dist!(br, MAX_MATCH_LENGTH)
    ml_al ≤ 9 && length(ml_dist) ≤ MAX_MATCH_LENGTH + 1 ||
        throw(ArgumentError("zstd: invalid match length table in dictionary"))
    ml_tab = build_fse_table(ml_dist, ml_al)

    ll_al, ll_dist = read_fse_dist!(br, MAX_LITERALS_LENGTH)
    ll_al ≤ 9 && length(ll_dist) ≤ MAX_LITERALS_LENGTH + 1 ||
        throw(ArgumentError("zstd: invalid literals length table in dictionary"))
    ll_tab = build_fse_table(ll_dist, ll_al)

    pos = pos + byte_pos(br) - 1
    length(data) ≥ pos + 11 ||
        throw(ArgumentError("zstd: dictionary truncated (repeat offsets)"))

    # 5. Repeat offsets
    repeat_offsets = Int.(only(reinterpret(NTuple{3, Int32}, @view data[pos:pos+11])))
    pos += 12

    all(>(0), repeat_offsets) ||
        throw(ArgumentError("zstd: invalid repeat offsets in dictionary: $repeat_offsets"))

    content = data[pos:end]
    return ZstdDict(dict_id, ht, of_tab, ml_tab, ll_tab, repeat_offsets, content)
end
