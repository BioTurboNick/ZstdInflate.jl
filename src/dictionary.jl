# ============================================================
# Dictionary support
#   Reference: RFC 8878 §5
# ============================================================

struct ZstdDict
    id      ::UInt32
    huffman ::Union{HuffmanTable, Nothing}
    of_tab  ::Union{FSETable, Nothing}
    ml_tab  ::Union{FSETable, Nothing}
    ll_tab  ::Union{FSETable, Nothing}
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
    ht, hdr_len = read_huffman_description(data, pos)
    pos += hdr_len

    # 2. FSE table for offsets
    br = ForwardBitReader(@view data[pos:end])
    of_al, of_dist = read_fse_dist!(br, MAX_OFFSET_CODE)
    of_tab = build_fse_table(of_dist, of_al)

    # 3. FSE table for match lengths
    ml_al, ml_dist = read_fse_dist!(br, MAX_MATCH_LENGTH)
    ml_tab = build_fse_table(ml_dist, ml_al)

    # 4. FSE table for literals lengths
    ll_al, ll_dist = read_fse_dist!(br, MAX_LITERALS_LENGTH)
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
