# ============================================================
# Section 7: Dictionary support
#   Reference: RFC 8878 §5
# ============================================================

struct ZstdDict
    id      ::UInt32
    huffman ::Union{HuffmanTable, Nothing}
    of_tab  ::Union{FSETable, Nothing}
    ml_tab  ::Union{FSETable, Nothing}
    ll_tab  ::Union{FSETable, Nothing}
    rep     ::NTuple{3,Int}
    content ::Vector{UInt8}
end

"""
    Base.parse(ZstdDict, raw::Vector{UInt8}; raw_content::Bool=false) -> ZstdDict

Parse a Zstandard dictionary (RFC 8878 §5).

When `raw_content=false` (the default), the magic number `0xEC30A437` is
checked: if present the dictionary is parsed as a structured dictionary
(with entropy tables and repeat offsets); if absent it is treated as a raw
content dictionary.

When `raw_content=true`, the bytes are always treated as a raw content
dictionary regardless of their contents — no magic check is performed.
"""
function Base.parse(::Type{ZstdDict}, raw::Vector{UInt8}; raw_content::Bool=false)
    length(raw) ≥ 1 || throw(ArgumentError("zstd: dictionary too short"))
    if raw_content
        return ZstdDict(UInt32(0), nothing, nothing, nothing, nothing, INIT_REPEAT_OFFSETS, raw)
    end
    length(raw) ≥ 8 || throw(ArgumentError("zstd: dictionary too short"))
    magic = UInt32(raw[1]) | (UInt32(raw[2]) << 8) |
            (UInt32(raw[3]) << 16) | (UInt32(raw[4]) << 24)
    if magic != ZSTD_DICT_MAGIC
        # Auto mode: no magic → raw content dictionary
        return ZstdDict(UInt32(0), nothing, nothing, nothing, nothing, INIT_REPEAT_OFFSETS, raw)
    end

    # Structured dictionary
    dict_id = UInt32(raw[5]) | (UInt32(raw[6]) << 8) |
              (UInt32(raw[7]) << 16) | (UInt32(raw[8]) << 24)
    pos = 9

    # 1. Huffman table for literals
    ht, hdr_len = read_huffman_description(raw, pos)
    pos += hdr_len

    # 2. FSE table for offsets
    br = ForwardBitReader(@view raw[pos:end])
    of_al, of_dist = read_fse_dist!(br, MAX_OFFSET_CODE)
    of_tab = build_fse_table(of_dist, of_al)

    # 3. FSE table for match lengths
    ml_al, ml_dist = read_fse_dist!(br, MAX_MATCH_LENGTH)
    ml_tab = build_fse_table(ml_dist, ml_al)

    # 4. FSE table for literals lengths
    ll_al, ll_dist = read_fse_dist!(br, MAX_LITERALS_LENGTH)
    ll_tab = build_fse_table(ll_dist, ll_al)

    pos = byte_pos(br)

    # 5. Three repeat offsets, 4 bytes LE each
    length(raw) ≥ pos + 11 || throw(ArgumentError("zstd: dictionary truncated (repeat offsets)"))
    rep1 = Int(_le32(raw, pos));     pos += 4
    rep2 = Int(_le32(raw, pos));     pos += 4
    rep3 = Int(_le32(raw, pos));     pos += 4

    content = raw[pos:end]
    return ZstdDict(dict_id, ht, of_tab, ml_tab, ll_tab, (rep1, rep2, rep3), content)
end
