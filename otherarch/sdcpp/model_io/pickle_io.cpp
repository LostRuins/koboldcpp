#include "pickle_io.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "binary_io.h"
#include "util.h"

// $ python -m pickletools sd-v1-4/archive/data.pkl | head -n 100
//     0: \x80 PROTO      2
//     2: }    EMPTY_DICT
//     3: q    BINPUT     0
//     5: (    MARK
//     6: X        BINUNICODE 'epoch'
//    16: q        BINPUT     1
//    18: K        BININT1    6
//    20: X        BINUNICODE 'global_step'
//    36: q        BINPUT     2
//    38: J        BININT     470000
//    43: X        BINUNICODE 'pytorch-lightning_version'
//    73: q        BINPUT     3
//    75: X        BINUNICODE '1.4.2'
//    85: q        BINPUT     4
//    87: X        BINUNICODE 'state_dict'
//   102: q        BINPUT     5
//   104: }        EMPTY_DICT
//   105: q        BINPUT     6
//   107: (        MARK
//   108: X            BINUNICODE 'betas'
//   118: q            BINPUT     7
//   120: c            GLOBAL     'torch._utils _rebuild_tensor_v2'
//   153: q            BINPUT     8
//   155: (            MARK
//   156: (                MARK
//   157: X                    BINUNICODE 'storage'
//   169: q                    BINPUT     9
//   171: c                    GLOBAL     'torch FloatStorage'
//   191: q                    BINPUT     10
//   193: X                    BINUNICODE '0'
//   199: q                    BINPUT     11
//   201: X                    BINUNICODE 'cpu'
//   209: q                    BINPUT     12
//   211: M                    BININT2    1000
//   214: t                    TUPLE      (MARK at 156)
//   215: q                BINPUT     13
//   217: Q                BINPERSID
//   218: K                BININT1    0
//   220: M                BININT2    1000
//  ...............................
//  3201: q            BINPUT     250
//  3203: R            REDUCE
//  3204: q            BINPUT     251
//  3206: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.weight'
//  3264: q            BINPUT     252
//  3266: h            BINGET     8
//  3268: (            MARK
//  3269: (                MARK
//  3270: h                    BINGET     9
//  3272: h                    BINGET     10
//  3274: X                    BINUNICODE '30'
//  3281: q                    BINPUT     253
//  3283: h                    BINGET     12
//  3285: J                    BININT     102400
//  3290: t                    TUPLE      (MARK at 3269)
//  3291: q                BINPUT     254
//  3293: Q                BINPERSID
//  3294: K                BININT1    0
//  3296: (                MARK
//  3297: M                    BININT2    320
//  3300: M                    BININT2    320
//  3303: K                    BININT1    1
//  3305: K                    BININT1    1
//  3307: t                    TUPLE      (MARK at 3296)
//  3308: q                BINPUT     255
//  3310: (                MARK
//  3311: M                    BININT2    320
//  3314: K                    BININT1    1
//  3316: K                    BININT1    1
//  3318: K                    BININT1    1
//  3320: t                    TUPLE      (MARK at 3310)
//  3321: r                LONG_BINPUT 256
//  3326: \x89             NEWFALSE
//  3327: h                BINGET     16
//  3329: )                EMPTY_TUPLE
//  3330: R                REDUCE
//  3331: r                LONG_BINPUT 257
//  3336: t                TUPLE      (MARK at 3268)
//  3337: r            LONG_BINPUT 258
//  3342: R            REDUCE
//  3343: r            LONG_BINPUT 259
//  3348: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.bias'
//  3404: r            LONG_BINPUT 260
//  3409: h            BINGET     8
//  3411: (            MARK
//  3412: (                MARK
//  3413: h                    BINGET     9
//  3415: h                    BINGET     10
//  3417: X                    BINUNICODE '31'
// https://github.com/python/cpython/blob/3.7/Lib/pickletools.py#L1048
// https://github.com/python/cpython/blob/main/Lib/pickle.py#L105

using model_io::find_char;
using model_io::read_int;
using model_io::read_short;
using model_io::read_u64;

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool skip_pickle_object(const uint8_t* buffer, size_t buffer_size, size_t* object_size) {
    const uint8_t* p   = buffer;
    const uint8_t* end = buffer + buffer_size;

    while (p < end) {
        uint8_t opcode = *p++;
        switch (opcode) {
            case '.':  // STOP             = b'.'   # every pickle ends with STOP
                *object_size = (size_t)(p - buffer);
                return true;
            case 0x80:  // PROTO            = b'\x80'  # protocol version indicator
            case 'K':   // BININT1          = b'K'   # push 1-byte unsigned int
            case 'h':   // BINGET           = b'h'   # read memo index, 1-byte arg
            case 'q':   // BINPUT           = b'q'   # write memo index, 1-byte arg
            case 'C':   // SHORT_BINBYTES   = b'C'   # push bytes; length < 256
            case 0x82:  // EXT1             = b'\x82'  # extension code, 1-byte arg
                p += 1;
                break;
            case 'M':   // BININT2          = b'M'   # push 2-byte unsigned int
            case 0x83:  // EXT2             = b'\x83'  # extension code, 2-byte arg
                p += 2;
                break;
            case 'J':   // BININT           = b'J'   # push 4-byte signed int
            case 'j':   // LONG_BINGET      = b'j'   # read memo index, 4-byte arg
            case 'r':   // LONG_BINPUT      = b'r'   # write memo index, 4-byte arg
            case 0x84:  // EXT4             = b'\x84'  # extension code, 4-byte arg
                p += 4;
                break;
            case 'I':    // INT              = b'I'   # push decimal integer line
            case 'L':    // LONG             = b'L'   # push decimal long integer line
            case 'F':    // FLOAT            = b'F'   # push decimal float line
            case 'S':    // STRING           = b'S'   # push quoted string line
            case 'V': {  // UNICODE          = b'V'   # push raw-unicode string line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
            } break;
            case 'G':  // BINFLOAT         = b'G'   # push 8-byte binary float
                p += 8;
                break;
            case 0x8A:  // LONG1            = b'\x8a'  # push long integer; 1-byte length
                if (p >= end) {
                    return false;
                }
                p += 1 + p[0];
                break;
            case 0x8B: {  // LONG4            = b'\x8b'  # push long integer; 4-byte length
                if (p + 4 > end) {
                    return false;
                }
                uint32_t n = (uint32_t)read_int(p);
                p += 4 + n;
            } break;
            case 'B': {  // BINBYTES         = b'B'   # push bytes; 4-byte length
                if (p + 4 > end) {
                    return false;
                }
                uint32_t n = (uint32_t)read_int(p);
                p += 4 + n;
            } break;
            case 'T':    // BINSTRING        = b'T'   # push string; 4-byte length
            case 'X': {  // BINUNICODE       = b'X'   # push UTF-8 string; 4-byte length
                if (p + 4 > end) {
                    return false;
                }
                uint32_t n = (uint32_t)read_int(p);
                p += 4 + n;
            } break;
            case 0x8D:    // BINUNICODE8      = b'\x8d'  # push UTF-8 string; 8-byte length
            case 0x8E:    // BINBYTES8        = b'\x8e'  # push bytes; 8-byte length
            case 0x96: {  // BYTEARRAY8       = b'\x96'  # push bytearray; 8-byte length
                if (p + 8 > end) {
                    return false;
                }
                uint64_t n = read_u64(p);
                p += 8;
                if (n > (uint64_t)(end - p)) {
                    return false;
                }
                p += n;
            } break;
            case 'U':   // SHORT_BINSTRING  = b'U'   # push string; length < 256
            case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push UTF-8 string; length < 256
                if (p >= end) {
                    return false;
                }
                p += 1 + p[0];
                break;
            case 'P': {  // PERSID           = b'P'   # persistent id, newline-terminated
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
            } break;
            case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                p += 8;
                break;
            case 'c': {  // GLOBAL           = b'c'   # push module/name global reference
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
                len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
            } break;
            case '}':   // EMPTY_DICT       = b'}'   # push empty dict
            case ']':   // EMPTY_LIST       = b']'   # push empty list
            case '(':   // MARK             = b'('   # push markobject
            case 't':   // TUPLE            = b't'   # build tuple from mark
            case 0x85:  // TUPLE1           = b'\x85'  # build 1-tuple from stack
            case 0x86:  // TUPLE2           = b'\x86'  # build 2-tuple from stack
            case 0x87:  // TUPLE3           = b'\x87'  # build 3-tuple from stack
            case ')':   // EMPTY_TUPLE      = b')'   # push empty tuple
            case 'l':   // LIST             = b'l'   # build list from mark
            case 'Q':   // BINPERSID        = b'Q'   # persistent id from stack
            case 0x94:  // MEMOIZE          = b'\x94'  # store top of stack in memo
            case 0x88:  // NEWTRUE          = b'\x88'  # push True
            case 0x89:  // NEWFALSE         = b'\x89'  # push False
            case 'R':   // REDUCE           = b'R'   # apply callable to args
            case 'u':   // SETITEMS         = b'u'   # add mark-delimited items to dict
            case 's':   // SETITEM          = b's'   # add key/value to dict
            case 'e':   // APPENDS          = b'e'   # extend list with mark-delimited items
            case 'a':   // APPEND           = b'a'   # append item to list
            case 'b':   // BUILD            = b'b'   # build object state
            case 0x81:  // NEWOBJ           = b'\x81'  # build object via __new__
            case 0x8F:  // EMPTY_SET        = b'\x8f'  # push empty set
            case 0x90:  // ADDITEMS         = b'\x90'  # add mark-delimited items to set
            case 0x91:  // FROZENSET        = b'\x91'  # build frozenset from mark
            case 0x92:  // NEWOBJ_EX        = b'\x92'  # build object with kwargs
            case 0x93:  // STACK_GLOBAL     = b'\x93'  # build global from module/name strings
            case 0x97:  // NEXT_BUFFER      = b'\x97'  # out-of-band buffer marker
            case 0x98:  // READONLY_BUFFER  = b'\x98'  # mark buffer readonly
            case 'N':   // NONE             = b'N'   # push None
            case '0':   // POP              = b'0'   # discard top stack item
            case '1':   // POP_MARK         = b'1'   # discard stack through topmost mark
            case '2':   // DUP              = b'2'   # duplicate top stack item
            case 'o':   // OBJ              = b'o'   # build class instance from mark
                break;
            case 'i': {  // INST             = b'i'   # build class instance from module/name
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
                len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
            } break;
            default:
                return false;
        }
        if (p > end) {
            return false;
        }
    }

    return false;
}

bool pickle_object_is_torch_magic_number(const uint8_t* buffer, size_t buffer_size) {
    static const uint8_t torch_magic_bytes[] = {0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19};

    if (buffer_size < 5 || buffer[0] != 0x80) {
        return false;
    }

    size_t pos = 2;
    if (pos >= buffer_size) {
        return false;
    }

    uint8_t opcode = buffer[pos++];
    if (opcode != 0x8A || pos >= buffer_size) {
        return false;
    }

    uint8_t len = buffer[pos++];
    if (len != sizeof(torch_magic_bytes) || pos + len >= buffer_size) {
        return false;
    }

    if (memcmp(buffer + pos, torch_magic_bytes, sizeof(torch_magic_bytes)) != 0) {
        return false;
    }
    pos += len;

    return pos < buffer_size && buffer[pos] == '.';
}

bool parse_pickle_uint32_object(const uint8_t* buffer, size_t buffer_size, uint32_t* value) {
    if (buffer_size < 4 || buffer[0] != 0x80) {
        return false;
    }

    size_t pos = 2;
    if (pos >= buffer_size) {
        return false;
    }

    uint8_t opcode = buffer[pos++];
    switch (opcode) {
        case 'K':  // BININT1          = b'K'   # push 1-byte unsigned int
            if (pos + 1 >= buffer_size) {
                return false;
            }
            *value = buffer[pos];
            pos += 1;
            break;
        case 'M':  // BININT2          = b'M'   # push 2-byte unsigned int
            if (pos + 2 >= buffer_size) {
                return false;
            }
            *value = read_short(buffer + pos);
            pos += 2;
            break;
        case 'J':  // BININT           = b'J'   # push 4-byte signed int
            if (pos + 4 >= buffer_size) {
                return false;
            }
            *value = (uint32_t)read_int(buffer + pos);
            pos += 4;
            break;
        default:
            return false;
    }

    return pos < buffer_size && buffer[pos] == '.';
}

struct PickleStorageInfo {
    std::string key;
    ggml_type type              = GGML_TYPE_COUNT;
    bool is_f64                 = false;
    bool is_i64                 = false;
    uint64_t raw_element_nbytes = 0;
    uint64_t nbytes             = 0;
};

struct PickleTensorInfo {
    TensorStorage tensor_storage;
    int stride_n_dims = 0;
    int64_t stride[SD_MAX_DIMS]{1, 1, 1, 1, 1};
};

struct PickleValue {
    enum Kind {
        MARK,
        NONE,
        BOOL,
        INT,
        STRING,
        GLOBAL,
        TUPLE,
        LIST,
        DICT,
        ORDERED_DICT,
        STORAGE,
        TENSOR,
    };

    Kind kind         = NONE;
    int64_t int_value = 0;
    bool bool_value   = false;
    std::string str_value;
    std::vector<PickleValue> items;
    std::vector<std::pair<PickleValue, PickleValue>> dict_items;
    PickleStorageInfo storage;
    PickleTensorInfo tensor;
};

static PickleValue make_mark_value() {
    PickleValue value;
    value.kind = PickleValue::MARK;
    return value;
}

static PickleValue make_none_value() {
    PickleValue value;
    value.kind = PickleValue::NONE;
    return value;
}

static PickleValue make_bool_value(bool b) {
    PickleValue value;
    value.kind       = PickleValue::BOOL;
    value.bool_value = b;
    return value;
}

static PickleValue make_int_value(int64_t x) {
    PickleValue value;
    value.kind      = PickleValue::INT;
    value.int_value = x;
    return value;
}

static PickleValue make_string_value(const std::string& s) {
    PickleValue value;
    value.kind      = PickleValue::STRING;
    value.str_value = s;
    return value;
}

static PickleValue make_global_value(const std::string& s) {
    PickleValue value;
    value.kind      = PickleValue::GLOBAL;
    value.str_value = s;
    return value;
}

static PickleValue make_tuple_value(std::vector<PickleValue> items) {
    PickleValue value;
    value.kind  = PickleValue::TUPLE;
    value.items = std::move(items);
    return value;
}

static PickleValue make_list_value() {
    PickleValue value;
    value.kind = PickleValue::LIST;
    return value;
}

static PickleValue make_dict_value(bool ordered) {
    PickleValue value;
    value.kind = ordered ? PickleValue::ORDERED_DICT : PickleValue::DICT;
    return value;
}

static PickleValue make_storage_value(const PickleStorageInfo& storage) {
    PickleValue value;
    value.kind    = PickleValue::STORAGE;
    value.storage = storage;
    return value;
}

static PickleValue make_tensor_value(const PickleTensorInfo& tensor) {
    PickleValue value;
    value.kind   = PickleValue::TENSOR;
    value.tensor = tensor;
    return value;
}

static std::string pickle_value_to_string(const PickleValue& value) {
    if (value.kind == PickleValue::STRING) {
        return value.str_value;
    }
    if (value.kind == PickleValue::INT) {
        return std::to_string(value.int_value);
    }
    return "";
}

static bool parse_storage_type(const std::string& global_name, PickleStorageInfo* storage) {
    if (global_name == "torch.FloatStorage") {
        storage->type               = GGML_TYPE_F32;
        storage->raw_element_nbytes = 4;
        return true;
    }
    if (global_name == "torch.DoubleStorage") {
        storage->type               = GGML_TYPE_F32;
        storage->is_f64             = true;
        storage->raw_element_nbytes = 8;
        return true;
    }
    if (global_name == "torch.HalfStorage") {
        storage->type               = GGML_TYPE_F16;
        storage->raw_element_nbytes = 2;
        return true;
    }
    if (global_name == "torch.BFloat16Storage") {
        storage->type               = GGML_TYPE_BF16;
        storage->raw_element_nbytes = 2;
        return true;
    }
    if (global_name == "torch.IntStorage") {
        storage->type               = GGML_TYPE_I32;
        storage->raw_element_nbytes = 4;
        return true;
    }
    if (global_name == "torch.LongStorage") {
        storage->type               = GGML_TYPE_I32;
        storage->is_i64             = true;
        storage->raw_element_nbytes = 8;
        return true;
    }
    return false;
}

static bool tensor_is_contiguous(const PickleTensorInfo& tensor) {
    if (tensor.tensor_storage.nelements() == 0) {
        return true;
    }
    if (tensor.stride_n_dims != tensor.tensor_storage.n_dims) {
        return false;
    }

    int64_t expected_stride = 1;
    for (int i = tensor.tensor_storage.n_dims - 1; i >= 0; --i) {
        if (tensor.stride[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor.tensor_storage.ne[i];
    }
    return true;
}

static void collect_tensors_from_pickle_value(const PickleValue& value,
                                              std::vector<TensorStorage>& tensor_storages) {
    if (value.kind != PickleValue::DICT && value.kind != PickleValue::ORDERED_DICT) {
        return;
    }

    for (const auto& item : value.dict_items) {
        if (item.first.kind == PickleValue::STRING && item.second.kind == PickleValue::TENSOR) {
            TensorStorage tensor_storage = item.second.tensor.tensor_storage;
            tensor_storage.name          = item.first.str_value;
            tensor_storage.reverse_ne();
            tensor_storages.push_back(tensor_storage);
        } else if (item.second.kind == PickleValue::DICT || item.second.kind == PickleValue::ORDERED_DICT) {
            collect_tensors_from_pickle_value(item.second, tensor_storages);
        }
    }
}

bool parse_torch_state_dict_pickle(const uint8_t* buffer,
                                   size_t buffer_size,
                                   std::vector<TensorStorage>& tensor_storages,
                                   std::unordered_map<std::string, uint64_t>& storage_nbytes,
                                   std::string* error) {
    if (buffer_size < 2 || buffer[0] != 0x80 || buffer[1] < 2 || buffer[1] > 5) {
        set_error(error, "unsupported torch pickle protocol");
        return false;
    }

    const uint8_t* p   = buffer + 2;
    const uint8_t* end = buffer + buffer_size;
    std::vector<PickleValue> stack;
    std::unordered_map<int32_t, PickleValue> memo;

    while (p < end) {
        uint8_t opcode = *p++;
        switch (opcode) {
            case '.': {  // STOP             = b'.'   # every pickle ends with STOP
                if (stack.empty()) {
                    set_error(error, "empty torch pickle stack");
                    return false;
                }
                size_t old_tensor_count = tensor_storages.size();
                collect_tensors_from_pickle_value(stack.back(), tensor_storages);
                if (tensor_storages.size() == old_tensor_count) {
                    set_error(error, "torch pickle does not contain a supported state_dict");
                    return false;
                }
                return true;
            }
            case '}':  // EMPTY_DICT       = b'}'   # push empty dict
                stack.push_back(make_dict_value(false));
                break;
            case ']':  // EMPTY_LIST       = b']'   # push empty list
                stack.push_back(make_list_value());
                break;
            case 'l': {  // LIST             = b'l'   # build list from mark
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx < 0) {
                    set_error(error, "torch pickle list without mark");
                    return false;
                }
                std::vector<PickleValue> items(stack.begin() + mark_idx + 1, stack.end());
                stack.erase(stack.begin() + mark_idx, stack.end());
                PickleValue list_value = make_list_value();
                list_value.items       = std::move(items);
                stack.push_back(std::move(list_value));
            } break;
            case '(':  // MARK             = b'('   # push markobject
                stack.push_back(make_mark_value());
                break;
            case ')':  // EMPTY_TUPLE      = b')'   # push empty tuple
                stack.push_back(make_tuple_value({}));
                break;
            case 'N':  // NONE             = b'N'   # push None
                stack.push_back(make_none_value());
                break;
            case 0x88:  // NEWTRUE          = b'\x88'  # push True
                stack.push_back(make_bool_value(true));
                break;
            case 0x89:  // NEWFALSE         = b'\x89'  # push False
                stack.push_back(make_bool_value(false));
                break;
            case 'K':  // BININT1          = b'K'   # push 1-byte unsigned int
                if (p >= end) {
                    return false;
                }
                stack.push_back(make_int_value(*p++));
                break;
            case 'M':  // BININT2          = b'M'   # push 2-byte unsigned int
                if (p + 2 > end) {
                    return false;
                }
                stack.push_back(make_int_value(read_short(p)));
                p += 2;
                break;
            case 'J':  // BININT           = b'J'   # push 4-byte signed int
                if (p + 4 > end) {
                    return false;
                }
                stack.push_back(make_int_value(read_int(p)));
                p += 4;
                break;
            case 'I': {  // INT              = b'I'   # push decimal integer line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                std::string s((const char*)p, len);
                p += len + 1;
                if (s == "01") {
                    stack.push_back(make_bool_value(true));
                } else if (s == "00") {
                    stack.push_back(make_bool_value(false));
                } else {
                    stack.push_back(make_int_value(std::strtoll(s.c_str(), nullptr, 10)));
                }
            } break;
            case 'L': {  // LONG             = b'L'   # push decimal long integer line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                std::string s((const char*)p, len);
                p += len + 1;
                if (!s.empty() && s.back() == 'L') {
                    s.pop_back();
                }
                stack.push_back(make_int_value(std::strtoll(s.c_str(), nullptr, 10)));
            } break;
            case 'F': {  // FLOAT            = b'F'   # push decimal float line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                p += len + 1;
                stack.push_back(make_none_value());
            } break;
            case 'G':  // BINFLOAT         = b'G'   # push 8-byte binary float
                if (p + 8 > end) {
                    return false;
                }
                p += 8;
                stack.push_back(make_none_value());
                break;
            case 0x8A: {  // LONG1            = b'\x8a'  # push long integer; 1-byte length
                if (p >= end) {
                    return false;
                }
                uint8_t n = *p++;
                if (p + n > end || n > 8) {
                    return false;
                }
                int64_t value = 0;
                for (uint8_t i = 0; i < n; ++i) {
                    value |= (int64_t)p[i] << (i * 8);
                }
                p += n;
                stack.push_back(make_int_value(value));
            } break;
            case 'C': {  // SHORT_BINBYTES   = b'C'   # push bytes; length < 256
                if (p >= end) {
                    return false;
                }
                uint8_t len = *p++;
                if (p + len > end) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, len)));
                p += len;
            } break;
            case 'B': {  // BINBYTES         = b'B'   # push bytes; 4-byte length
                if (p + 4 > end) {
                    return false;
                }
                int32_t len = read_int(p);
                p += 4;
                if (len < 0 || p + len > end) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, len)));
                p += len;
            } break;
            case 'T':    // BINSTRING        = b'T'   # push string; 4-byte length
            case 'X': {  // BINUNICODE       = b'X'   # push UTF-8 string; 4-byte length
                if (p + 4 > end) {
                    return false;
                }
                int32_t len = read_int(p);
                p += 4;
                if (len < 0 || p + len > end) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, len)));
                p += len;
            } break;
            case 0x8D:    // BINUNICODE8      = b'\x8d'  # push UTF-8 string; 8-byte length
            case 0x8E:    // BINBYTES8        = b'\x8e'  # push bytes; 8-byte length
            case 0x96: {  // BYTEARRAY8       = b'\x96'  # push bytearray; 8-byte length
                if (p + 8 > end) {
                    return false;
                }
                uint64_t len = read_u64(p);
                p += 8;
                if (len > (uint64_t)(end - p)) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, (size_t)len)));
                p += len;
            } break;
            case 'U':     // SHORT_BINSTRING  = b'U'   # push string; length < 256
            case 0x8C: {  // SHORT_BINUNICODE = b'\x8c'  # push UTF-8 string; length < 256
                if (p >= end) {
                    return false;
                }
                uint8_t len = *p++;
                if (p + len > end) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, len)));
                p += len;
            } break;
            case 'S': {  // STRING           = b'S'   # push quoted string line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                std::string s((const char*)p, len);
                p += len + 1;
                if (s.size() >= 2 && (s[0] == '\'' || s[0] == '"') && s.back() == s[0]) {
                    s = s.substr(1, s.size() - 2);
                }
                stack.push_back(make_string_value(s));
            } break;
            case 'V': {  // UNICODE          = b'V'   # push raw-unicode string line
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                stack.push_back(make_string_value(std::string((const char*)p, len)));
                p += len + 1;
            } break;
            case 'c': {  // GLOBAL           = b'c'   # push module/name global reference
                int len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                std::string module((const char*)p, len);
                p += len + 1;
                len = find_char(p, (int)(end - p), '\n');
                if (len < 0) {
                    return false;
                }
                std::string name((const char*)p, len);
                p += len + 1;
                stack.push_back(make_global_value(module + "." + name));
            } break;
            case 0x93: {  // STACK_GLOBAL     = b'\x93'  # build global from module/name strings
                if (stack.size() < 2 || stack[stack.size() - 2].kind != PickleValue::STRING ||
                    stack.back().kind != PickleValue::STRING) {
                    return false;
                }
                std::string name = stack.back().str_value;
                stack.pop_back();
                std::string module = stack.back().str_value;
                stack.pop_back();
                stack.push_back(make_global_value(module + "." + name));
            } break;
            case 'h':  // BINGET           = b'h'   # read memo index, 1-byte arg
                if (p >= end || !memo.count(*p)) {
                    return false;
                }
                stack.push_back(memo[*p++]);
                break;
            case 'j': {  // LONG_BINGET      = b'j'   # read memo index, 4-byte arg
                if (p + 4 > end) {
                    return false;
                }
                int32_t memo_idx = read_int(p);
                if (!memo.count(memo_idx)) {
                    return false;
                }
                stack.push_back(memo[memo_idx]);
                p += 4;
            } break;
            case 'q':  // BINPUT           = b'q'   # write memo index, 1-byte arg
                if (p >= end || stack.empty()) {
                    return false;
                }
                memo[*p++] = stack.back();
                break;
            case 'r':  // LONG_BINPUT      = b'r'   # write memo index, 4-byte arg
                if (p + 4 > end || stack.empty()) {
                    return false;
                }
                memo[read_int(p)] = stack.back();
                p += 4;
                break;
            case 0x94:  // MEMOIZE          = b'\x94'  # store top of stack in memo
                if (stack.empty()) {
                    return false;
                }
                memo[(int32_t)memo.size()] = stack.back();
                break;
            case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                if (p + 8 > end) {
                    return false;
                }
                p += 8;
                break;
            case '0':  // POP              = b'0'   # discard top stack item
                if (stack.empty()) {
                    return false;
                }
                stack.pop_back();
                break;
            case '1': {  // POP_MARK         = b'1'   # discard stack through topmost mark
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx < 0) {
                    return false;
                }
                stack.erase(stack.begin() + mark_idx, stack.end());
            } break;
            case '2':  // DUP              = b'2'   # duplicate top stack item
                if (stack.empty()) {
                    return false;
                }
                stack.push_back(stack.back());
                break;
            case 0x8F:  // EMPTY_SET        = b'\x8f'  # push empty set
                stack.push_back(make_list_value());
                break;
            case 0x90: {  // ADDITEMS         = b'\x90'  # add mark-delimited items to set
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx <= 0 || stack[mark_idx - 1].kind != PickleValue::LIST) {
                    return false;
                }
                PickleValue& set_value = stack[mark_idx - 1];
                set_value.items.insert(set_value.items.end(), stack.begin() + mark_idx + 1, stack.end());
                stack.erase(stack.begin() + mark_idx, stack.end());
            } break;
            case 0x91: {  // FROZENSET        = b'\x91'  # build frozenset from mark
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx < 0) {
                    return false;
                }
                PickleValue set_value = make_list_value();
                set_value.items.insert(set_value.items.end(), stack.begin() + mark_idx + 1, stack.end());
                stack.erase(stack.begin() + mark_idx, stack.end());
                stack.push_back(std::move(set_value));
            } break;
            case 0x85:    // TUPLE1           = b'\x85'  # build 1-tuple from stack
            case 0x86:    // TUPLE2           = b'\x86'  # build 2-tuple from stack
            case 0x87: {  // TUPLE3           = b'\x87'  # build 3-tuple from stack
                int tuple_size = opcode == 0x85 ? 1 : (opcode == 0x86 ? 2 : 3);
                if ((int)stack.size() < tuple_size) {
                    return false;
                }
                std::vector<PickleValue> items(stack.end() - tuple_size, stack.end());
                stack.erase(stack.end() - tuple_size, stack.end());
                stack.push_back(make_tuple_value(std::move(items)));
            } break;
            case 't': {  // TUPLE            = b't'   # build tuple from mark
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx < 0) {
                    return false;
                }
                std::vector<PickleValue> items(stack.begin() + mark_idx + 1, stack.end());
                stack.erase(stack.begin() + mark_idx, stack.end());
                stack.push_back(make_tuple_value(std::move(items)));
            } break;
            case 'Q': {  // BINPERSID        = b'Q'   # persistent id from stack
                if (stack.empty()) {
                    return false;
                }
                PickleValue pid = stack.back();
                stack.pop_back();
                if (pid.kind != PickleValue::TUPLE || pid.items.size() < 5 || pid.items[0].kind != PickleValue::STRING ||
                    pid.items[1].kind != PickleValue::GLOBAL || pid.items[4].kind != PickleValue::INT ||
                    pid.items[0].str_value != "storage") {
                    return false;
                }

                PickleStorageInfo storage;
                storage.key = pickle_value_to_string(pid.items[2]);
                if (storage.key.empty() || !parse_storage_type(pid.items[1].str_value, &storage)) {
                    return false;
                }
                storage.nbytes              = (uint64_t)pid.items[4].int_value * storage.raw_element_nbytes;
                storage_nbytes[storage.key] = storage.nbytes;
                stack.push_back(make_storage_value(storage));
            } break;
            case 'R': {  // REDUCE           = b'R'   # apply callable to args
                if (stack.size() < 2) {
                    return false;
                }
                PickleValue args = stack.back();
                stack.pop_back();
                PickleValue callable = stack.back();
                stack.pop_back();
                if (callable.kind != PickleValue::GLOBAL || args.kind != PickleValue::TUPLE) {
                    stack.push_back(make_none_value());
                    break;
                }

                if (callable.str_value == "collections.OrderedDict" && args.items.empty()) {
                    stack.push_back(make_dict_value(true));
                    break;
                }

                if ((callable.str_value == "torch._utils._rebuild_tensor_v2" || callable.str_value == "torch._utils._rebuild_tensor") &&
                    args.items.size() >= 4 && args.items[0].kind == PickleValue::STORAGE &&
                    args.items[1].kind == PickleValue::INT && args.items[2].kind == PickleValue::TUPLE &&
                    args.items[3].kind == PickleValue::TUPLE) {
                    PickleTensorInfo tensor;
                    tensor.tensor_storage.type        = args.items[0].storage.type;
                    tensor.tensor_storage.is_f64      = args.items[0].storage.is_f64;
                    tensor.tensor_storage.is_i64      = args.items[0].storage.is_i64;
                    tensor.tensor_storage.storage_key = args.items[0].storage.key;
                    tensor.tensor_storage.offset      = (uint64_t)args.items[1].int_value * args.items[0].storage.raw_element_nbytes;

                    for (const auto& item : args.items[2].items) {
                        if (item.kind != PickleValue::INT || tensor.tensor_storage.n_dims >= SD_MAX_DIMS) {
                            return false;
                        }
                        tensor.tensor_storage.ne[tensor.tensor_storage.n_dims++] = item.int_value;
                    }

                    for (const auto& item : args.items[3].items) {
                        if (item.kind != PickleValue::INT || tensor.stride_n_dims >= SD_MAX_DIMS) {
                            return false;
                        }
                        tensor.stride[tensor.stride_n_dims++] = item.int_value;
                    }

                    if (!tensor_is_contiguous(tensor)) {
                        return false;
                    }
                    stack.push_back(make_tensor_value(tensor));
                    break;
                }

                // Non-tensor checkpoint metadata can use REDUCE for arbitrary
                // Python objects. Do not execute it; keep stack shape only.
                stack.push_back(make_none_value());
                break;
            }
            case 'b':  // BUILD            = b'b'   # build object state
                if (stack.size() < 2) {
                    return false;
                }
                stack.pop_back();
                break;
            case 'u': {  // SETITEMS         = b'u'   # add mark-delimited items to dict
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx <= 0) {
                    return false;
                }
                PickleValue& dict = stack[mark_idx - 1];
                if (dict.kind != PickleValue::DICT && dict.kind != PickleValue::ORDERED_DICT) {
                    return false;
                }
                for (int i = mark_idx + 1; i + 1 < (int)stack.size(); i += 2) {
                    dict.dict_items.emplace_back(stack[i], stack[i + 1]);
                }
                stack.erase(stack.begin() + mark_idx, stack.end());
            } break;
            case 's': {  // SETITEM          = b's'   # add key/value to dict
                if (stack.size() < 3) {
                    return false;
                }
                PickleValue value = stack.back();
                stack.pop_back();
                PickleValue key = stack.back();
                stack.pop_back();
                PickleValue& dict = stack.back();
                if (dict.kind != PickleValue::DICT && dict.kind != PickleValue::ORDERED_DICT) {
                    return false;
                }
                dict.dict_items.emplace_back(key, value);
            } break;
            case 'e': {  // APPENDS          = b'e'   # extend list with mark-delimited items
                int mark_idx = (int)stack.size() - 1;
                while (mark_idx >= 0 && stack[mark_idx].kind != PickleValue::MARK) {
                    --mark_idx;
                }
                if (mark_idx <= 0 || stack[mark_idx - 1].kind != PickleValue::LIST) {
                    return false;
                }
                PickleValue& list_value = stack[mark_idx - 1];
                list_value.items.insert(list_value.items.end(), stack.begin() + mark_idx + 1, stack.end());
                stack.erase(stack.begin() + mark_idx, stack.end());
            } break;
            case 'a': {  // APPEND           = b'a'   # append item to list
                if (stack.size() < 2) {
                    return false;
                }
                PickleValue item = stack.back();
                stack.pop_back();
                if (stack.back().kind != PickleValue::LIST) {
                    return false;
                }
                stack.back().items.push_back(item);
            } break;
            default:
                set_error(error,
                          "unsupported torch pickle opcode 0x" + sd_format("%02X", opcode) +
                              " at offset " + std::to_string((p - buffer) - 1));
                return false;
        }
    }

    set_error(error, "unterminated torch state_dict pickle");
    return false;
}
