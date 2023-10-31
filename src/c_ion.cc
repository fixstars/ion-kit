#include <exception>

#include <ion/ion.h>
#include <ion/c_ion.h>

#include <HalideBuffer.h>

using namespace ion;

//
// ion_port_t
//
int ion_port_create(ion_port_t *ptr, const char *key, ion_type_t type, int dim)
{
    try {
        *ptr = reinterpret_cast<ion_port_t>(new Port(key, halide_type_t(static_cast<halide_type_code_t>(type.code), type.bits, type.lanes), dim));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_port_destroy(ion_port_t obj)
{
    try {
        delete reinterpret_cast<Port*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

//
// ion_param_t
//
int ion_param_create(ion_param_t *ptr, const char *key, const char *value)
{
    try {
        *ptr = reinterpret_cast<ion_param_t>(new Param(key, value));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_param_destroy(ion_param_t obj)
{
    try {
        delete reinterpret_cast<Param*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

//
// ion_node_t
//
int ion_node_create(ion_node_t *ptr)
{
    try {
        *ptr = reinterpret_cast<ion_node_t>(new Node);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_node_destroy(ion_node_t obj)
{
    try {
        delete reinterpret_cast<Node*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;

}

int ion_node_get_port(ion_node_t obj, const char *key, ion_port_t *port_ptr)
{
    try {
        *port_ptr = reinterpret_cast<ion_port_t>(new Port((*reinterpret_cast<Node*>(obj))[key]));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_node_set_port(ion_node_t obj, ion_port_t *ports_ptr, int ports_num)
{
    try {
        std::vector<Port> ports(ports_num);
        for (int i=0; i<ports_num; ++i) {
            ports[i] = *reinterpret_cast<Port*>(ports_ptr[i]);
        }
        (*reinterpret_cast<Node*>(obj))(ports);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_node_set_param(ion_node_t obj, ion_param_t *params_ptr, int params_num)
{
    try {
        std::vector<Param> params(params_num);
        for (int i=0; i<params_num; ++i) {
            params[i] = *reinterpret_cast<Param*>(params_ptr[i]);
        }
        reinterpret_cast<Node*>(obj)->set_param(params);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

//
// ion_builder_t
//
int ion_builder_create(ion_builder_t *ptr)
{
    try {
        *ptr = reinterpret_cast<ion_builder_t>(new Builder);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_destroy(ion_builder_t obj)
{
    try {
        delete reinterpret_cast<Builder*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_set_target(ion_builder_t obj, const char *target)
{
    try {
        reinterpret_cast<Builder *>(obj)->set_target(Halide::Target(target));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_with_bb_module(ion_builder_t obj, const char *module_name)
{
    try {
        reinterpret_cast<Builder *>(obj)->with_bb_module(module_name);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_add_node(ion_builder_t obj, const char *key, ion_node_t *node_ptr)
{
    try {
        *node_ptr = reinterpret_cast<ion_node_t>(new Node(reinterpret_cast<Builder*>(obj)->add(key)));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_compile(ion_builder_t obj, const char *function_name, ion_builder_compile_option_t option)
{
    try {
        reinterpret_cast<Builder*>(obj)->compile(function_name, Builder::CompileOption{option.output_directory});
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}
int ion_builder_load(ion_builder_t obj, const char *file_name)
{
    try {
        reinterpret_cast<Builder*>(obj)->load(file_name);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_save(ion_builder_t obj, const char *file_name)
{
    try {
        reinterpret_cast<Builder*>(obj)->save(file_name);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_bb_metadata(ion_builder_t obj, char *ptr, int n, int *ret_n)
{
    try {
        auto md = reinterpret_cast<Builder*>(obj)->bb_metadata();
        if (ptr != nullptr) {
            auto copy_size = (std::min)(static_cast<size_t>(n), md.size());
            std::memcpy(ptr, md.c_str(), copy_size);
        } else {
            if (ret_n != nullptr) {
                *ret_n = static_cast<int>(md.size());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_builder_run(ion_builder_t obj, ion_port_map_t pm)
{
    try {
        reinterpret_cast<Builder*>(obj)->run(*reinterpret_cast<PortMap*>(pm));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

template<typename T>
Halide::Buffer<T> *make_buffer(const std::vector<int>& sizes) {
    if (sizes.empty()) {
        auto p = new Halide::Buffer<T>();
        *p = Halide::Buffer<T>::make_scalar();
        return p;
    } else {
        return new Halide::Buffer<T>(sizes);
    }
}

int ion_buffer_create(ion_buffer_t *ptr, ion_type_t type, int *sizes_, int dim)
{
    try {
        std::vector<int> sizes(dim);
        std::memcpy(sizes.data(), sizes_, dim * sizeof(int));
        if (type.lanes != 1) {
            throw std::runtime_error("Unsupported lane number");
        }
        if (type.code == ion_type_int) {
            if (type.bits == 8) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<int8_t>(sizes));
            } else if (type.bits == 16) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<int16_t>(sizes));
            } else if (type.bits == 32) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<int32_t>(sizes));
            } else if (type.bits == 64) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<int64_t>(sizes));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.code == ion_type_uint) {
            if (type.bits == 1) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<bool>(sizes));
            } else if (type.bits == 8) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<uint8_t>(sizes));
            } else if (type.bits == 16) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<uint16_t>(sizes));
            } else if (type.bits == 32) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<uint32_t>(sizes));
            } else if (type.bits == 64) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<uint64_t>(sizes));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.code == ion_type_float) {
            if (type.bits == 32) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<float>(sizes));
            } else if (type.bits == 64) {
                *ptr = reinterpret_cast<ion_buffer_t>(make_buffer<double>(sizes));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else {
            throw std::runtime_error("Unsupported type code");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_buffer_destroy(ion_buffer_t obj)
{
    try {
        // NOTE: Halide::Buffer class layout is safe to be deleted as T=void
        delete reinterpret_cast<Halide::Buffer<void>*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_buffer_write(ion_buffer_t obj, void *ptr, int size)
{
    try {
        // NOTE: Halide::Buffer class layout is safe to call Halide::Buffer<void>::type()
        auto type = reinterpret_cast<Halide::Buffer<void>*>(obj)->type();
        if (type.is_int()) {
            if (type.bits() == 8) {
                std::memcpy(reinterpret_cast<Halide::Buffer<int8_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 16) {
                std::memcpy(reinterpret_cast<Halide::Buffer<int16_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 32) {
                std::memcpy(reinterpret_cast<Halide::Buffer<int32_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 64) {
                std::memcpy(reinterpret_cast<Halide::Buffer<int64_t>*>(obj)->data(), ptr, size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_uint()) {
            if (type.bits() == 1) {
                std::memcpy(reinterpret_cast<Halide::Buffer<bool>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 8) {
                std::memcpy(reinterpret_cast<Halide::Buffer<uint8_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 16) {
                std::memcpy(reinterpret_cast<Halide::Buffer<uint16_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 32) {
                std::memcpy(reinterpret_cast<Halide::Buffer<uint32_t>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 64) {
                std::memcpy(reinterpret_cast<Halide::Buffer<uint64_t>*>(obj)->data(), ptr, size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_float()) {
            if (type.bits() == 32) {
                std::memcpy(reinterpret_cast<Halide::Buffer<float>*>(obj)->data(), ptr, size);
            } else if (type.bits() == 64) {
                std::memcpy(reinterpret_cast<Halide::Buffer<double>*>(obj)->data(), ptr, size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else {
            throw std::runtime_error("Unsupported type code");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_buffer_read(ion_buffer_t obj, void *ptr, int size)
{
    try {
        // NOTE: Halide::Buffer class layout is safe to call Halide::Buffer<void>::type()
        auto type = reinterpret_cast<Halide::Buffer<void>*>(obj)->type();
        if (type.is_int()) {
            if (type.bits() == 8) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<int8_t>*>(obj)->data(), size);
            } else if (type.bits() == 16) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<int16_t>*>(obj)->data(), size);
            } else if (type.bits() == 32) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<int32_t>*>(obj)->data(), size);
            } else if (type.bits() == 64) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<int64_t>*>(obj)->data(), size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_uint()) {
            if (type.bits() == 1) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<bool>*>(obj)->data(), size);
            } else if (type.bits() == 8) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<uint8_t>*>(obj)->data(), size);
            } else if (type.bits() == 16) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<uint16_t>*>(obj)->data(), size);
            } else if (type.bits() == 32) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<uint32_t>*>(obj)->data(), size);
            } else if (type.bits() == 64) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<uint64_t>*>(obj)->data(), size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_float()) {
            if (type.bits() == 32) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<float>*>(obj)->data(), size);
            } else if (type.bits() == 64) {
                std::memcpy(ptr, reinterpret_cast<Halide::Buffer<double>*>(obj)->data(), size);
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else {
            throw std::runtime_error("Unsupported type code");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_port_map_create(ion_port_map_t *ptr)
{
    try {
        *ptr = reinterpret_cast<ion_port_map_t>(new PortMap);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

int ion_port_map_destroy(ion_port_map_t obj)
{
    try {
        delete reinterpret_cast<PortMap*>(obj);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}


#define ION_PORT_MAP_SET_IMPL(T, POSTFIX)                                         \
    int ion_port_map_set_##POSTFIX(ion_port_map_t obj, ion_port_t p, T v) {       \
        try {                                                                     \
            reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), v); \
        } catch (const std::exception& e) {                                 \
            std::cerr << e.what() << std::endl;                             \
            return -1;                                                      \
        } catch (...) {                                                     \
            std::cerr << "Unknown exception was happened." << std::endl;    \
            return -1;                                                      \
        }                                                                   \
                                                                            \
        return 0;                                                           \
    }

ION_PORT_MAP_SET_IMPL(int8_t, i8)
ION_PORT_MAP_SET_IMPL(int16_t, i16)
ION_PORT_MAP_SET_IMPL(int32_t, i32)
ION_PORT_MAP_SET_IMPL(int64_t, i64)
ION_PORT_MAP_SET_IMPL(bool, u1)
ION_PORT_MAP_SET_IMPL(uint8_t, u8)
ION_PORT_MAP_SET_IMPL(uint16_t, u16)
ION_PORT_MAP_SET_IMPL(uint32_t, u32)
ION_PORT_MAP_SET_IMPL(uint64_t, u64)
ION_PORT_MAP_SET_IMPL(float, f32)
ION_PORT_MAP_SET_IMPL(double, f64)

#undef ION_PORT_MAP_SET_IMPL

int ion_port_map_set_buffer(ion_port_map_t obj, ion_port_t p, ion_buffer_t b)
{
    try {
        // NOTE: Halide::Buffer class layout is safe to call Halide::Buffer<void>::type()
        auto type = reinterpret_cast<Halide::Buffer<void>*>(b)->type();
        if (type.is_int()) {
            if (type.bits() == 8) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<int8_t>*>(b));
            } else if (type.bits() == 16) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<int16_t>*>(b));
            } else if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<int32_t>*>(b));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<int64_t>*>(b));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_uint()) {
            if (type.bits() == 1) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<bool>*>(b));
            } else if (type.bits() == 8) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<uint8_t>*>(b));
            } else if (type.bits() == 16) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<uint16_t>*>(b));
            } else if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<uint32_t>*>(b));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<uint64_t>*>(b));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_float()) {
            if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<float>*>(b));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), *reinterpret_cast<Halide::Buffer<double>*>(b));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else {
            throw std::runtime_error("Unsupported type code");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

template<typename T>
std::vector<Halide::Buffer<T>> convert(ion_buffer_t *b, int n) {
    std::vector<Halide::Buffer<T>> bs(n);
    for (int i=0; i<n; ++i) {
        bs[i] = *reinterpret_cast<Halide::Buffer<T>*>(b[i]);
    }
    return bs;
}

int ion_port_map_set_buffer_array(ion_port_map_t obj, ion_port_t p, ion_buffer_t *b, int n)
{
    try {
        // NOTE: Halide::Buffer class layout is safe to call Halide::Buffer<void>::type()
        auto type = reinterpret_cast<Halide::Buffer<void>*>(*b)->type();
        if (type.is_int()) {
            if (type.bits() == 8) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<int8_t>(b, n));
            } else if (type.bits() == 16) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<int16_t>(b, n));
            } else if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<int32_t>(b, n));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<int64_t>(b, n));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_uint()) {
            if (type.bits() == 1) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<bool>(b, n));
            } else if (type.bits() == 8) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<uint8_t>(b, n));
            } else if (type.bits() == 16) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<uint16_t>(b, n));
            } else if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<uint32_t>(b, n));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<uint64_t>(b, n));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else if (type.is_float()) {
            if (type.bits() == 32) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<float>(b, n));
            } else if (type.bits() == 64) {
                reinterpret_cast<PortMap*>(obj)->set(*reinterpret_cast<Port*>(p), convert<double>(b, n));
            } else {
                throw std::runtime_error("Unsupported bits number");
            }
        } else {
            throw std::runtime_error("Unsupported type code");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception was happened." << std::endl;
        return -1;
    }

    return 0;
}

