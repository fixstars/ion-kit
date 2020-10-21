#include <exception>

#include <ion/ion.h>

#include <ion/c_ion.h>

using namespace ion;

//
// ion_port_t
//
int ion_port_create(ion_port_t *ptr, const char *key, ion_type_t type)
{
    try {
        *ptr = reinterpret_cast<ion_port_t>(new Port(key, halide_type_t(static_cast<halide_type_code_t>(type.code), type.bits, type.lanes)));
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

int ion_builder_compile(ion_builder_t obj, const char *function_name, ion_builder_compile_option_t *option)
{
    try {
        reinterpret_cast<Builder*>(obj)->compile(function_name, Builder::CompileOption{option->output_directory});
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
