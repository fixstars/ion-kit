#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <sstream>

#include "Descriptor.h"

// *****************************************************************************
// NOTE: the layout of the first 8 bytes will never change.
//       this contains signature and version info
// *****************************************************************************

bool isGenDC(char* buf){
    int32_t signature;
    std::memcpy(&signature, buf + SIGNATURE_OFFSET, sizeof(int32_t));

    if (signature != GENDC_SIGNATURE){
        std::cout << "[LOG ion-kit(gendc-separator)] The data is not genDC format" << std::endl;
        return false;
    }
    return true;
}

std::array<int8_t, 3> getGenDCVersion(char* buf){
    std::array<int8_t, 3> version;
    for (int i = 0; i < version.size(); ++i){
        std::memcpy(&version.at(i), buf + VERSION_OFFSET + sizeof(int8_t)*i, sizeof(int8_t));
    }
    return version;
}

int32_t getDescriptorSize(char* buf, const int container_version, std::array<int8_t, 3>& v){
    int8_t hex_offset = 0x00;
    int32_t descriptor_size;

    try{
        std::memcpy(&descriptor_size, buf + (offset_for_version.at(container_version)).at(::descriptor_size), sizeof(int32_t));
    }catch (std::out_of_range& e){
        std::stringstream ss;
        ss << "ERROR\t" << e.what() << ": "
            << "The version of container " 
            << v.at(0) - hex_offset << "."
            << v.at(1) - hex_offset << "."
            << v.at(2) - hex_offset << " is not supported.";
        const std::string error_message = ss.str();
        throw std::out_of_range(error_message);
    }catch(std::exception& e){
        throw e;
    } 
    return descriptor_size;
}

#endif /*TOOLS_H*/