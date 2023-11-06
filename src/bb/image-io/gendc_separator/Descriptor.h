#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#include <iostream>
#include <vector>

#include <iomanip>
#include <map>
#include <tuple>

#define GENDC_SIGNATURE 0x43444E47

// *****************************************************************************
// const offset v1.0.1
// *****************************************************************************
#define DEFAULT_CONT_HEADER_SIZE 56
#define DEFAULT_COMP_HEADER_SIZE 48
#define DEFAULT_PART_HEADER_SIZE 40

// *****************************************************************************
// ID & hex values
// *****************************************************************************

// TypeId (i.e. ComponentIdValue)
#define Type_Undefined 0
#define Type_Intensity 1
#define Type_Infrared 2
#define Type_Ultraviolet 3
#define Type_Range 4
#define Type_Reflectance 5
#define Type_Confidence 6
#define Type_Scatter 7
#define Type_Disparity 8
#define Type_Multispectral 9
#define Type_Metadata 0x8001

// header type
#define GDC_1D 0x4100
#define GDC_2D 0x4200

//format
#define Mono12 0x01100005
#define Data8 0x01080116
#define Data16 0x01100118
#define Data32 0x0120011A
#define Data32f 0x0120011C

// *****************************************************************************
// dispaly
// *****************************************************************************

#define DISPLAY_ITEM_WIDTH 16
#define DISPLAY_SIZE_WIDTH 4
#define DISPLAY_VALUE_WIDTH 10

// *****************************************************************************
// for container version and descriptor offset
// *****************************************************************************
#define SIGNATURE_OFFSET 0
#define VERSION_OFFSET 4

namespace {
    enum offset {
        descriptor_size,
        deta_size,
        data_offset,
    };
}

#define GENDC_V10 0x0100

// https://www.emva.org/wp-content/uploads/GenICam_GenDC_v1_1.pdf
std::map<int32_t, std::array<int32_t, 3>> offset_for_version = 
{
    {GENDC_V10, std::array<int32_t, 3>{48, 32, 40}},
};


enum display_lebel{
    default_display,
    container_header_display,
    component_header_display,
    part_header_display
};

std::string display_indent(int level=default_display){
    std::string ret="";
    for (int i = 0; i < level; ++i){
        ret += "\t";
    }
    return ret;
} 

class Header{
public:
    size_t getHeaderSize(){
        return HeaderSize_;
    }

protected:
    template <typename T>
    void DisplayItem(T item, bool hex_format){
        if(sizeof(item) == sizeof(char)){
            DisplayItem(static_cast<int>(item), hex_format);
        }else{
            std::cout << std::right << std::setw(DISPLAY_VALUE_WIDTH);
            if (hex_format){
                std::cout << std::hex << "0x" << item << std::endl;
            }else{
                std::cout << std::dec << item << std::endl;
            }
        }

    }

    template <typename T>
    int DisplayItemInfo(std::string item_name, T item, int level=default_display, bool hex_format=false){
        std::string indent = display_indent(level);
        int sizeof_item = sizeof(item);
        std::cout << indent << std::right << std::setw(DISPLAY_ITEM_WIDTH) << item_name;
        std::cout << std::right << std::setw(DISPLAY_SIZE_WIDTH)  << " (" << sizeof_item << "):";
        DisplayItem<T>(item, hex_format);
        return sizeof_item;
    }

    template <typename T>
    int DisplayContainer(std::string container_name, const std::vector<T>&container, int level=default_display, bool hex=false){
        int total_size = 0;
        if (container.size() > 0){
            std::string key = container_name;
            for(int i=0; i < container.size(); ++i){
                total_size += DisplayItemInfo(i > 0 ? "" : key, container.at(i), level, hex);
            }
        }else{
            std::cout << display_indent(level) << std::right << std::setw(DISPLAY_ITEM_WIDTH) << container_name;
            std::cout << std::right << std::setw(DISPLAY_SIZE_WIDTH)  << " (" << 0 << "):\n";
        }
        return total_size;
    }

    template <typename T, size_t N>
    int DisplayContainer(std::string container_name, const std::array<T, N>&container, int level=default_display, bool hex=false){
        int total_size = 0;
        if (container.size() > 0){
            std::string key = container_name;
            for(int i=0; i < container.size(); ++i){
                total_size += DisplayItemInfo(i > 0 ? "" : key, container.at(i), level, hex);
            }
        }else{
            std::cout << display_indent(level) << std::right << std::setw(DISPLAY_ITEM_WIDTH) << container_name;
            std::cout << std::right << std::setw(DISPLAY_SIZE_WIDTH)  << " (" << 0 << "):\n";
        }
        return total_size;
    }
    

    template <typename T>
    size_t Read(char* ptr, size_t offset, T& item){
        memcpy(&item, ptr+static_cast<int>(offset), sizeof(item));
        return sizeof(item);
    }

    template <typename T>
    size_t Write(char* ptr, size_t offset, T item){
        memcpy(ptr+static_cast<int>(offset), &item, sizeof(item));
        return sizeof(item);
    }

    template <typename T>
    size_t WriteContainer(char* ptr, size_t offset, std::vector<T>&container){
        size_t container_offset = 0;
        for (T& item : container){
            container_offset += Write(ptr, offset + container_offset, item);
        }
        return container_offset;
    }

    template <typename T, size_t N>
    size_t WriteContainer(char* ptr, size_t offset, std::array<T, N>&container){
        size_t container_offset = 0;
        for (T& item : container){
            container_offset += Write(ptr, offset + container_offset, item);
        }
        return container_offset;
    }

protected:
    int32_t HeaderSize_ = 0;
};
#endif /*DESCRIPTOR_H*/