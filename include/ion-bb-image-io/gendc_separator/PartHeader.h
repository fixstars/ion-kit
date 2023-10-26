#ifndef PARTHEADER_H
#define PARTHEADER_H
#include <iostream>
#include <vector>
#include <array>

// for std::setw
#include <iomanip>

#include "Descriptor.h"

int getByteInFormat(int format){
    switch (format){
        case Mono12:
            return 2;
        case Data8:
            return 1;
        case Data16:
            return 2;
        case Data32:
            return 4;
        case Data32f:
            return 4;
        default:
            throw std::invalid_argument("wrong format\n");
    }
}
// namespace {

class PartHeader : public Header{
public: 
    PartHeader(){}
    // constructor with existing header info
    PartHeader(char* header_info, size_t offset = 0){

        size_t total_size = 0;
        offset += Read(header_info, offset, HeaderType_);
        offset += Read(header_info, offset, Flags_);
        offset += Read(header_info, offset, HeaderSize_);
        offset += Read(header_info, offset, Format_);
        offset += sizeof(Reserved_);
        offset += Read(header_info, offset, FlowId_);
        offset += Read(header_info, offset, FlowOffset_);
        offset += Read(header_info, offset, DataSize_);
        offset += Read(header_info, offset, DataOffset_);

        // get number of typespecific fields from HeaderSize_
        int num_typespecific = getNumTypeSpecific(HeaderSize_);

        if (num_typespecific > 0){
            offset += Read(header_info, offset, Dimension_[0]);
            offset += Read(header_info, offset, Dimension_[1]);
        }
        if (num_typespecific > 1){
            offset += Read(header_info, offset, Padding_[0]);
            offset += Read(header_info, offset, Padding_[1]);
        }
        if (num_typespecific > 2){
            offset += sizeof(InfoReserved_);
            int64_t typespecific_item;
            for (int i = 0; i < num_typespecific - 2; ++i){
                offset += Read(header_info, offset, typespecific_item);
                TypeSpecific_.push_back(typespecific_item);
            }         
        }
    }

    PartHeader& operator=(const PartHeader& src) {
        HeaderType_ = src.HeaderType_;
        Flags_= src.Flags_;
        HeaderSize_= src.HeaderSize_;
        Format_= src.Format_;
        // Reserved_ = 0;
        FlowId_= src.FlowId_;
        FlowOffset_= src.FlowOffset_;
        DataSize_= src.DataSize_;
        DataOffset_= src.DataOffset_;

        Dimension_= src.Dimension_;
        Padding_= src.Padding_;
        TypeSpecific_= src.TypeSpecific_;
        return *this;
    }

    size_t GenerateDescriptor(char* ptr, size_t offset=0){
        offset = GenerateHeader(ptr, offset);
        return offset;
    }

    bool isData2DImage(){
        return HeaderType_ == 0x4200;
    }

    int64_t getDataOffset(){
        return DataOffset_;
    }

    int64_t getDataSize(){
        return DataSize_;
    }

    int32_t getOffsetFromTypeSpecific(int32_t kth_typespecific, int32_t typespecific_offset = 0){
        return offset_for_version[GENDC_V10].at(2) + 8 * (kth_typespecific - 1) + typespecific_offset;
    }

    void DisplayHeaderInfo(){
        int total_size = 0;
        std::cout << "\nPART HEADER" << std::endl;
        total_size += DisplayItemInfo("HeaderType_", HeaderType_, 3, true);
        total_size += DisplayItemInfo("Flags_", Flags_, 3, true);
        total_size += DisplayItemInfo("HeaderSize_", HeaderSize_, 3);
        total_size += DisplayItemInfo("Format_", Format_, 3, true);
        total_size += DisplayItemInfo("Reserved_", Reserved_, 3, true);
        total_size += DisplayItemInfo("FlowId_", FlowId_, 3);
        total_size += DisplayItemInfo("FlowOffset_", FlowOffset_, 3);
        total_size += DisplayItemInfo("DataSize_", DataSize_, 3);
        total_size += DisplayItemInfo("DataOffset_", DataOffset_, 3);

        total_size += DisplayContainer("Dimension_", Dimension_, 3);
        total_size += DisplayContainer("Padding_", Padding_, 3);
        total_size += DisplayItemInfo("InfoReserved_", InfoReserved_, 3);

        total_size += DisplayContainer("TypeSpecific_", TypeSpecific_, 3);     
    }

private:
    // you need parameters to create the object

    int getNumTypeSpecific(size_t header_size){
        return static_cast<int>(( header_size - 40 ) / 8);
    }

    size_t GenerateHeader(char* ptr, size_t offset = 0){
        // modify the order/items only when the structure is changed.
        // when you change this, don't forget to change copy constructor.
        size_t cpy_offset = offset;
        offset += Write(ptr, offset, HeaderType_);
        offset += Write(ptr, offset, Flags_);
        offset += Write(ptr, offset, HeaderSize_);
        offset += Write(ptr, offset, Format_);
        offset += Write(ptr, offset, Reserved_);
        offset += Write(ptr, offset, FlowId_);
        offset += Write(ptr, offset, FlowOffset_);
        offset += Write(ptr, offset, DataSize_);
        offset += Write(ptr, offset, DataOffset_);
        offset += WriteContainer(ptr, offset, Dimension_);
        offset += WriteContainer(ptr, offset, Padding_);
        offset += Write(ptr, offset, InfoReserved_);
        offset += WriteContainer(ptr, offset, TypeSpecific_);

        if ((offset - cpy_offset) != HeaderSize_){
            std::cerr << "Part header size is wrong" << HeaderSize_ << " != " << offset - cpy_offset << std::endl;
        }
        return offset;
    }

    int16_t HeaderType_;
    int16_t Flags_;
    // int32_t HeaderSize_; 
    int32_t Format_;
    const int16_t Reserved_ = 0;
    int16_t FlowId_;
    int64_t FlowOffset_;
    int64_t DataSize_;
    int64_t DataOffset_;

    // optional
    std::array<int32_t, 2> Dimension_;
    std::array<int16_t, 2> Padding_;
    const int32_t InfoReserved_ = 0;
    std::vector<int64_t> TypeSpecific_;
};
// }
#endif /*PARTHEADER_H*/