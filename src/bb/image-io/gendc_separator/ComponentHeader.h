#ifndef COMPONENTHEADER_H
#define COMPONENTHEADER_H

#include "PartHeader.h"

class ComponentHeader : public Header{
public:
    ComponentHeader(){}

    ComponentHeader(char* header_info, size_t offset = 0){
        int16_t header_type;
        offset += Read(header_info, offset, header_type);
        if (header_type != HeaderType_){
            std::cerr << "wrong header type in component header" << std::endl;
        }

        offset += Read(header_info, offset, Flags_);
        offset += Read(header_info, offset, HeaderSize_);
        offset += sizeof(Reserved_);
        offset += Read(header_info, offset, GroupId_);
        offset += Read(header_info, offset, SourceId_);
        offset += Read(header_info, offset, RegionId_);
        offset += Read(header_info, offset, RegionOffsetX_);
        offset += Read(header_info, offset, RegionOffsetY_);
        offset += Read(header_info, offset, Timestamp_);
        offset += Read(header_info, offset, TypeId_);
        offset += Read(header_info, offset, Format_);
        offset += sizeof(Reserved2_);
        offset += Read(header_info, offset, PartCount_);

        for (int i = 0; i < PartCount_; ++i){
            int64_t single_part_offset;
            offset += Read(header_info, offset, single_part_offset);
            PartOffset_.push_back(single_part_offset);
        }

        for (int64_t & po : PartOffset_){
            partheader_.push_back(PartHeader(header_info, po));
        }

    }

    ComponentHeader& operator=(const ComponentHeader& src) {
        partheader_ = src.partheader_;
    
        // HeaderType_ = 0x2000;
        Flags_ = src.Flags_;
        HeaderSize_ = src.HeaderSize_;
        // Reserved_ = 0;
        GroupId_ = src.GroupId_;
        SourceId_ = src.SourceId_;
        RegionId_ = src.RegionId_;
        RegionOffsetX_ = src.RegionOffsetX_;
        RegionOffsetY_ = src.RegionOffsetY_;
        Timestamp_ = src.Timestamp_;
        TypeId_ = src.TypeId_;
        Format_ = src.Format_;
        // Reserved2_ = 0;
        PartCount_ = src.PartCount_;
        PartOffset_ = src.PartOffset_;
        return *this;
    }

    size_t GenerateDescriptor(char* ptr, size_t offset=0){
        offset = GenerateHeader(ptr, offset);

        for (PartHeader & ph : partheader_){
            offset = ph.GenerateDescriptor(ptr, offset);
        }
        return offset;
    }

    bool isComponentValid(){
        return Flags_ == 0;
    }

    int32_t getFirstAvailableDataOffset(bool image){
        // returns the part header index where
        // - component is valid
        // - part header type is 0x4200 (GDC_2D) if image is true
        int32_t jth_part = 0;
        for (PartHeader &ph : partheader_){
            if (image && ph.isData2DImage()){
                return jth_part;
            }else if (!image && !ph.isData2DImage()){
                return jth_part;
            }
            ++jth_part;
        }
        return -1;
    }

    int64_t getDataOffset(int32_t jth_part){
        return partheader_.at(jth_part).getDataOffset();
    }

    int64_t getDataSize(int32_t jth_part){
        return partheader_.at(jth_part).getDataSize();
    }

    int32_t getOffsetFromTypeSpecific(int32_t jth_part, int32_t kth_typespecific, int32_t typespecific_offset = 0){
        return PartOffset_.at(jth_part) + partheader_.at(jth_part).getOffsetFromTypeSpecific(kth_typespecific, typespecific_offset);
    }

    void DisplayHeaderInfo(){
        int total_size = 0;
        std::cout << "\nCOMPONENT HEADER" << std::endl;
        total_size += DisplayItemInfo("HeaderType_", HeaderType_, 2, true);
        total_size += DisplayItemInfo("Flags_", Flags_, 2, true);
        total_size += DisplayItemInfo("HeaderSize_", HeaderSize_, 2);
        total_size += DisplayItemInfo("Reserved_", Reserved_, 2, true);
        total_size += DisplayItemInfo("GroupId_", GroupId_, 2, true);
        total_size += DisplayItemInfo("SourceId_", SourceId_, 2, true);
        total_size += DisplayItemInfo("RegionId_", RegionId_, 2, true);
        total_size += DisplayItemInfo("RegionOffsetX_", RegionOffsetX_, 2);
        total_size += DisplayItemInfo("RegionOffsetY_", RegionOffsetY_, 2);
        total_size += DisplayItemInfo("Timestamp_", Timestamp_, 2);
        total_size += DisplayItemInfo("TypeId_", TypeId_, 2, true);
        total_size += DisplayItemInfo("Format_", Format_, 2, true);
        total_size += DisplayItemInfo("Reserved2_", Reserved2_, 2, true);
        total_size += DisplayItemInfo("PartCount_", PartCount_, 2);

        total_size += DisplayContainer("PartOffset_", PartOffset_, 2);

        for (PartHeader &ph : partheader_){
            ph.DisplayHeaderInfo();
        }
    }

private:
    size_t GenerateHeader(char* ptr, size_t offset=0){
        // modify the order/items only when the structure is changed.
        // when you change this, don't forget to change copy constructor.
        size_t cpy_offset = offset;
        offset += Write(ptr, offset, HeaderType_);
        offset += Write(ptr, offset, Flags_);
        offset += Write(ptr, offset, HeaderSize_);
        offset += Write(ptr, offset, Reserved_);
        offset += Write(ptr, offset, GroupId_);
        offset += Write(ptr, offset, SourceId_);
        offset += Write(ptr, offset, RegionId_);
        offset += Write(ptr, offset, RegionOffsetX_);
        offset += Write(ptr, offset, RegionOffsetY_);
        offset += Write(ptr, offset, Timestamp_);
        offset += Write(ptr, offset, TypeId_);
        offset += Write(ptr, offset, Format_);
        offset += Write(ptr, offset, Reserved2_);
        offset += Write(ptr, offset, PartCount_);
        
        offset += WriteContainer(ptr, offset, PartOffset_);

        if ((offset - cpy_offset) != HeaderSize_){
            std::cerr << "Component header size is wrong" << HeaderSize_ << " != " << offset - cpy_offset << std::endl;
        }
        return offset;
    }

    std::vector<PartHeader> partheader_;
    
    const int16_t HeaderType_ = 0x2000;
    int16_t Flags_;
    // int32_t HeaderSize_;
    const int16_t Reserved_ = 0;
    int16_t GroupId_;
    int16_t SourceId_;
    int16_t RegionId_;
    int32_t RegionOffsetX_;
    int32_t RegionOffsetY_;
    int64_t Timestamp_;
    int64_t TypeId_;
    int32_t Format_;
    const int16_t Reserved2_ = 0;
    int16_t PartCount_;
    std::vector<int64_t> PartOffset_;
};


#endif /*COMPONENTHEADER_H*/