#ifndef CONTAINERHEADER_H
#define CONTAINERHEADER_H

#include "ComponentHeader.h"

class ContainerHeader : public Header{
public:

    ContainerHeader(){}

    ContainerHeader(char* descriptor){
        size_t offset = 0;
        int32_t signature;
        int16_t header_type;

        // check if the container is GenDC
        offset += Read(descriptor, offset, signature);
        if (signature != Signature_){
            std::cerr << "This ptr does NOT hace GenDC Signature" << std::endl;
        }

        for (int i = 0 ; i < Version_.size(); i++){
            int8_t v;
            offset += Read(descriptor, offset, v);
            Version_.at(i) = v;
        }
        offset += sizeof(Reserved_);

        offset += Read(descriptor, offset, header_type);
        if (header_type != HeaderType_){
            std::cerr << "wrong header type in container header" << std::endl;
        }
        offset += Read(descriptor, offset, Flags_);
        offset += Read(descriptor, offset, HeaderSize_);

        offset += Read(descriptor, offset, Id_);
        offset += Read(descriptor, offset, VariableFields_);
        offset += Read(descriptor, offset, DataSize_);
        offset += Read(descriptor, offset, DataOffset_);
        offset += Read(descriptor, offset, DescriptorSize_);
        offset += Read(descriptor, offset, ComponentCount_);

        for (int i = 0; i < ComponentCount_; ++i){
            int64_t single_component_offset;
            offset += Read(descriptor, offset, single_component_offset);
            ComponentOffset_.push_back(single_component_offset);
        }

        for (int64_t & co : ComponentOffset_){
            component_header_.push_back(ComponentHeader(descriptor, co));
        }
    }

    ContainerHeader& operator=(const ContainerHeader& src) {
        component_header_ = src.component_header_;

        // Signature_ = 0x43444E47;
        Version_ = src.Version_;
        // Reserved_ = 0;
        // HeaderType_ = 0x1000;
        Flags_ = src.Flags_;
        HeaderSize_ = src.HeaderSize_;
        Id_ = src.Id_;
        VariableFields_ = src.VariableFields_;
        DataSize_ = src.DataSize_;
        DataOffset_ = src.DataOffset_;
        DescriptorSize_ = src.DescriptorSize_;
        ComponentCount_ = src.ComponentCount_;
        ComponentOffset_ = src.ComponentOffset_;

        return *this;
    }

    int32_t getDescriptorSize(){
        return DescriptorSize_;
    }

    size_t GenerateDescriptor(char* ptr){
        size_t offset = 0;
        offset = GenerateHeader(ptr);

        for ( ComponentHeader &ch : component_header_){
            offset = ch.GenerateDescriptor(ptr, offset);
        }

        if ( offset != DescriptorSize_){
            std::cerr << "Descriptor size is wrong" << DescriptorSize_ << " != " << offset << std::endl;
        }
        return offset;
    }

    std::tuple<int32_t, int32_t> getFirstAvailableDataOffset(bool image){
        // returns the component and part header index where
        // - component is valid
        // - part header type is 0x4200 (GDC_2D) if image is true
        int32_t ith_comp = 0;
        for (ComponentHeader &ch : component_header_){
            if (ch.isComponentValid()){
                int32_t jth_part = ch.getFirstAvailableDataOffset(image);
                if (jth_part != -1){
                    return std::make_tuple(ith_comp, jth_part);
                }
                ++ith_comp;
            }
        }
        return std::make_tuple(-1, -1);
    }

    int64_t getDataOffset(int32_t ith_component = 0, int32_t jth_part = 0){
        if (ith_component == 0 && jth_part == 0){
            return DataOffset_;
        }
        return component_header_.at(ith_component).getDataOffset(jth_part);
    }

    int64_t getDataSize(){
        return DataSize_;
    }

    int64_t getDataSize(int32_t ith_component = 0, int32_t jth_part = 0){
        return component_header_.at(ith_component).getDataSize(jth_part);
    }

    int32_t getOffsetFromTypeSpecific(int32_t ith_component, int32_t jth_part,
        int32_t kth_typespecific, int32_t typespecific_offset = 0){

        return component_header_.at(ith_component).getOffsetFromTypeSpecific(jth_part, kth_typespecific, typespecific_offset);
    }

    void DisplayHeaderInfo(){
        int total_size = 0;
        std::cout << "\nCONTAINER HEADER" << std::endl;
        total_size += DisplayItemInfo("Signature_", Signature_, 1, true);
        total_size += DisplayContainer("Version_", Version_, 1, true);
        total_size += DisplayItemInfo("Reserved_", Reserved_, 1);
        total_size += DisplayItemInfo("HeaderType_", HeaderType_, 1, true);
        total_size += DisplayItemInfo("Flags_", Flags_, 1, true);
        total_size += DisplayItemInfo("HeaderSize_", HeaderSize_, 1);
        total_size += DisplayItemInfo("Id_", Id_, 1);
        total_size += DisplayItemInfo("VariableFields_", VariableFields_, 1, true);
        total_size += DisplayItemInfo("DataSize_", DataSize_, 1);
        total_size += DisplayItemInfo("DataOffset_", DataOffset_, 1);
        total_size += DisplayItemInfo("DescriptorSize_", DescriptorSize_, 1);
        total_size += DisplayItemInfo("ComponentCount_", ComponentCount_, 1);

        total_size += DisplayContainer("ComponentOffset_", ComponentOffset_, 1);

        for (ComponentHeader &ch : component_header_){
            ch.DisplayHeaderInfo();
        }
    }

private:
    size_t GenerateHeader(char* ptr){
        // modify the order/items only when the structure is changed.
        // when you change this, don't forget to change copy constructor.
        size_t offset = 0;
        offset += Write(ptr, offset, Signature_);
        offset += WriteContainer(ptr, offset, Version_);
        offset += Write(ptr, offset, Reserved_);
        offset += Write(ptr, offset, HeaderType_);
        offset += Write(ptr, offset, Flags_);
        offset += Write(ptr, offset, HeaderSize_);
        offset += Write(ptr, offset, Id_);
        offset += Write(ptr, offset, VariableFields_);
        offset += Write(ptr, offset, DataSize_);
        offset += Write(ptr, offset, DataOffset_);
        offset += Write(ptr, offset, DescriptorSize_);
        offset += Write(ptr, offset, ComponentCount_);
        offset += WriteContainer(ptr, offset, ComponentOffset_);

        if ( offset != HeaderSize_){
            std::cerr << "Container header size is wrong" << HeaderSize_ << " != " << offset << std::endl;
        }
        return offset;
    }

    // variables to interpret user input
    std::vector<ComponentHeader> component_header_;

    // member variables
    const int32_t Signature_ = 0x43444E47;
    std::array<int8_t, 3> Version_;
    const int8_t Reserved_ = 0;
    const int16_t HeaderType_ = 0x1000;
    int16_t Flags_;
    // int32_t HeaderSize_;
    int64_t Id_;
    int64_t VariableFields_; //including 6 Byte-wide Reserved
    int64_t DataSize_;
    int64_t DataOffset_;
    int32_t DescriptorSize_;
    int32_t ComponentCount_;
    std::vector<int64_t> ComponentOffset_;
};

#endif /*CONTAINERHEADER_H*/
