#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Halide.h>

namespace ion {

class ParamContainer {
 public:
    ParamContainer() {
    }

    ParamContainer(const std::string& k, Halide::Type t, int32_t d = 0)
        : type_(t), dim_(d)
    {
        if (d == 0) {
            // Scalar Param
            if (t.is_int()) {
                switch (t.bits()) {
                case 8:  i8_  = Halide::Param<int8_t>(k);  break;
                case 16: i16_ = Halide::Param<int16_t>(k); break;
                case 32: i32_ = Halide::Param<int32_t>(k); break;
                case 64: i64_ = Halide::Param<int64_t>(k); break;
                default: throw std::runtime_error("Unsupported type");
                }
            } else if (t.is_uint()) {
                switch (t.bits()) {
                case 1:  b_   = Halide::Param<bool>(k);     break;
                case 8:  u8_  = Halide::Param<uint8_t>(k);  break;
                case 16: u16_ = Halide::Param<uint16_t>(k); break;
                case 32: u32_ = Halide::Param<uint32_t>(k); break;
                case 64: u64_ = Halide::Param<uint64_t>(k); break;
                default: throw std::runtime_error("Unsupported type");
                }
            } else if (t.is_float()) {
                switch (t.bits()) {
                case 32: f32_ = Halide::Param<float>(k);  break;
                case 64: f64_ = Halide::Param<double>(k); break;
                default: throw std::runtime_error("Unsupported type");
                }
            } else {
                throw std::runtime_error("Unsupported type");
            }
        } else {
            // Vector Param
            v_ = Halide::ImageParam(t, d, k);
        }
    }

    Halide::Expr expr() {
        if (dim_ != 0) {
            throw std::runtime_error("Invalid port type");
        }

        if (type_.is_int()) {
            switch (type_.bits()) {
            case 8:  return i8_;
            case 16: return i16_;
            case 32: return i32_;
            case 64: return i64_;
            default: throw std::runtime_error("unsupported type");
            }
        } else if (type_.is_uint()) {
            switch (type_.bits()) {
            case 1:  return b_;
            case 8:  return u8_;
            case 16: return u16_;
            case 32: return u32_;
            case 64: return u64_;
            default: throw std::runtime_error("unsupported type");
            }
        } else if (type_.is_float()) {
            switch (type_.bits()) {
            case 32: return f32_;
            case 64: return f64_;
            default: throw std::runtime_error("unsupported type");
            }
        } else {
            throw std::runtime_error("unsupported type");
        }

        return Halide::Expr();
    }

    Halide::Func func() {
        return v_;
    }

    template<typename T>
    void set_to_param_map(Halide::ParamMap& pm, T v) {
        throw std::runtime_error("Implement me");
    }

    template<typename T>
    void set_to_param_map(Halide::ParamMap& pm, Halide::Buffer<T> &buf) {
        if (dim_ == 0) {
            throw std::runtime_error("Invalid port type");
        }
        pm.set(v_, buf);
    }

 private:

    Halide::Param<bool>     b_;
    Halide::Param<int8_t>   i8_;
    Halide::Param<int16_t>  i16_;
    Halide::Param<int32_t>  i32_;
    Halide::Param<int64_t>  i64_;
    Halide::Param<uint8_t>  u8_;
    Halide::Param<uint16_t> u16_;
    Halide::Param<uint32_t> u32_;
    Halide::Param<uint64_t> u64_;
    Halide::Param<float>    f32_;
    Halide::Param<double>   f64_;
    Halide::ImageParam      v_;

    Halide::Type type_;
    int32_t dim_;
};

#define DECLARE_SET_TO_PARAM(TYPE, MEMBER_V)                              \
template<>                                                                \
void ParamContainer::set_to_param_map<TYPE>(Halide::ParamMap& pm, TYPE v)
DECLARE_SET_TO_PARAM(bool, b_);
DECLARE_SET_TO_PARAM(int8_t, i8_);
DECLARE_SET_TO_PARAM(int16_t, i16_);
DECLARE_SET_TO_PARAM(int32_t, i32_);
DECLARE_SET_TO_PARAM(int64_t, i64_);
DECLARE_SET_TO_PARAM(uint8_t, u8_);
DECLARE_SET_TO_PARAM(uint16_t, u16_);
DECLARE_SET_TO_PARAM(uint32_t, u32_);
DECLARE_SET_TO_PARAM(uint64_t, u64_);
DECLARE_SET_TO_PARAM(float, f32_);
DECLARE_SET_TO_PARAM(double, f64_);
#undef DECLARE_SET_TO_PARAM

/**
 * Port class is used to create dynamic i/o for each node.
 */
class Port {
 public:
     friend class Node;

     Port()
         : key_(), type_(), dimensions_(0), index_(-1), node_id_(), param_info_() {}

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& k, Halide::Type t)
         : key_(k), type_(t), dimensions_(0), index_(-1), node_id_(), param_info_(new ParamContainer(k, t)) {}

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& k, Halide::Type t, int32_t d)
         : key_(k), type_(t), dimensions_(d), index_(-1), node_id_(), param_info_(new ParamContainer(k, t, d)) {}

     std::string key() const { return key_; }
     std::string& key() { return key_; }

     Halide::Type type() const { return type_; }
     Halide::Type& type() { return type_; }

     int32_t dimensions() const { return dimensions_; }
     int32_t& dimensions() { return dimensions_; }

     int32_t index() const { return index_; }
     int32_t& index() { return index_; }

     std::string node_id() const { return node_id_; }
     std::string& node_id() { return node_id_; }

     std::shared_ptr<ParamContainer> param_info() const { return param_info_; }
     std::shared_ptr<ParamContainer>& param_info() { return param_info_; }

     Halide::Expr expr() const {
         return param_info_->expr();
     }

     Halide::Func func() const {
         return param_info_->func();
     }

     template<typename t>
     void set_to_param_map(Halide::ParamMap& pm, t v) {
         param_info_->set_to_param_map(pm, v);
     }

     template<typename t>
     void set_to_param_map(Halide::ParamMap& pm, Halide::Buffer<t> &buf) {
         param_info_->set_to_param_map(pm, buf);
     }

     bool bound() const {
         return !node_id_.empty();
     }


     void set_index(int idx) {
         this->index_ = idx;
     }


    /**
     * Overloaded operator to set the port index and return a reference to the current port. eg. port[0]
     */
     Port operator[](int idx) {
         this->set_index(idx);
         return *this;
     }

private:
    /**
     * This port is bound with some node.
     */
     Port(const std::string& k, const std::string& ni) : key_(k), type_(), index_(-1), dimensions_(0), node_id_(ni), param_info_(nullptr){}

     std::string key_;
     Halide::Type type_;
     int32_t dimensions_;
     int32_t index_;
     std::string node_id_;
     std::shared_ptr<ParamContainer> param_info_;
};

} // namespace ion

#endif // ION_PORT_H
