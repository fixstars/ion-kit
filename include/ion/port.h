#ifndef ION_PORT_H
#define ION_PORT_H

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Halide.h>

namespace ion {

/**
 * Port class is used to create dynamic i/o for each node.
 */
class Port {
 public:
     friend class Node;

     Port()
         : key_(), type_(), dimensions_(0), index_(-1), node_id_() {}

     /**
      * Construct new port for scalar value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the value.
      */
     Port(const std::string& k, Halide::Type t)
         : key_(k), type_(t), dimensions_(0), index_(-1), node_id_() {}

     /**
      * Construct new port for vector value.
      * @arg k: The key of the port which should be matched with BuildingBlock Input/Output name.
      * @arg t: The type of the element value.
      * @arg d: The dimension of the port. The range is 1 to 4.
      */
     Port(const std::string& k, Halide::Type t, int32_t d)
         : key_(k), type_(t), dimensions_(d), index_(-1), node_id_() {}

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

     std::vector<Halide::Internal::Parameter>& params() {
         if (index_ == -1) {
             params_.resize(1, Halide::Internal::Parameter{type_, dimensions_ != 0, dimensions_, key_});
         } else {
             params_.resize(index_+1, Halide::Internal::Parameter{type_, dimensions_ != 0, dimensions_, key_});
         }
         return params_;
     }

     bool is_bound() const {
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
     Port(const std::string& k, const std::string& ni) : key_(k), type_(), index_(-1), dimensions_(0), node_id_(ni) {}

     std::string key_;
     Halide::Type type_;
     int32_t dimensions_;
     int32_t index_;
     std::string node_id_;

     std::vector<Halide::Internal::Parameter> params_;
};

} // namespace ion

#endif // ION_PORT_H
