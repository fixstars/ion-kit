#include "ion/port.h"

#include "uuid/sole.hpp"
#include "log.h"

namespace ion {

Port::Impl::Impl()
    : id(sole::uuid4().str()), pred_chan{"", ""}, succ_chans{}, type(), dimensions(-1)
{
}

Port::Impl::Impl(const std::string& pid, const std::string& pn, const Halide::Type& t, int32_t d, const GraphID & gid)
    : id(sole::uuid4().str()), pred_chan{pid, pn}, succ_chans{}, type(t), dimensions(d), graph_id(gid)
{
    \
    params[0] = Halide::Internal::Parameter(type, dimensions != 0, dimensions, argument_name(pid, pn, 0, gid.value()));
}

void Port::determine_succ(const std::string& nid, const std::string& old_pn, const std::string& new_pn) {
    auto it = std::find(impl_->succ_chans.begin(), impl_->succ_chans.end(), Channel{nid, old_pn});
    if (it == impl_->succ_chans.end()) {
        log::error("fixme");
        throw std::runtime_error("fixme");
    }

    log::debug("Determine free port {} as {} on Node {}", old_pn, new_pn, nid);
    impl_->succ_chans.erase(it);
    impl_->succ_chans.insert(Channel{nid, new_pn});
}

std::tuple<std::shared_ptr<Port::Impl>, bool> Port::find_impl(const std::string& id) {
    static std::unordered_map<std::string, std::shared_ptr<Impl>> impls;
    static std::mutex mutex;
    std::scoped_lock lock(mutex);
    bool found = true;
    if (!impls.count(id)) {
        impls[id] = std::make_shared<Impl>();
        found = false;
    }
    log::debug("Port {} is {}found", id, found ? "" : "not ");
    return std::make_tuple(impls[id], found);
}

} // namespace ion
