#include "ion/port.h"

#include "log.h"

namespace ion {

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


} // namespace ion
