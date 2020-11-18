#ifndef ION_BB_DNN_SSD_UTILS_H
#define ION_BB_DNN_SSD_UTILS_H

#include <limits>
#include <vector>

#include "rt_util.h"

namespace ion {
namespace bb {
namespace dnn {

std::vector<DetectionBox> ssd_post_processing(const float *boxes, const float *classes, const float *scores, const int num, const float conf_thresh = 0.4, const float nms_thresh = 0.4) {
    std::vector<DetectionBox> all_boxes;

    for (int i = 0; i < num; i++) {
        const auto max_conf = scores[i];
        const auto max_id = classes[i];

        if (max_conf > conf_thresh) {
            DetectionBox b;
            b.max_conf = max_conf;
            b.max_id = max_id;
            b.x1 = boxes[i * 4 + 1];
            b.y1 = boxes[i * 4 + 0];
            b.x2 = boxes[i * 4 + 3];
            b.y2 = boxes[i * 4 + 2];
            all_boxes.push_back(b);
        }
    }

    std::vector<bool> is_valid(all_boxes.size(), true);

    for (int i = 0; i < all_boxes.size(); i++) {
        if (!is_valid[i]) continue;
        const auto main = all_boxes[i];
        for (int j = i + 1; j < all_boxes.size(); j++) {
            if (!is_valid[j]) continue;
            const auto other = all_boxes[j];
            const auto iou = intersection(main, other) / union_(main, other);
            is_valid[j] = iou <= nms_thresh;
        }
    }

    std::vector<DetectionBox> detected_boxes;
    for (int i = 0; i < all_boxes.size(); i++) {
        if (is_valid[i]) detected_boxes.push_back(all_boxes[i]);
    }

    return detected_boxes;
}


} // dnn
} // bb
} // ion

#endif
