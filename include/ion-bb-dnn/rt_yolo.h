#ifndef ION_BB_DNN_YOLOV4_UTILS_H
#define ION_BB_DNN_YOLOV4_UTILS_H

#include <limits>
#include <vector>

#include "rt_util.h"

namespace ion {
namespace bb {
namespace dnn {

std::vector<DetectionBox> yolo_post_processing(const float *boxes, const float *confs, const int num, const int num_classes, const float conf_thresh = 0.4, const float nms_thresh = 0.4) {
    std::vector<float> max_confs(num, std::numeric_limits<float>::min());
    std::vector<int> max_ids(num);

    for (int i = 0; i < num; i++)
        for (int j = 0; j < num_classes; j++) {
            const auto src_idx = i * num_classes + j;
            const auto dst_idx = i;
            if (confs[src_idx] > max_confs[dst_idx]) {
                max_confs[dst_idx] = confs[src_idx];
                max_ids[dst_idx] = j;
            }
        }

    std::vector<DetectionBox> all_boxes;

    for (int i = 0; i < num; i++) {
        const auto max_conf = max_confs[i];
        const auto max_id = max_ids[i];

        if (max_conf > conf_thresh) {
            DetectionBox b;
            b.max_conf = max_conf;
            b.max_id = max_id;
            b.x1 = boxes[i * 4];
            b.y1 = boxes[i * 4 + 1];
            b.x2 = boxes[i * 4 + 2];
            b.y2 = boxes[i * 4 + 3];
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
