#ifndef ION_BB_DNN_SSD_UTILS_H
#define ION_BB_DNN_SSD_UTILS_H

#include <limits>
#include <vector>

#include "rt_util.h"

namespace ion {
namespace bb {
namespace dnn {

template<typename T>
struct Vec2d {
    T x;
    T y;

    Vec2d(T x_, T y_)
        : x(x_), y(y_) {
    }
};

struct BoxSpec {
    float scale;
    float aspect_ratio;

    BoxSpec(float scale_, float aspect_ratio_)
        : scale(scale_), aspect_ratio(aspect_ratio_) {
    }
};

struct Anchor {
    Vec2d<float> center;
    Vec2d<float> size;

    Anchor(float center_x, float center_y, float width, float height)
        : center({center_x, center_y}), size(width, height) {
    }
};

std::vector<Anchor> tile_anchors(const Vec2d<int> &grid_shape, const std::vector<BoxSpec> &specs,
                                 const Vec2d<float> &base_anchor_shape,
                                 const Vec2d<float> &anchor_stride, const Vec2d<float> &anchor_offset) {
    std::vector<Anchor> bboxs;

    for (int y = 0; y < grid_shape.y; ++y) {
        const auto center_y = static_cast<float>(y) * anchor_stride.y + anchor_offset.y;
        for (int x = 0; x < grid_shape.x; ++x) {
            const auto center_x = static_cast<float>(x) * anchor_stride.x + anchor_offset.x;
            for (const auto &spec : specs) {
                const auto ratio_sqrt = std::sqrt(spec.aspect_ratio);
                const auto width = spec.scale * ratio_sqrt * base_anchor_shape.x;
                const auto height = spec.scale / ratio_sqrt * base_anchor_shape.y;

                bboxs.emplace_back(center_x, center_y, width, height);
            }
        }
    }

    return bboxs;
}

std::vector<Anchor> build_anchors(int im_width, int im_height,
                                  size_t layer_num,
                                  float min_scale, float max_scale,
                                  const std::vector<float> &aspect_ratios,
                                  const Vec2d<float> &base_anchor_size,
                                  const std::vector<Vec2d<int>> &feature_map_shape_list,
                                  bool reduce_boxes_in_lower_layer,
                                  float interpolated_scale_aspect_ratio) {
    std::vector<float> scales;
    for (auto i = decltype(layer_num)(0); i < layer_num; ++i) {
        const auto v = min_scale + (max_scale - min_scale) * i / (layer_num - 1);
        scales.push_back(v);
    }
    scales.push_back(1.f);

    std::vector<std::vector<BoxSpec>> box_specs_list;

    for (size_t i = 0; i < layer_num; ++i) {
        const auto scale = scales[i];
        const auto scale_next = scales[i + 1];

        std::vector<BoxSpec> layer_box_specs;
        if (i == 0 && reduce_boxes_in_lower_layer) {
            layer_box_specs = {{0.1f, 1.f}, {scale, 2.f}, {scale, 0.5f}};
        } else {
            for (const auto &r : aspect_ratios) {
                layer_box_specs.emplace_back(scale, r);
            }
            if (interpolated_scale_aspect_ratio > 0.f) {
                layer_box_specs.emplace_back(std::sqrt(scale * scale_next), interpolated_scale_aspect_ratio);
            }
        }

        box_specs_list.push_back(layer_box_specs);
    }

    std::vector<Vec2d<float>> anchor_strides;
    for (const auto &shape : feature_map_shape_list) {
        const auto stride_x = 1.f / static_cast<float>(shape.x);
        const auto stride_y = 1.f / static_cast<float>(shape.y);
        anchor_strides.emplace_back(stride_x, stride_y);
    }

    std::vector<Vec2d<float>> anchor_offsets;
    for (const auto &stride : anchor_strides) {
        const auto offset_x = 0.5f * stride.x;
        const auto offset_y = 0.5f * stride.y;
        anchor_offsets.emplace_back(offset_x, offset_y);
    }

    const auto min_im_shape = std::min(im_width, im_height);
    const auto scale_width = min_im_shape / im_width;
    const auto scale_height = min_im_shape / im_height;
    const Vec2d<float> scaled_base_anchor_size = {scale_width * base_anchor_size.x, scale_height * base_anchor_size.y};

    std::vector<Anchor> tiled_anchors;
    for (size_t i = 0; i < layer_num; ++i) {
        const auto anchors = tile_anchors(
            feature_map_shape_list[i], box_specs_list[i], scaled_base_anchor_size,
            anchor_strides[i], anchor_offsets[i]);
        tiled_anchors.insert(tiled_anchors.end(), anchors.begin(), anchors.end());
    }

    return tiled_anchors;
}

float iou(const DetectionBox &box1, const DetectionBox &box2) {
    return intersection(box1, box2) / union_(box1, box2);
}

std::vector<DetectionBox> nms(std::vector<DetectionBox> &boxes, float nms_thresh) {
    std::vector<DetectionBox> kept_boxes;
    std::vector<std::pair<int, float>> sorted_boxes(boxes.size());
    std::vector<bool> box_processed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++) {
        sorted_boxes[i].first = i;
        sorted_boxes[i].second = boxes[i].confidence;
    }
    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
              [](const std::pair<int, float> &ls, const std::pair<int, float> &rs) {
                  return ls.second > rs.second;
              });

    for (auto pair_i = decltype(boxes.size())(0); pair_i < boxes.size(); pair_i++) {
        const auto i = sorted_boxes[pair_i].first;
        if (box_processed[i]) {
            continue;
        }
        kept_boxes.emplace_back(boxes[i]);
        for (auto pair_j = decltype(boxes.size())(pair_i + 1); pair_j < boxes.size(); pair_j++) {
            const auto j = sorted_boxes[pair_j].first;
            if (box_processed[j]) {
                continue;
            }
            if (iou(boxes[i], boxes[j]) >= nms_thresh) {
                box_processed[j] = true;
            }
        }
    }

    return kept_boxes;
}

std::vector<DetectionBox> ssd_post_processing_dnndk(const std::vector<float> &boxes, const std::vector<float> &scores,
                                                    size_t label_num,
                                                    const std::vector<Anchor> &anchors,
                                                    const Anchor &scale_factor,
                                                    float nms_thresh, float conf_thresh,
                                                    size_t top_k_per_class, size_t top_k) {
    const auto scores_sigmoid = sigmoid(scores);

    std::vector<std::vector<DetectionBox>> all_boxes(label_num);

    for (auto i = decltype(anchors.size())(0); i < anchors.size(); ++i) {
        // Compute c-x, c-y, width, height
        const auto &anchor = anchors[i];
        const float center_x = anchor.size.x * boxes[i * 4 + 1] / scale_factor.center.x + anchor.center.x;
        const float center_y = anchor.size.y * boxes[i * 4] / scale_factor.center.y + anchor.center.y;
        const float width = anchor.size.x * std::exp(boxes[i * 4 + 3] / scale_factor.size.x);
        const float height = anchor.size.y * std::exp(boxes[i * 4 + 2] / scale_factor.size.y);

        // Transform box to left-x, top-y, right-x, bottom-y, label, confidence
        DetectionBox box;
        box.x1 = std::min(std::max(center_x - width / 2.f, 0.f), 1.f);   // left-x
        box.y1 = std::min(std::max(center_y - height / 2.f, 0.f), 1.f);  // top-y
        box.x2 = std::min(std::max(center_x + width / 2.f, 0.f), 1.f);   // right-x
        box.y2 = std::min(std::max(center_y + height / 2.f, 0.f), 1.f);  // bottom-y

        // label and confidence
        for (auto label_index = decltype(label_num)(0); label_index < label_num; ++label_index) {
            DetectionBox box_with_label = box;
            const float conf = scores_sigmoid[i * label_num + label_index];
            if (conf > conf_thresh) {
                box_with_label.class_id = label_index;
                box_with_label.confidence = conf;
                all_boxes[label_index].push_back(box_with_label);
            }
        }
    }

    // Apply NMS and get top_k boxes for each class individually
    std::vector<DetectionBox> results;
    for (auto label_index = decltype(all_boxes.size())(1); label_index < all_boxes.size(); ++label_index) {
        auto &one_class_boxes = all_boxes[label_index];
        auto one_class_nms = nms(one_class_boxes, nms_thresh);
        if (top_k_per_class > one_class_nms.size()) {
            top_k_per_class = one_class_nms.size();
        }
        for (size_t j = 0; j < top_k_per_class; ++j) {
            results.emplace_back(one_class_nms[j]);
        }
    }

    // Keep keep_top_k boxes per image
    sort(results.begin(), results.end(),
         [](const DetectionBox &ls, const DetectionBox &rs) { return ls.confidence > rs.confidence; });
    if (results.size() > top_k) {
        results.resize(top_k);
    }

    return results;
}

std::vector<DetectionBox> ssd_post_processing(const float *boxes, const float *classes, const float *scores, const int num, const float conf_thresh = 0.4, const float nms_thresh = 0.4) {
    std::vector<DetectionBox> all_boxes;

    for (int i = 0; i < num; i++) {
        const auto max_conf = scores[i];
        const auto max_id = classes[i];

        if (max_conf > conf_thresh) {
            DetectionBox b;
            b.confidence = max_conf;
            b.class_id = max_id;
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

}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif
