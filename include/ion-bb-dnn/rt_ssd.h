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
                                 const Vec2d<float> &anchor_stride, const Vec2d<float> &anchor_offset,
                                 const Anchor &scale_factor) {
    std::vector<Anchor> bboxs;

    for (int y = 0; y < grid_shape.y; ++y) {
        const auto center_y = static_cast<float>(y) * anchor_stride.y + anchor_offset.y / scale_factor.center.y;
        for (int x = 0; x < grid_shape.x; ++x) {
            const auto center_x = static_cast<float>(x) * anchor_stride.x + anchor_offset.x / scale_factor.center.x;
            for (const auto &spec : specs) {
                const auto ratio_sqrt = std::sqrt(spec.aspect_ratio);
                const auto width = spec.scale * ratio_sqrt * base_anchor_shape.x / scale_factor.size.x;
                const auto height = spec.scale / ratio_sqrt * base_anchor_shape.y / scale_factor.size.y;

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
                                  const Anchor &scale_factor,
                                  const std::vector<Vec2d<int>> &feature_map_shape_list,
                                  bool reduce_boxes_in_lower_layer,
                                  float interpolated_scale_aspect_ratio) {
    std::vector<float> scales;
    for (size_t i = 0; i < layer_num; ++i) {
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
            anchor_strides[i], anchor_offsets[i], scale_factor);
        tiled_anchors.insert(tiled_anchors.end(), anchors.begin(), anchors.end());
    }

    return tiled_anchors;
}

std::vector<float> sigmoid(const int8_t *tensor, float scale, int size) {
    std::vector<float> sigmoid_tensor(size);
    for (int i = 0; i < size; ++i) {
        sigmoid_tensor[i] = 1.f / (1.f + std::exp(-(tensor[i] * scale)));
    }

    return sigmoid_tensor;
}

std::vector<DetectionBox> ssd_post_processing_dnndk(const int8_t *boxes, float boxes_scale,
                                                    const int8_t *scores, float scores_scale, int scores_size,
                                                    size_t label_num,
                                                    const std::vector<Anchor> &anchors,
                                                    float nms_thresh, float conf_thresh) {
    const auto scores_sigmoid = sigmoid(scores, scores_scale, scores_size);

    std::vector<DetectionBox> all_boxes;

    for (auto i = decltype(anchors.size())(0); i < anchors.size(); ++i) {
        // Compute c-x, c-y, width, height
        const auto &anchor = anchors[i];
        const float center_x = anchor.size.x * boxes[i * 4 + 1] * boxes_scale + anchor.center.x;
        const float center_y = anchor.size.y / boxes[i * 4] * boxes_scale + anchor.center.y;
        const float width = anchors[i].size.x * std::exp(boxes[i * 4 + 3] * boxes_scale);
        const float height = anchors[i].size.y * std::exp(boxes[i * 4 + 2] * boxes_scale);

        // Transform box to left-x, top-y, right-x, bottom-y, label, confidence
        DetectionBox box;
        box.x1 = std::min(std::max(center_x - width / 2.f, 0.f), 1.f);   // left-x
        box.y1 = std::min(std::max(center_y - height / 2.f, 0.f), 1.f);  // top-y
        box.x2 = std::min(std::max(center_x + width / 2.f, 0.f), 1.f);   // right-x
        box.y2 = std::min(std::max(center_y + height / 2.f, 0.f), 1.f);  // bottom-y

        // label and confidence
        for (auto label_index = decltype(label_num)(0); label_index < label_num; ++label_index) {
            DetectionBox box_with_label = box;
            const float conf = scores[i * label_num + label_index];
            if (conf > conf_thresh) {
                box_with_label.class_id = label_index;
                box_with_label.confidence = conf;
                all_boxes.push_back(box_with_label);
            }
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
