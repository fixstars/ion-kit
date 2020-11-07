#ifndef ION_BB_DNN_YOLOV4_UTILS_H
#define ION_BB_DNN_YOLOV4_UTILS_H

#include <limits>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct YoloBox {
    float max_conf;
    int max_id;
    float x1, x2, y1, y2;
} YoloBox;

float area(const YoloBox &b) {
    return (b.x2 - b.x1) * (b.y2 - b.y1);
}

float intersection(const YoloBox &a, const YoloBox &b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = x2 - x1;
    const float h = y2 - y1;
    if (w <= 0 || h <= 0) return 0;
    return w * h;
}

float union_(const YoloBox &a, const YoloBox &b) {
    const auto area1 = area(a);
    const auto area2 = area(b);
    const auto inter = intersection(a, b);
    return area1 + area2 - inter;
}

std::vector<YoloBox> post_processing(const float *boxes, const float *confs, const int num, const int num_classes, const float conf_thresh = 0.4, const float nms_thresh = 0.4) {
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

    std::vector<YoloBox> yolo_boxes;

    for (int i = 0; i < num; i++) {
        const auto max_conf = max_confs[i];
        const auto max_id = max_ids[i];

        if (max_conf > conf_thresh) {
            YoloBox b;
            b.max_conf = max_conf;
            b.max_id = max_id;
            b.x1 = boxes[i * 4];
            b.y1 = boxes[i * 4 + 1];
            b.x2 = boxes[i * 4 + 2];
            b.y2 = boxes[i * 4 + 3];
            yolo_boxes.push_back(b);
        }
    }

    std::vector<bool> is_valid(yolo_boxes.size(), true);

    for (int i = 0; i < yolo_boxes.size(); i++) {
        if (!is_valid[i]) continue;
        const auto main = yolo_boxes[i];
        for (int j = i + 1; j < yolo_boxes.size(); j++) {
            if (!is_valid[j]) continue;
            const auto other = yolo_boxes[j];
            const auto iou = intersection(main, other) / union_(main, other);
            is_valid[j] = iou <= nms_thresh;
        }
    }

    std::vector<YoloBox> predicted_boxes;
    for (int i = 0; i < yolo_boxes.size(); i++) {
        if (is_valid[i]) predicted_boxes.push_back(yolo_boxes[i]);
    }

    return predicted_boxes;
}

cv::Mat copy_with_boxes(const cv::Mat &frame, const std::vector<YoloBox> &boxes, const int h, const int w) {
    cv::Mat new_frame;
    frame.copyTo(new_frame);

    static const std::pair<const char *, cv::Scalar> label_color_map[] = {
        {"person", cv::Scalar(111, 221, 142)},
        {"bicycle", cv::Scalar(199, 151, 121)},
        {"car", cv::Scalar(145, 233, 34)},
        {"motorbike", cv::Scalar(110, 131, 63)},
        {"aeroplane", cv::Scalar(251, 141, 195)},
        {"bus", cv::Scalar(136, 137, 194)},
        {"train", cv::Scalar(114, 27, 34)},
        {"truck", cv::Scalar(172, 221, 65)},
        {"boat", cv::Scalar(7, 30, 178)},
        {"traffic light", cv::Scalar(31, 28, 230)},
        {"fire hydrant", cv::Scalar(66, 214, 26)},
        {"stop sign", cv::Scalar(133, 39, 182)},
        {"parking meter", cv::Scalar(33, 20, 48)},
        {"bench", cv::Scalar(174, 253, 25)},
        {"bird", cv::Scalar(212, 160, 0)},
        {"cat", cv::Scalar(88, 78, 255)},
        {"dog", cv::Scalar(183, 35, 220)},
        {"horse", cv::Scalar(118, 157, 99)},
        {"sheep", cv::Scalar(81, 39, 129)},
        {"cow", cv::Scalar(253, 97, 253)},
        {"elephant", cv::Scalar(208, 170, 203)},
        {"bear", cv::Scalar(209, 175, 193)},
        {"zebra", cv::Scalar(43, 32, 163)},
        {"giraffe", cv::Scalar(246, 162, 213)},
        {"backpack", cv::Scalar(150, 199, 251)},
        {"umbrella", cv::Scalar(225, 165, 42)},
        {"handbag", cv::Scalar(56, 139, 51)},
        {"tie", cv::Scalar(235, 82, 61)},
        {"suitcase", cv::Scalar(219, 129, 248)},
        {"frisbee", cv::Scalar(120, 74, 139)},
        {"skis", cv::Scalar(164, 201, 240)},
        {"snowboard", cv::Scalar(238, 83, 85)},
        {"sports ball", cv::Scalar(134, 120, 102)},
        {"kite", cv::Scalar(166, 149, 183)},
        {"baseball bat", cv::Scalar(243, 13, 18)},
        {"baseball glove", cv::Scalar(56, 182, 85)},
        {"skateboard", cv::Scalar(117, 60, 48)},
        {"surfboard", cv::Scalar(109, 204, 30)},
        {"tennis racket", cv::Scalar(245, 221, 109)},
        {"bottle", cv::Scalar(74, 27, 47)},
        {"wine glass", cv::Scalar(229, 166, 29)},
        {"cup", cv::Scalar(158, 219, 241)},
        {"fork", cv::Scalar(95, 153, 84)},
        {"knife", cv::Scalar(218, 183, 12)},
        {"spoon", cv::Scalar(146, 37, 136)},
        {"bowl", cv::Scalar(63, 212, 25)},
        {"banana", cv::Scalar(174, 9, 96)},
        {"apple", cv::Scalar(180, 104, 193)},
        {"sandwich", cv::Scalar(160, 117, 33)},
        {"orange", cv::Scalar(224, 42, 115)},
        {"broccoli", cv::Scalar(9, 49, 96)},
        {"carrot", cv::Scalar(124, 213, 203)},
        {"hot dog", cv::Scalar(187, 193, 196)},
        {"pizza", cv::Scalar(57, 25, 171)},
        {"donut", cv::Scalar(189, 74, 145)},
        {"cake", cv::Scalar(73, 119, 11)},
        {"chair", cv::Scalar(37, 253, 178)},
        {"sofa", cv::Scalar(83, 223, 49)},
        {"pottedplant", cv::Scalar(111, 216, 113)},
        {"bed", cv::Scalar(167, 152, 203)},
        {"diningtable", cv::Scalar(99, 144, 184)},
        {"toilet", cv::Scalar(100, 204, 167)},
        {"tvmonitor", cv::Scalar(203, 87, 87)},
        {"laptop", cv::Scalar(139, 188, 41)},
        {"mouse", cv::Scalar(23, 84, 185)},
        {"remote", cv::Scalar(79, 160, 205)},
        {"keyboard", cv::Scalar(63, 7, 87)},
        {"cell phone", cv::Scalar(197, 255, 152)},
        {"microwave", cv::Scalar(199, 123, 207)},
        {"oven", cv::Scalar(211, 86, 200)},
        {"toaster", cv::Scalar(232, 184, 61)},
        {"sink", cv::Scalar(226, 254, 156)},
        {"refrigerator", cv::Scalar(195, 207, 141)},
        {"book", cv::Scalar(238, 101, 223)},
        {"clock", cv::Scalar(24, 84, 233)},
        {"vase", cv::Scalar(39, 104, 233)},
        {"scissors", cv::Scalar(49, 115, 78)},
        {"teddy bear", cv::Scalar(199, 193, 20)},
        {"hair drier", cv::Scalar(156, 85, 108)},
        {"toothbrush", cv::Scalar(189, 59, 8)},
    };

    for (const auto &b : boxes) {
        const auto lc = label_color_map[b.max_id];
        const auto label = lc.first;
        const auto color = lc.second;
        const int x1 = b.x1 * w;
        const int y1 = b.y1 * h;
        const int x2 = b.x2 * w;
        const int y2 = b.y2 * h;
        const cv::Point2d p1(x1, y1);
        const cv::Point2d p2(x2, y2);
        cv::rectangle(new_frame, p1, p2, color);
        cv::putText(new_frame, label, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);
    }

    return new_frame;
}

#endif
