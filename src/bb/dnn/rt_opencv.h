#ifndef ION_BB_DNN_RT_OPENCV_H
#define ION_BB_DNN_RT_OPENCV_H

#include <unordered_map>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "rt_util.h"
#include "httplib.h"
#include "picosha2.h"

namespace ion {
namespace bb {
namespace dnn {
namespace opencv {

using json = nlohmann::json;

using ClassifyResult = std::unordered_map<std::string, uint32_t>;

class Classifier {
 public:
     static Classifier& get_instance(const std::string& uuid, const std::string& model_root_url, const std::string& cache_root) {
         static std::map<std::string, std::unique_ptr<Classifier>> map_;
         Classifier *c;
         if (map_.count(uuid) == 0) {
             map_[uuid] = std::unique_ptr<Classifier>(new Classifier(model_root_url, cache_root));
         }
         return *map_[uuid].get();
     }

     ClassifyResult classify(
         const cv::Mat& image,
         const std::vector<DetectionBox>& boxes) {
         ClassifyResult result;

         const int PeopleNetClassID_Face = 2;
         const cv::Scalar MODEL_MEAN_VALUES = cv::Scalar(78.4263377603, 87.7689143744, 114.895847746);

         result["Male"] = 0;
         result["Female"] = 0;

         for (auto b: boxes) {
             if (b.class_id == PeopleNetClassID_Face) {
                 if (b.x2-b.x1 < 100 || b.y2-b.y1 < 100) {
                     continue;
                 }
                 cv::Mat face(image, cv::Rect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1));
                 cv::normalize(face, face, 0, 255, cv::NORM_MINMAX, CV_8UC3);
                 cv::cvtColor(face, face, cv::COLOR_RGB2BGR);

                 cv::Mat blob = cv::dnn::blobFromImage(face, 1, cv::Size(227, 227), MODEL_MEAN_VALUES, false);
                 net_.setInput(blob);
                 // // string gender_preds;
                 std::vector<float> genderPreds = net_.forward();
                 // // printing gender here
                 // // find max element index
                 // // distance function does the argmax() work in C++
                 const char *genderList[] = {"Male", "Female"};
                 int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
                 std::string gender = genderList[max_index_gender];
                 result[gender]++;
             }
         }
         return result;
     }

 private:

     std::string cache_load(const std::string& model_root_url, const std::string& file_name, const std::string& cache_root) {
         const std::string url = model_root_url + file_name;

         std::vector<unsigned char> hash(picosha2::k_digest_size);
         picosha2::hash256(url.begin(), url.end(), hash.begin(), hash.end());
         auto hash_str = picosha2::bytes_to_hex_string(hash.begin(), hash.end());

         auto path = cache_root + file_name + "." + hash_str;

         std::ifstream ifs(path, std::ios::binary);
         if (ifs.is_open()) {
             return path;
         }
         ifs.close();

         std::string host_name;
         std::string path_name;
         std::tie(host_name, path_name) = parse_url(url);
         if (host_name.empty() || path_name.empty()) {
             throw std::runtime_error("Invalid URL : " + url);
         }

         httplib::Client cli(host_name.c_str());
         cli.set_follow_location(true);
         auto res = cli.Get(path_name.c_str());
         if (!res || res->status != 200) {
             throw std::runtime_error("Failed to download file: " + url);
         }

         std::ofstream ofs(path, std::ios::binary);
         ofs.write(res->body.c_str(), res->body.size());

         return path;
     }

     Classifier(const std::string& model_root_url, const std::string& cache_root)
     {
         auto model_define = cache_load(model_root_url, "model_define.prototxt", cache_root);
         auto model_weight = cache_load(model_root_url, "model_weight.caffemodel", cache_root);
         net_ = cv::dnn::readNet(model_weight, model_define, "caffe");
     }

     cv::dnn::Net net_;
};

void classify_gender(halide_buffer_t *in_img,
                     halide_buffer_t *in_md,
                     int32_t output_size,
                     const std::string& session_id,
                     const std::string& model_root_url,
                     const std::string& cache_root,
                     halide_buffer_t *out) {

    using namespace cv;
    using json = nlohmann::json;

    auto& classifier = Classifier::get_instance(session_id, model_root_url, cache_root);

    const int width = in_img->dim[1].extent;
    const int height = in_img->dim[2].extent;

    cv::Mat image(height, width, CV_32FC3, in_img->host);
    auto boxes = json::parse(reinterpret_cast<const char*>(in_md->host)).get<std::vector<DetectionBox>>();

    ClassifyResult classify_result = classifier.classify(image, boxes);

    json j = classify_result;
    std::string output_string(j.dump());

    if (output_string.size()+1 >= output_size) {
        throw std::runtime_error("Output buffer size is not sufficient");
    }

    std::memcpy(out->host, output_string.c_str(), output_string.size());
    out->host[output_string.size()] = 0;

    return;
}


} // cv
} // dnn
} // bb
} // ion

#endif // ION_BB_DNN_RT_OPENCV_H
