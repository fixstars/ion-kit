#ifndef ION_BB_DNN_RT_OPENCV_H
#define ION_BB_DNN_RT_OPENCV_H

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

     void classify(
         halide_buffer_t *in_img,
         halide_buffer_t *in_md,
         int32_t output_size,
         halide_buffer_t *out
         ) {
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
     }

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

    json j = json::parse(reinterpret_cast<const char*>(in_md->host));

    auto boxes = j.get<std::vector<DetectionBox>>();

    classifier.classify(in_img, in_md, output_size, out);

    //Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
    //    Mat face = frame(rec); // take the ROI of box on the frame

    //    Mat blob;
    //    blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
    //    genderNet.setInput(blob);
    //    // string gender_preds;
    //    vector<float> genderPreds = genderNet.forward();
    //    // printing gender here
    //    // find max element index
    //    // distance function does the argmax() work in C++
    //    int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
    //    string gender = genderList[max_index_gender];

    out->host[0] = '{';
    out->host[1] = '}';
    out->host[2] = 0;
    return;
}


} // cv
} // dnn
} // bb
} // ion

#endif // ION_BB_DNN_RT_OPENCV_H
