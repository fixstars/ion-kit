#include <ion/ion.h>

using namespace ion;

class Transpose : public BuildingBlock<Transpose> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Camera"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,camera"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3, parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 2};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 2};

    void generate() {
        using namespace Halide;

        input_wrap(x, y) = input(x, y);
        output(x, y) = input_wrap(x, y);

        // output(x, y) = undef<uint8_t>();
        // output(x, y) = select(x == 0, 1, likely(output(x - 1, y) + 1));
    }

    void schedule() {
        if (get_target().has_gpu_feature()) {
            Var tx("tx"), ty("ty");
            output.tile(x, y, tx, ty, 32, 32).gpu_blocks(x, y).gpu_threads(ty).gpu_lanes(tx);
            input_wrap.compute_at(output, x).reorder_storage(y, x).reorder(y, x).gpu_threads(x).gpu_lanes(y);
        }
    }

private:
    Var x{"x"}, y{"y"};
    Func input_wrap{"input_wrap"};
};

ION_REGISTER_BUILDING_BLOCK(Transpose, transpose);

int main() {
    try {
        const std::vector<int> extents{64, 64};
        Halide::Buffer<uint8_t> ibuf(extents), obuf(extents);

        Builder b;
        b.set_target(Halide::get_target_from_environment());
        Port input{"input", Halide::type_of<uint8_t>(), 2};
        auto n = b.add("transpose")(input);

        input.bind(ibuf);
        n["output"].bind(obuf);

        b.run();

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
