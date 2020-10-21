#ifndef ION_BB_STD_BB_H
#define ION_BB_STD_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace std {

template<typename T>
class Producerx2 : public BuildingBlock<Producerx2<T>> {
public:
    GeneratorInput<uint8_t> v{"v", 0};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 2};
    void generate() {
        output(x, y) = v;
    }

    void schedule() {
        output.compute_root();
    }
private:
    Halide::Var x, y;
};

using ProducerU16x2 = Producerx2<uint16_t>;

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::ProducerU16x2, std_producer_u16x2);

namespace ion {
namespace bb {
namespace std {

template<typename T>
class Producerx3 : public BuildingBlock<Producerx3<T>> {
public:
    GeneratorInput<uint8_t> v{"v", 0};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 3};
    void generate() {
        output(c, x, y) = v;
    }

    void schedule() {
        output.compute_root();
    }
private:
    Halide::Var c, x, y;
};

using ProducerU8x3 = Producerx3<uint8_t>;

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::ProducerU8x3, std_producer_u8x3);

namespace ion {
namespace bb {
namespace std {

template<typename T, int D>
class Gain : public BuildingBlock<Gain<T, D>> {
public:
    GeneratorParam<T> gain{"gain", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        using namespace Halide;
        output(_) = input(_) * gain;
    }

    void schedule() {
        output.compute_root();
    }
};

using GainU8x3 = Gain<uint8_t, 3>;
using GainU16x2 = Gain<uint16_t, 2>;

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::GainU8x3, std_gain_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::std::GainU16x2, std_gain_u16x2);

namespace ion {
namespace bb {
namespace std {

class Mono2BGR : public BuildingBlock<Mono2BGR> {
public:
    GeneratorInput<Halide::Func> input{"input", UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", UInt(8), 3};

    void generate() {
        output(c, x, y) = cast<uint8_t>(input(x, y) >> 8);
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var c, x, y;
};

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::Mono2BGR, std_mono2bgr);

namespace ion {
namespace bb {
namespace std {

class RGB2BGR : public BuildingBlock<RGB2BGR> {
public:
    GeneratorInput<Halide::Func> input{"input", UInt(8), 3};
    GeneratorOutput<Halide::Func> output{"output", UInt(8), 3};

    void generate() {
        using namespace Halide;
        output(c, x, y) = select(c == 0, input(2, x, y),
                                 c == 1, input(1, x, y),
                                         input(0, x, y));
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var c, x, y;
};

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::RGB2BGR, std_rgb2bgr);

namespace ion {
namespace bb {
namespace std {

class BayerBG2BGR : public BuildingBlock<BayerBG2BGR> {
public:
    GeneratorInput<Halide::Func> input{ "input", UInt(16), 2 };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorOutput<Halide::Func> output{ "output", UInt(8), 3 };

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Func in = BoundaryConditions::repeat_edge(input, { {0, width}, {0, height} });

        Expr is_b  = (x % 2 == 0) && (y % 2 == 0);
        Expr is_gr = (x % 2 == 1) && (y % 2 == 0);
        Expr is_r  = (x % 2 == 0) && (y % 2 == 1);
        Expr is_gb = (x % 2 == 1) && (y % 2 == 1);

        Expr self = in(x, y);
        Expr hori = (in(x - 1, y) + in(x + 1, y)) / 2;
        Expr vert = (in(x, y - 1) + in(x, y + 1)) / 2;
        Expr latt = (in(x - 1, y) + in(x + 1, y) + in(x, y - 1) + in(x, y + 1)) / 4;
        Expr diag = (in(x - 1, y - 1) + in(x + 1, y - 1) + in(x - 1, y + 1) + in(x + 1, y + 1)) / 4;

        // Assumes RAW has 12 bit resolutions
        Expr r = cast<uint8_t>(select(is_r, self, is_gr, hori, is_gb, vert, diag) >> 4);
        Expr g = cast<uint8_t>(select(is_r, latt, is_gr, diag, is_gb, diag, latt) >> 4);
        Expr b = cast<uint8_t>(select(is_r, diag, is_gr, vert, is_gb, hori, self) >> 4);

        output(c, x, y) = select(c == 0, b, c == 1, g, r);
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var c, x, y;
};

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::BayerBG2BGR, std_bayerbg2bgr);

namespace ion {
namespace bb {
namespace std {

class DownScale : public BuildingBlock<DownScale> {
public:
    GeneratorParam<int32_t> h{ "horizontal", 1 };
    GeneratorParam<int32_t> v{ "vertical", 1 };
    GeneratorInput<Halide::Func> input{ "input", UInt(8), 3 };
    GeneratorOutput<Halide::Func> output{ "output", UInt(8), 3 };

    void generate() {
        using namespace Halide;
        output(c, x, y) = input(c, x * h, y * v);
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var c, x, y;
};

} // std
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::std::DownScale, std_downscale);

#endif
