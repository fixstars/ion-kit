#ifndef ION_BB_DEMO_BB_SGM_H
#define ION_BB_DEMO_BB_SGM_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace demo {

namespace {

Halide::Func addCosts(const std::vector<Halide::Func> &cost_funcs) {
    using namespace Halide;

    Var d, x, y;

    Expr cost = cast<uint16_t>(0);
    for (auto f : cost_funcs) {
        cost += f(d, x, y);
    }
    Func f;
    f(d, x, y) = cost;
    return f;
}

Halide::Func disparity(Halide::Func cost, int32_t disp) {
    using namespace Halide;

    Var x("x"), y("y");
    RDom r(0, disp);

    Expr e = cost(r, x, y);

    Func g("argmin");
    g(x, y) = Tuple(0, e.type().max());
    g(x, y) = tuple_select(e < g(x, y)[1], Tuple(r, e), g(x, y));

    g.unroll(x).unroll(y).update().unroll(x).unroll(y).unroll(r[0]);

    Func f("disparity");
    f(x, y) = cast<uint8_t>(g(x, y)[0] * (UINT8_MAX + 1) / disp);

    return f;
}

Halide::Func census(Halide::Func input, int32_t width, int32_t height, int32_t hori, int32_t vert) {
    using namespace Halide;

    Var x("x"), y("y");
    const int32_t radh = hori / 2, radv = vert / 2;

    Func f("census");
    RDom r(-radh, hori, -radv, vert);
    Expr rX = radh - r.x;
    Expr rY = radv - r.y;
    Expr vrX = select(rX > radh, rX - 1, rX);
    Expr vrY = select(rY > radh, rY - 1, rY);
    Expr shift = cast<uint64_t>(vrY * (hori - 1) + vrX);
    Expr inside = x >= radh && x < width - radh && y >= radh && y < height - radh;

    Func in = BoundaryConditions::constant_exterior(input, 0, 0, width, 0, height);

    f(x, y) += select(inside,
                      select(r.x == 0 || r.y == 0,
                             cast<uint64_t>(0),
                             select(in(x, y) > in(x + r.x, y + r.y),
                                    cast<uint64_t>(1) << shift,
                                    cast<uint64_t>(0))),
                      cast<uint64_t>(0));

    f.update().unroll(r.x).unroll(r.y);

    return f;
}

Halide::Func census(Halide::Func input, int32_t width, int32_t height) {
    return census(input, width, height, 9, 7);
}

Halide::Func matchingCost(Halide::Func left, Halide::Func right, int32_t width, int32_t height) {
    using namespace Halide;

    Var x("x"), y("y"), d("d");

    Func f;
    Func r = BoundaryConditions::constant_exterior(right, 0, 0, width, 0, height);
    f(d, x, y) = cast<uint8_t>(popcount(left(x, y) ^ select((x - d) > 0, r(x - d, y), cast<uint64_t>(0))));

    return f;
}

Halide::Func scanCost(Halide::Func cost, int32_t width, int32_t height, int32_t disp, int32_t RX, int32_t RY, bool forward, const std::string &prefix, const Halide::Target &target) {
    using namespace Halide;

    Var x("x"), y("y"), d("d");

    const auto name = prefix + "lcost_" + std::to_string(RX) + "_" + std::to_string(RY);

    Func cost_mid("cost_mid"), cost_wrap(prefix + "cost_wrap"), lcost(name), output_mid("output_mid"), output(prefix + "cost_output");

    cost_mid(d, x, y) = cost(d, x, y);
    cost_wrap(d, x, y) = cost_mid(d, x, y);

    lcost(d, x, y) = Tuple{undef<uint16_t>(), undef<uint16_t>()};

    Expr PENALTY1 = cast<uint16_t>(20);
    Expr PENALTY2 = cast<uint16_t>(100);

    RDom r(0, disp, 0, width, 0, height);
    RVar rd = r[0];
    RVar rx = r[1];
    RVar ry = r[2];

    Expr bx = forward ? rx : width - 1 - rx;
    Expr by = forward ? ry : height - 1 - ry;
    Expr px = bx - RX;
    Expr py = by - RY;
    Expr inside = py >= 0 && py < height && px >= 0 && px < width;
    Expr outside_x = px < 0 || px >= width;
    Expr outside_y = py < 0 || py >= height;

    Expr minCost = select(outside_x, 0, outside_y, 0, likely(lcost(disp - 1, px, py)[1]));

    Expr cost0 = select(outside_x, 0, outside_y, 0, lcost(rd, px, py)[0]);
    Expr cost1 = select(rd - 1 < 0, INT32_MAX, outside_x, 0, outside_y, 0, likely(lcost(rd - 1, px, py)[0]) + PENALTY1);
    Expr cost2 = select(rd + 1 >= disp, INT32_MAX, outside_x, 0, outside_y, 0, likely(lcost(rd + 1, px, py)[0]) + PENALTY1);
    Expr cost3 = minCost + PENALTY2;
    Expr pen = min(min(cost0, cost1), min(cost2, cost3));

    Expr newCost = select(inside,
                          cast<uint16_t>(cost_wrap(rd, bx, by) + pen - minCost),
                          cast<uint16_t>(cost_wrap(rd, bx, by)));

    lcost(rd, bx, by) = Tuple(
        newCost,
        cast<uint16_t>(select(rd - 1 < 0,
                              newCost,
                              likely(lcost(rd - 1, bx, by)[1]) > newCost, newCost, likely(lcost(rd - 1, bx, by)[1]))));

    output_mid(d, x, y) = lcost(d, x, y)[0];
    output(d, x, y) = output_mid(d, x, y);

    lcost.compute_root();

    Var tx("tx"), ty("ty");
    RVar rtx("rtx"), rty("rty");

    if (RX == 0 && RY != 0) {
        if (target.has_gpu_feature()) {
            lcost.reorder_storage(x, y, d);
            lcost.update().gpu_tile(rx, rtx, 32).reorder(rd, ry, rtx, rx).unroll(rd);
            output.compute_root().reorder_storage(x, y, d).gpu_tile(x, y, tx, ty, 32, 8);
        } else {
            lcost.update().unroll(rd).parallel(rx).allow_race_conditions();
        }
    } else if (RX != 0 && RY == 0) {
        if (target.has_gpu_feature()) {
            cost_wrap.compute_root().tile(y, x, ty, tx, 32, 32).reorder(ty, tx, y, x, d).reorder_storage(y, x, d).gpu_blocks(y, x, d).gpu_threads(tx).gpu_lanes(ty);
            cost_mid.compute_at(cost_wrap, y).gpu_threads(y).gpu_lanes(x);
            lcost.reorder_storage(y, x, d);
            lcost.update().split(ry, ry, rty, 32).reorder(rd, rx, rty, ry).gpu_blocks(ry).gpu_lanes(rty).unroll(rd);
            output.compute_root().tile(x, y, tx, ty, 32, 32).reorder(tx, ty, x, y, d).reorder_storage(x, y, d).gpu_blocks(x, y, d).gpu_threads(ty).gpu_lanes(tx);
            output_mid.compute_at(output, x).reorder(y, x).gpu_threads(x).gpu_lanes(y);
        } else {
            lcost.update().unroll(rd).parallel(ry).allow_race_conditions();
        }
    } else {
        if (target.has_gpu_feature()) {
            output.compute_root().reorder_storage(x, y, d).gpu_tile(x, y, tx, ty, 32, 8);
        }
    }

    return output;
}

Halide::Func scanCost(Halide::Func cost, int32_t width, int32_t height, int32_t disp, int32_t RX, int32_t RY, bool forward, const Halide::Target &target) {
    return scanCost(cost, width, height, disp, RX, RY, forward, "", target);
}

Halide::Func scanCost(Halide::Func cost, int32_t width, int32_t height, int32_t disp, int32_t RX, int32_t RY, bool forward) {
    return scanCost(cost, width, height, disp, RX, RY, forward, "", Halide::Target("host"));
}

Halide::Func semi_global_matching(Halide::Func in_l, Halide::Func in_r, int32_t width, int32_t height, int32_t disp, const std::string &prefix, const Halide::Target &target) {
    using namespace Halide;

    Var d, x, y;
    Func census_l(prefix + "census_left");
    census_l(x, y) = census(in_l, width, height)(x, y);

    Func census_r(prefix + "census_right");
    census_r(x, y) = census(in_r, width, height)(x, y);

    Func mc(prefix + "matching_cost");
    mc(d, x, y) = matchingCost(census_l, census_r, width, height)(d, x, y);

    Func sc_u(prefix + "scan_cost_u");
    sc_u(d, x, y) = scanCost(mc, width, height, disp, 0, 1, true, prefix, target)(d, x, y);

    Func sc_l(prefix + "scan_cost_l");
    sc_l(d, x, y) = scanCost(mc, width, height, disp, 1, 0, true, prefix, target)(d, x, y);

    Func sc_r(prefix + "scan_cost_r");
    sc_r(d, x, y) = scanCost(mc, width, height, disp, -1, 0, false, prefix, target)(d, x, y);

    Func sc_d(prefix + "scan_cost_d");
    sc_d(d, x, y) = scanCost(mc, width, height, disp, 0, -1, false, prefix, target)(d, x, y);

    Func ac(prefix + "add_cost");
    ac(d, x, y) = addCosts({sc_u, sc_l, sc_r, sc_d})(d, x, y);

    Func dis(prefix + "disparity");
    dis(x, y) = disparity(ac, disp)(x, y);

    if (target.has_gpu_feature()) {
        Var tx, ty;
        const int thread_width = 32;
        const int thread_height = 8;
        census_l.compute_root().gpu_tile(x, y, tx, ty, thread_width, thread_height);
        census_r.compute_root().gpu_tile(x, y, tx, ty, thread_width, thread_height);
        mc.compute_root().reorder_storage(x, y, d).unroll(d).gpu_tile(x, y, tx, ty, thread_width, thread_height);
        // sc is GPU scheduled into scanCost function.
        dis.compute_root().gpu_tile(x, y, tx, ty, thread_width, thread_height);
    } else {
        census_l.compute_root().parallel(y);
        census_r.compute_root().parallel(y);
        mc.compute_root().parallel(y);
        sc_u.compute_root();
        sc_l.compute_root();
        sc_r.compute_root();
        sc_d.compute_root();
        dis.compute_root().parallel(y);
    }

    return dis;
}

}  // namespace

class Census : public ion::BuildingBlock<Census> {
public:
    // TODO: Write suitable description
    GeneratorParam<std::string> gc_title{"gc_title", "Census Transform"};
    GeneratorParam<std::string> gc_description{"gc_description", "Census transform."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint64_t>(), 2};

    void generate() {
        output(Halide::_) = census(input, width, height)(Halide::_);
    }

    void schedule() {
        const auto t = get_target();
        if (t.has_gpu_feature()) {
            const auto x = output.args().at(0);
            const auto y = output.args().at(1);
            Halide::Var tx, ty;
            output.compute_root().gpu_tile(x, y, tx, ty, 32, 8);
        } else {
            output.compute_root();
        }
    }
};

class MatchingCost : public ion::BuildingBlock<MatchingCost> {
public:
    // TODO: Write suitable description
    GeneratorParam<std::string> gc_title{"gc_title", "Matching cost"};
    GeneratorParam<std::string> gc_description{"gc_description", "Matching cost."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.disp), parseInt(v.width), parseInt(v.height)]}; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};

    GeneratorParam<int32_t> disp{"disp", 16};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};

    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint64_t>(), 2};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint64_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        output(Halide::_) = matchingCost(input0, input1, width, height)(Halide::_);
    }

    void schedule() {
        const auto t = get_target();
        if (t.has_gpu_feature()) {
            const auto d = output.args().at(0);
            const auto x = output.args().at(1);
            const auto y = output.args().at(2);
            Halide::Var tx, ty;
            output.compute_root().unroll(d).reorder_storage(x, y, d).gpu_tile(x, y, tx, ty, 32, 8);
        } else {
            output.compute_root();
        }
    }
};

class ScanCost : public ion::BuildingBlock<ScanCost> {
public:
    // TODO: Write suitable description
    GeneratorParam<std::string> gc_title{"gc_title", "Scan cost"};
    GeneratorParam<std::string> gc_description{"gc_description", "Scan cost."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};

    GeneratorParam<int32_t> disp{"disp", 16};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 3};

    GeneratorParam<int32_t> direction_x{"dx", 0};
    GeneratorParam<int32_t> direction_y{"dy", 0};
    // TODO: forward

    void generate() {
        output(Halide::_) = scanCost(input, width, height, disp, direction_x, direction_y, true, get_target())(Halide::_);
    }

    void schedule() {
        const auto t = get_target();
        if (!t.has_gpu_feature()) {
            output.compute_root();
        }
    }
};

// // TODO: Use Halide::Func[] and handle variable size for GeneratorInput
class AddCost4 : public ion::BuildingBlock<AddCost4> {
public:
    // TODO: Write suitable description
    GeneratorParam<std::string> gc_title{"gc_title", "Add cost"};
    GeneratorParam<std::string> gc_description{"gc_description", "Add cost."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0 }; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};

    GeneratorParam<int32_t> disp{"disp", 16};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> num{"num", 1};

    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint16_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint16_t>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<uint16_t>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<uint16_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 3};

    void generate() {
        output(Halide::_) = addCosts({input0, input1, input2, input3})(Halide::_);
    }

    void schedule() {
        const auto t = get_target();
        if (t.has_gpu_feature()) {
            const auto d = output.args().at(0);
            const auto x = output.args().at(1);
            const auto y = output.args().at(2);
            Halide::Var tx, ty;
            output.compute_root().reorder_storage(x, y, d).gpu_tile(x, y, tx, ty, 32, 8);
        } else {
            output.compute_root();
        }
    }
};

class Disparity : public ion::BuildingBlock<Disparity> {
public:
    // TODO: Write suitable description
    GeneratorParam<std::string> gc_title{"gc_title", "Disparity"};
    GeneratorParam<std::string> gc_description{"gc_description", "Disparity."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)]}; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};

    GeneratorParam<int32_t> disp{"disp", 16};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint16_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 2};

    void generate() {
        output(Halide::_) = disparity(input, disp)(Halide::_);
    }

    void schedule() {
        const auto t = get_target();
        if (t.has_gpu_feature()) {
            const auto x = output.args().at(0);
            const auto y = output.args().at(1);
            Halide::Var tx, ty;
            output.compute_root().gpu_tile(x, y, tx, ty, 32, 8);
        } else {
            output.compute_root();
        }
    }
};

class SGM : public ion::BuildingBlock<SGM> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Stereo Matching"};
    GeneratorParam<std::string> gc_description{"gc_description", "This calculates disparity from stereo image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "image,stereo"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input_l }; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> disp{"disp", 16};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};

    GeneratorInput<Halide::Func> input_l{"input_l", Halide::type_of<uint8_t>(), 2};
    GeneratorInput<Halide::Func> input_r{"input_r", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 2};

    void generate() {
        output(Halide::_) = semi_global_matching(input_l, input_r, width, height, disp, gc_prefix, get_target())(Halide::_);
    }

private:
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Census, demo_census);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MatchingCost, demo_matching_cost);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ScanCost, demo_scan_cost);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::AddCost4, demo_add_cost4);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Disparity, demo_disparity);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::SGM, demo_sgm);

#endif
