// model_server.cpp
// 说明：将你给的 Python 代码等价转写为 C++17 单文件版本（去掉可视化/多进程），并用 Crow 提供 /model POST 接口。
// 依赖：crow.h（header-only）、XGBoost C API（libxgboost + xgboost/c_api.h）
// 编译示例：g++ -O2 -std=c++17 model_server.cpp -o model_server -lxgboost -pthread
//
// 默认启动加载：
//   - CSV: zoom_to_P2.csv
//   - XGBoost 模型：train_data_x_new_new_new_no_K_no_feature.json / train_data_y_new_new_new_no_K_no_feature.json
//   - 端口：18080
//
// 你可用环境变量覆盖：
//   CSV_PATH, MODEL_X_PATH, MODEL_Y_PATH, PORT
//
// /model 入参见你给的示例：
// {
//   "proportion":0.06,"n":1.0,"s":0.0,"canvas_width":2560,"canvas_height":1440,"z_max":22.5,
//   "p_offset":0.0,"t_offset":0.0,"mode":0,
//   "pos_now":[20.2,43.1,5.2],
//   "va":[51.2400016784668,30.190000534057617],
//   "boxes":[[667,96,781,209]]
// }
//
// 返回：{ "t": <ms double>, "json": "<json_string>" }

#include "crow.h"

#include <xgboost/c_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <array>


static inline double deg2rad(double d) { return d * M_PI / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / M_PI; }

static inline double round_to_1(double x) { return std::round(x * 10.0) / 10.0; }
static inline double clamp(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

struct Vec2 { double x{0}, y{0}; };
struct Vec3 { double x{0}, y{0}, z{0}; };

struct Box { double x1{0}, y1{0}, x2{0}, y2{0}; };

struct ViewingAngles { double x_deg{0}, y_deg{0}; }; // HFOV, VFOV

// ===================== 全局参数（与 Python 一致） =====================
static constexpr double P_H[4] = {1.89815744e+02, -7.38153896e+00, 2.66237109e+00, -1.07067711e-01};
static constexpr double P_V[4] = {1.34745171e+02, -5.05830455e+00, 3.42660254e+00, -1.33612377e-01};
static constexpr double X_MIN = 1.0;
static constexpr double X_MAX = 23.0;

// 图像尺寸（默认与 Python 一致，运行时按请求覆盖）
static constexpr double PAN_MIN = 0.0;
static constexpr double PAN_MAX = 360.0;
static constexpr double TILT_MIN = -5.0;
static constexpr double TILT_MAX = 90.0;

// 相机/图像参数（与 Python 一致）
static constexpr double SENSOR_X = 5.13;
static constexpr double SENSOR_Y = 2.89;
static constexpr double FOCAL_F = 5.35;

// ===================== 3x3 矩阵（避免引入 Eigen） =====================
struct Mat3 {
    double m[3][3]{};

    static Mat3 identity() {
        Mat3 I;
        I.m[0][0] = I.m[1][1] = I.m[2][2] = 1.0;
        return I;
    }

    Mat3 transpose() const {
        Mat3 t;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) t.m[i][j]=m[j][i];
        return t;
    }

    double det() const {
        const double a = m[0][0], b = m[0][1], c = m[0][2];
        const double d = m[1][0], e = m[1][1], f = m[1][2];
        const double g = m[2][0], h = m[2][1], i = m[2][2];
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    }

    Mat3 inverse() const {
        const double a = m[0][0], b = m[0][1], c = m[0][2];
        const double d = m[1][0], e = m[1][1], f = m[1][2];
        const double g = m[2][0], h = m[2][1], i = m[2][2];

        const double A =  (e*i - f*h);
        const double B = -(d*i - f*g);
        const double C =  (d*h - e*g);
        const double D = -(b*i - c*h);
        const double E =  (a*i - c*g);
        const double F = -(a*h - b*g);
        const double G =  (b*f - c*e);
        const double H = -(a*f - c*d);
        const double I =  (a*e - b*d);

        const double detv = a*A + b*B + c*C;
        if (std::abs(detv) < 1e-15) throw std::runtime_error("Mat3 inverse: singular matrix");

        Mat3 inv;
        inv.m[0][0] = A/detv; inv.m[0][1] = D/detv; inv.m[0][2] = G/detv;
        inv.m[1][0] = B/detv; inv.m[1][1] = E/detv; inv.m[1][2] = H/detv;
        inv.m[2][0] = C/detv; inv.m[2][1] = F/detv; inv.m[2][2] = I/detv;
        return inv;
    }
};

static inline Mat3 matmul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            double s=0;
            for (int k=0;k<3;k++) s += A.m[i][k]*B.m[k][j];
            C.m[i][j]=s;
        }
    }
    return C;
}

static inline Vec3 matvec(const Mat3& A, const Vec3& v) {
    Vec3 r;
    r.x = A.m[0][0]*v.x + A.m[0][1]*v.y + A.m[0][2]*v.z;
    r.y = A.m[1][0]*v.x + A.m[1][1]*v.y + A.m[1][2]*v.z;
    r.z = A.m[2][0]*v.x + A.m[2][1]*v.y + A.m[2][2]*v.z;
    return r;
}

static inline Mat3 Rx_rotate(double b_rad) {
    Mat3 R = Mat3::identity();
    R.m[1][1] =  std::cos(b_rad);
    R.m[1][2] = -std::sin(b_rad);
    R.m[2][1] =  std::sin(b_rad);
    R.m[2][2] =  std::cos(b_rad);
    return R;
}
static inline Mat3 Ry_rotate(double a_rad) {
    Mat3 R = Mat3::identity();
    R.m[0][0] =  std::cos(a_rad);
    R.m[0][2] =  std::sin(a_rad);
    R.m[2][0] = -std::sin(a_rad);
    R.m[2][2] =  std::cos(a_rad);
    return R;
}

static inline double wrap_pan_0_360(double a) {
    a = std::fmod(a, 360.0);
    if (a < 0) a += 360.0;
    if (a == 360.0) a = 0.0;
    return a;
}

static inline double clamp_tilt(double a) {
    return clamp(a, TILT_MIN, TILT_MAX);
}

// ===================== Zoom CSV 表 =====================
struct ZoomTable {
    std::vector<double> zoom;
    std::vector<double> x2;
    std::vector<double> y2;

    void sort_by_zoom() {
        std::vector<size_t> idx(zoom.size());
        for (size_t i=0;i<idx.size();++i) idx[i]=i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return zoom[a] < zoom[b]; });

        auto z=zoom, xx=x2, yy=y2;
        for (size_t i=0;i<idx.size();++i){
            zoom[i]=z[idx[i]];
            x2[i]=xx[idx[i]];
            y2[i]=yy[idx[i]];
        }
    }

    static bool parse_csv_line(const std::string& line, std::vector<std::string>& cols) {
        cols.clear();
        std::string cur;
        bool in_quote=false;
        for (size_t i=0;i<line.size();++i){
            char c=line[i];
            if (c=='"') in_quote = !in_quote;
            else if (c==',' && !in_quote) { cols.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        cols.push_back(cur);
        return !cols.empty();
    }

    static std::string strip_bom(std::string s) {
        // UTF-8 BOM: EF BB BF
        if (s.size() >= 3 && (unsigned char)s[0]==0xEF && (unsigned char)s[1]==0xBB && (unsigned char)s[2]==0xBF)
            return s.substr(3);
        return s;
    }

    static ZoomTable load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Failed to open CSV: " + path);

        std::string header;
        if (!std::getline(ifs, header)) throw std::runtime_error("CSV empty: " + path);
        header = strip_bom(header);

        // 找列索引：zoom/x2/y2
        std::vector<std::string> cols;
        parse_csv_line(header, cols);
        int iz=-1, ix=-1, iy=-1;
        for (int i=0;i<(int)cols.size();++i){
            auto c=cols[i];
            // trim
            while(!c.empty() && (c.back()=='\r' || c.back()=='\n' || c.back()==' ' || c.back()=='\t')) c.pop_back();
            while(!c.empty() && (c.front()==' ' || c.front()=='\t')) c.erase(c.begin());
            if (c=="zoom") iz=i;
            else if (c=="x2") ix=i;
            else if (c=="y2") iy=i;
        }
        if (iz<0 || ix<0 || iy<0) throw std::runtime_error("CSV header must contain zoom,x2,y2");

        ZoomTable t;
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            parse_csv_line(line, cols);
            if ((int)cols.size() <= std::max({iz,ix,iy})) continue;
            try{
                double z = std::stod(cols[iz]);
                double x = std::stod(cols[ix]);
                double y = std::stod(cols[iy]);
                if (std::isfinite(z) && std::isfinite(x) && std::isfinite(y)) {
                    t.zoom.push_back(z);
                    t.x2.push_back(x);
                    t.y2.push_back(y);
                }
            } catch(...) {}
        }
        if (t.zoom.empty()) throw std::runtime_error("CSV has no valid rows: " + path);
        t.sort_by_zoom();
        return t;
    }

    std::pair<double,double> query_x2y2(double z, double exact_tol=1e-9, bool interp=true) const {
        // 精确匹配
        for (size_t i=0;i<zoom.size();++i){
            if (std::abs(zoom[i]-z) <= exact_tol) return {x2[i], y2[i]};
        }
        if (!interp) throw std::runtime_error("zoom no exact match and interp=false");

        const double zmin = zoom.front();
        const double zmax = zoom.back();
        if (!(zmin <= z && z <= zmax)) {
            std::ostringstream oss;
            oss << "zoom " << z << " out of range [" << zmin << "," << zmax << "]";
            throw std::runtime_error(oss.str());
        }

        auto it = std::lower_bound(zoom.begin(), zoom.end(), z);
        if (it == zoom.begin()) return {x2.front(), y2.front()};
        if (it == zoom.end()) return {x2.back(), y2.back()};
        size_t j = (size_t)(it - zoom.begin());
        size_t i = j-1;

        const double z0=zoom[i], z1=zoom[j];
        const double t = (z - z0) / (z1 - z0 + 1e-30);
        const double xx = x2[i] + t*(x2[j]-x2[i]);
        const double yy = y2[i] + t*(y2[j]-y2[i]);
        return {xx, yy};
    }
};

// ===================== XGBoost 模型封装（单行预测） =====================
struct XGBModel {
    BoosterHandle booster{nullptr};

    void load(const std::string& path) {
        if (booster) { XGBoosterFree(booster); booster=nullptr; }
        XGBoosterCreate(nullptr, 0, &booster);
        // 尽量兼容 GPU / 新老参数（无 GPU 也能跑 CPU）

        XGBoosterSetParam(booster, "device", "cpu");
        XGBoosterSetParam(booster, "verbosity", "0");

        int rc = XGBoosterLoadModel(booster, path.c_str());
        if (rc != 0) {
            std::string msg = XGBGetLastError();
            throw std::runtime_error("XGBoosterLoadModel failed: " + msg + " path=" + path);
        }
    }

    ~XGBModel() {
        if (booster) { XGBoosterFree(booster); booster=nullptr; }
    }

    double predict_one_row(const float features[5]) const {
        // XGDMatrixCreateFromMat expects row-major float/double matrix
        DMatrixHandle dmat{nullptr};
        // missing = NaN
        int rc = XGDMatrixCreateFromMat((const float*)features, 1, 5, std::numeric_limits<float>::quiet_NaN(), &dmat);
        if (rc != 0) throw std::runtime_error(std::string("XGDMatrixCreateFromMat failed: ") + XGBGetLastError());

        bst_ulong out_len = 0;
        const float* out_result = nullptr;

        // 兼容新旧 predict API：尽量走 JSON 配置
        // option_mask / ntree_limit / training 旧接口也存在
        // 这里用旧接口更稳：XGBoosterPredict
        rc = XGBoosterPredict(booster, dmat, 0 /*option_mask*/, 0 /*ntree_limit*/, 0 /*training*/,
                              &out_len, &out_result);
        XGDMatrixFree(dmat);

        if (rc != 0) throw std::runtime_error(std::string("XGBoosterPredict failed: ") + XGBGetLastError());
        if (out_len < 1) throw std::runtime_error("XGBoosterPredict returned empty");
        return (double)out_result[0];
    }
};

// ===================== 全局上下文（启动加载一次） =====================
struct Context {
    ZoomTable table;
    XGBModel model_x;
    XGBModel model_y;
    mutable std::mutex predict_mtx; // Crow 多线程，预测上锁最稳妥
};

// ===================== FOV 拟合 / 反解（与 Python 一致） =====================
static inline double fov_from_zoom(double x, const double p[4]) {
    const double a=p[0], b=p[1], c=p[2], d=p[3];
    const double den = 1.0 + c*x + d*x*x;
    return (a + b*x)/den;
}

static inline double dy_dx(double x, const double p[4]) {
    const double a=p[0], b=p[1], c=p[2], d=p[3];
    const double den = 1.0 + c*x + d*x*x;
    const double num = a + b*x;
    return (b*den - num*(c + 2.0*d*x)) / (den*den);
}

static double invert_fov_to_zoom_continuous(double fov_deg, const double p[4],
                                            double x_min=X_MIN, double x_max=X_MAX, double eps=1e-12) {
    const double a=p[0], b=p[1], c=p[2], d=p[3];
    const double y = fov_deg;

    const double A = y*d;
    const double B = y*c - b;
    const double C = y - a;

    std::vector<std::pair<double,double>> candidates; // (err, x)

    if (std::abs(A) < eps) {
        if (std::abs(B) >= eps) {
            double x = -C / B;
            if (std::isfinite(x) && (x_min-1e-9 <= x && x <= x_max+1e-9)) {
                double yb = fov_from_zoom(x, p);
                candidates.push_back({std::abs(yb-y), x});
            }
        }
    } else {
        double disc = B*B - 4.0*A*C;
        if (disc >= -1e-10) {
            if (disc < 0) disc = 0;
            double sd = std::sqrt(disc);
            double x1 = (-B + sd) / (2.0*A);
            double x2 = (-B - sd) / (2.0*A);
            for (double x : {x1,x2}) {
                if (std::isfinite(x) && (x_min-1e-9 <= x && x <= x_max+1e-9)) {
                    double yb = fov_from_zoom(x, p);
                    candidates.push_back({std::abs(yb-y), x});
                }
            }
        }
    }

    if (!candidates.empty()) {
        struct Cand { double err, x, slope; };
        std::vector<Cand> cand;
        cand.reserve(candidates.size());
        for (auto& cnd : candidates) cand.push_back({cnd.first, cnd.second, dy_dx(cnd.second, p)});

        // 1) 优先 dy/dx < 0
        std::vector<Cand> neg;
        for (auto& t : cand) if (t.slope < 0) neg.push_back(t);
        if (!neg.empty()) cand.swap(neg);

        // 2) 误差最小附近允许并列，选更大的 x
        std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b){ return a.err < b.err; });
        const double best_err = cand.front().err;
        const double tol = 1e-3;
        double x_best = cand.front().x;
        for (auto& t : cand) {
            if (t.err <= best_err + tol) x_best = std::max(x_best, t.x);
        }
        return x_best;
    } else {
        // 边界兜底
        const double y_at_min = fov_from_zoom(x_min, p);
        const double y_at_max = fov_from_zoom(x_max, p);
        if (y >= y_at_min) return x_min;
        if (y <= y_at_max) return x_max;
        // 中间兜底：选更近的边界
        return (std::abs(y_at_min - y) <= std::abs(y_at_max - y)) ? x_min : x_max;
    }
}

static double zoom_min_from_fovs_continuous(double fov_h_deg, double fov_v_deg,
                                            double x_min=X_MIN, double x_max=X_MAX) {
    double z_h = invert_fov_to_zoom_continuous(fov_h_deg, P_H, x_min, x_max);
    double z_v = invert_fov_to_zoom_continuous(fov_v_deg, P_V, x_min, x_max);
    return std::min(z_h, z_v);
}

// ===================== K_simple（与 Python 一致） =====================
static Mat3 make_K_simple(double L, double H) {
    // K_simple = [
    //  [((2560 / 5.13) * 5.35), 0, 1280.00],
    //  [0, ((1440 / 2.89) * 5.35), 720.00],
    //  [0, 0, 1.00]
    // ]
    Mat3 K{};
    K.m[0][0] = (L / SENSOR_X) * FOCAL_F;
    K.m[0][1] = 0; K.m[0][2] = L/2.0;
    K.m[1][0] = 0; K.m[1][1] = (H / SENSOR_Y) * FOCAL_F; K.m[1][2] = H/2.0;
    K.m[2][0] = 0; K.m[2][1] = 0; K.m[2][2] = 1.0;
    return K;
}

// ===================== PTZ/像素变换（与 Python 一致） =====================
static Vec3 point1_to_point2_get_pt_D_value(const Vec3& raw_pos1, const Vec2& point1, const Vec2& point2,
                                            double X=SENSOR_X, double Y=SENSOR_Y, double L=2560, double H=1440, double f=FOCAL_F) {
    Vec3 result;
    double p0 = raw_pos1.x;
    double t0 = raw_pos1.y;
    double z0 = raw_pos1.z;

    double pos_p = std::round(p0 * 10.0)/10.0;
    double pos_t = std::round(t0 * 10.0)/10.0;

    const double x0_sensor = X/2.0;
    const double y0_sensor = Y/2.0;

    const Vec2 p1_phys{ (point1.x * X)/L, (point1.y * Y)/H };
    const Vec2 p2_phys{ (point2.x * X)/L, (point2.y * Y)/H };

    double a1 = rad2deg(std::atan((y0_sensor - p1_phys.y)/f));
    double a2 = rad2deg(std::atan((y0_sensor - p2_phys.y)/f));

    double new_t = pos_t - (a2 - a1);

    double b1 = rad2deg(std::atan((p1_phys.x - x0_sensor)/(f * std::cos(deg2rad(new_t)))));
    double b2 = rad2deg(std::atan((p2_phys.x - x0_sensor)/(f * std::cos(deg2rad(new_t)))));

    double new_p = pos_p + (b2 - b1);

    result.x = std::round(new_p * 1000.0)/1000.0;
    result.y = std::round(new_t * 1000.0)/1000.0;
    result.z = z0;
    return result;
}

static Vec2 pt1_and_pt2_to_p2_n_n_n(const Vec3& raw_pos1, const Vec3& raw_pos2, const Vec2& point1, const Mat3& CAL) {
    double b1 = raw_pos1.y;
    double b1_rad = deg2rad(-b1);
    double a2 = raw_pos2.x - raw_pos1.x;
    double b2 = raw_pos2.y;
    double a2_rad = deg2rad(a2);
    double b2_rad = deg2rad(b2);

    Mat3 Ra2_to_a1 = Ry_rotate(a2_rad);
    Mat3 Rb1 = Rx_rotate(b1_rad);
    Mat3 Rb2 = Rx_rotate(-b2_rad);

    Mat3 A_np = CAL;
    Mat3 A_inv = A_np.inverse();

    Vec3 base_vector_1{point1.x, point1.y, 1.0};

    // point2_vector = A_np @ Rb2 @ Ra2_to_a1 @ Rb1.T @ A_inv @ base_vector_1
    Mat3 M = matmul(A_np, matmul(Rb2, matmul(Ra2_to_a1, matmul(Rb1.transpose(), A_inv))));
    Vec3 v = matvec(M, base_vector_1);
    if (std::abs(v.z) < 1e-30) v.z = 1e-30;
    return Vec2{ v.x / v.z, v.y / v.z };
}

enum class P1P2Mode { ALL=0, ONLY_Y=1, ONLY_X=2, NO=3 };

static Vec2 point1_to_point2_get_p2_new_new_add_xg(
        const Context& ctx,
        const Vec3& raw_pos1, const Vec3& raw_pos2, const Vec2& point1,
        const Mat3& K_real,
        P1P2Mode mode,
        bool K_simple_mode,
        double L, double H
) {
    const Mat3 K_use = K_simple_mode ? make_K_simple(L,H) : K_real;

    Vec2 point2 = pt1_and_pt2_to_p2_n_n_n(raw_pos1, raw_pos2, point1, K_use);

    double delta_P = round_to_1(raw_pos2.x - raw_pos1.x);
    double delta_T = round_to_1(raw_pos2.y - raw_pos1.y);

    double x_nor1 = (point1.x - (L/2.0)) / (L/2.0);
    double y_nor1 = ((H/2.0) - point1.y) / (H/2.0);

    double T1_scale = (raw_pos1.y + 5.0) / 95.0;
    double delta_P_scale = delta_P / 360.0;
    double delta_T_scale = delta_T / 95.0;

    float feats[5] = {
            (float)x_nor1, (float)y_nor1, (float)delta_P_scale, (float)delta_T_scale, (float)T1_scale
    };

    if (mode == P1P2Mode::NO) {
        return Vec2{ (double)point2.x, (double)point2.y };
    }

    // 预测上锁（多线程）
    double x_pred=0.0, y_pred=0.0;
    {
        std::lock_guard<std::mutex> lk(ctx.predict_mtx);
        if (mode == P1P2Mode::ALL || mode == P1P2Mode::ONLY_X) x_pred = ctx.model_x.predict_one_row(feats);
        if (mode == P1P2Mode::ALL || mode == P1P2Mode::ONLY_Y) y_pred = ctx.model_y.predict_one_row(feats);
    }

    if (mode == P1P2Mode::ALL || mode == P1P2Mode::ONLY_X) point2.x = point2.x - x_pred;
    if (mode == P1P2Mode::ALL || mode == P1P2Mode::ONLY_Y) point2.y = point2.y - y_pred;

    return Vec2{ point2.x, point2.y };
}

// ===================== ROI / Zoom（与 Python 一致） =====================
static std::pair<Vec2,Vec2> roi_points_from_mid_center(
        double fov_min_center_x1, double fov_min_center_y1,
        double fov_min_center_x2, double fov_min_center_y2,
        double fov_min_w, double fov_min_h,
        double L, double H
) {
    double cx = (fov_min_center_x1 + fov_min_center_x2)/2.0;
    double cy = (fov_min_center_y1 + fov_min_center_y2)/2.0;

    double w = fov_min_w, h = fov_min_h;

    if (w >= L || h >= H) {
        return {Vec2{0,0}, Vec2{(double)std::lround(L), (double)std::lround(H)}};
    }

    cx = std::max(w/2.0, std::min(cx, L - w/2.0));
    cy = std::max(h/2.0, std::min(cy, H - h/2.0));

    double x1 = cx - w/2.0;
    double y1 = cy - h/2.0;
    double x2 = cx + w/2.0;
    double y2 = cy + h/2.0;

    return { Vec2{(double)std::lround(x1),(double)std::lround(y1)},
             Vec2{(double)std::lround(x2),(double)std::lround(y2)} };
}

static double ROI_to_Zoom(double Lpix, double Hpix, const Vec2& ROI1, const Vec2& ROI2,
                          double rate=1.0, double f0=FOCAL_F,
                          double X=SENSOR_X, double Y=SENSOR_Y,
                          double max_zoom=23.0) {
    double length = ((ROI2.x - ROI1.x) * X) / Lpix;
    double height = ((ROI2.y - ROI1.y) * Y) / Hpix;

    double aspect_img = Lpix / Hpix;

    if (((ROI2.x-ROI1.x)/(ROI2.y-ROI1.y)) >= aspect_img) {
        double length_new = length;
        double height_new = (Hpix/Lpix) * length_new;
        double cam_angle_h = 2.0 * rad2deg(std::atan(length_new / (2.0*rate*f0)));
        double cam_angle_v = 2.0 * rad2deg(std::atan(height_new / (2.0*rate*f0)));
        double z = zoom_min_from_fovs_continuous(cam_angle_h, cam_angle_v);
        double z_final = std::floor(z*10.0)/10.0;
        return (z_final <= max_zoom) ? z_final : max_zoom;
    } else {
        double height_new = height;
        double length_new = (Lpix/Hpix) * height_new;
        double cam_angle_h = 2.0 * rad2deg(std::atan(length_new / (2.0*rate*f0)));
        double cam_angle_v = 2.0 * rad2deg(std::atan(height_new / (2.0*rate*f0)));
        double z = zoom_min_from_fovs_continuous(cam_angle_h, cam_angle_v);
        double z_final = std::floor(z*10.0)/10.0;
        return (z_final <= max_zoom) ? z_final : max_zoom;
    }
}

static double ROI_to_Zoom_fov(double Lpix, double Hpix,
                              const Vec2& ROI1, const Vec2& ROI2,
                              const ViewingAngles& va,
                              double rate=1.0,
                              double X=SENSOR_X, double Y=SENSOR_Y,
                              double max_zoom=23.0) {
    double f0 = (X/2.0) / std::tan(deg2rad(va.x_deg/2.0));
    double length = ((ROI2.x - ROI1.x) * X) / Lpix;
    double height = ((ROI2.y - ROI1.y) * Y) / Hpix;

    double aspect_img = Lpix / Hpix;

    if (((ROI2.x-ROI1.x)/(ROI2.y-ROI1.y)) >= aspect_img) {
        double length_new = length;
        double height_new = (Hpix/Lpix) * length_new;
        double cam_angle_h = 2.0 * rad2deg(std::atan(length_new / (2.0*rate*f0)));
        double cam_angle_v = 2.0 * rad2deg(std::atan(height_new / (2.0*rate*f0)));
        double z = zoom_min_from_fovs_continuous(cam_angle_h, cam_angle_v);
        double z_final = std::floor(z*10.0)/10.0;
        return (z_final <= max_zoom) ? z_final : max_zoom;
    } else {
        double height_new = height;
        double length_new = (Lpix/Hpix) * height_new;
        double cam_angle_h = 2.0 * rad2deg(std::atan(length_new / (2.0*rate*f0)));
        double cam_angle_v = 2.0 * rad2deg(std::atan(height_new / (2.0*rate*f0)));
        double z = zoom_min_from_fovs_continuous(cam_angle_h, cam_angle_v);
        double z_final = std::floor(z*10.0)/10.0;
        return (z_final <= max_zoom) ? z_final : max_zoom;
    }
}

static std::pair<double,double> inverse_translate_Z(
        double z,
        const ViewingAngles& va,
        double Lpix, double Hpix,
        double rate=1.0, double X=SENSOR_X, double Y=SENSOR_Y,
        double max_zoom=23.0,
        int iters=45,
        double shrink=0.9
) {
    if (Lpix <= 0 || Hpix <= 0) throw std::runtime_error("Invalid resolution");
    if (z <= 1.0) {
        double l=Lpix, h=Hpix;
        l = std::max(1.0, l*shrink);
        h = std::max(1.0, h*shrink);
        return {l,h};
    }
    if (z > max_zoom) z = max_zoom;

    double f0 = (X/2.0) / std::tan(deg2rad(va.x_deg/2.0));

    auto v_from_h = [&](double h_deg)->double{
        return rad2deg(2.0 * std::atan((Hpix/Lpix) * std::tan(deg2rad(h_deg)/2.0)));
    };
    auto z_required = [&](double h_deg)->double{
        double v_deg = v_from_h(h_deg);
        return zoom_min_from_fovs_continuous(h_deg, v_deg);
    };

    double h_hi = va.x_deg;
    double h_lo = 0.05;

    if (z <= z_required(h_hi)) {
        double l=Lpix, h=Hpix;
        l = std::min(Lpix, std::max(1.0, l*shrink));
        h = std::min(Hpix, std::max(1.0, h*shrink));
        return {l,h};
    }

    double lo=h_lo, hi=h_hi;
    for (int k=0;k<iters;k++){
        double mid = (lo+hi)/2.0;
        double zm = z_required(mid);
        if (zm > z) lo = mid;
        else hi = mid;
    }
    double h_deg = (lo+hi)/2.0;

    double length_new = 2.0 * rate * f0 * std::tan(deg2rad(h_deg)/2.0);
    double l = (length_new * Lpix) / X;
    double h = l * (Hpix/Lpix);

    l *= shrink;
    h *= shrink;

    l = std::min(Lpix, std::max(1.0, l));
    h = std::min(Hpix, std::max(1.0, h));
    return {l,h};
}

// ===================== final_method（与 Python 一致） =====================
static Vec2 final_method(const ZoomTable& df, double init_zoom, double zoom,
                         double cx=1280.0, double cy=720.0, double eps=1e-6) {
    auto [x2,y2] = df.query_x2y2(zoom);

    if (std::abs(zoom - 1.0) < eps) {
        return Vec2{cx, cy};
    }

    double dx = x2 - cx;
    double dy = y2 - cy;
    double kx = dx / (zoom - 1.0);
    double ky = dy / (zoom - 1.0);

    double dx_init = kx * (init_zoom - 1.0);
    double dy_init = ky * (init_zoom - 1.0);

    return Vec2{cx + dx_init, cy + dy_init};
}

// ===================== LM 优化（与 Python optimize_pan_tilt_LM_v2 对齐） =====================
struct LMInfo {
    int iters{0};
    std::string reason;
    double final_err_max_axis_cont{0};
};

static std::pair<Vec2,Vec2> ROI_trans(
        const Context& ctx,
        double Lpix, double Hpix,
        const Vec2& ROI_point1, const Vec2& ROI_point2,
        const Vec3& raw_pos1,
        const Vec2& point_res,           // target_coord in pixel (point2)
        const Vec3& result_ptz,
        const Mat3& K_real,
        P1P2Mode forward_mode,
        bool K_simple_mode
) {
    // 4 corners
    Vec2 ROI_point1_right{ROI_point2.x, ROI_point1.y};
    Vec2 ROI_point2_left {ROI_point1.x, ROI_point2.y};

    Vec2 trans1 = point1_to_point2_get_p2_new_new_add_xg(ctx, raw_pos1, result_ptz, ROI_point1, K_real, forward_mode, K_simple_mode, Lpix, Hpix);
    Vec2 trans2 = point1_to_point2_get_p2_new_new_add_xg(ctx, raw_pos1, result_ptz, ROI_point2, K_real, forward_mode, K_simple_mode, Lpix, Hpix);
    Vec2 trans3 = point1_to_point2_get_p2_new_new_add_xg(ctx, raw_pos1, result_ptz, ROI_point1_right, K_real, forward_mode, K_simple_mode, Lpix, Hpix);
    Vec2 trans4 = point1_to_point2_get_p2_new_new_add_xg(ctx, raw_pos1, result_ptz, ROI_point2_left,  K_real, forward_mode, K_simple_mode, Lpix, Hpix);

    std::vector<Vec2> pts{trans1,trans3,trans4,trans2};

    Vec2 init_center{ (ROI_point1.x + ROI_point2.x)/2.0, (ROI_point1.y + ROI_point2.y)/2.0 };

    double dx_max=0, dy_max=0;
    for (auto& p : pts) {
        dx_max = std::max(dx_max, std::abs(p.x - point_res.x));
        dy_max = std::max(dy_max, std::abs(p.y - point_res.y));
    }
    double new_w = 2.0*dx_max;
    double new_h = 2.0*dy_max;

    double init_w = std::abs(ROI_point2.x - ROI_point1.x);
    double init_h = std::abs(ROI_point2.y - ROI_point1.y);

    double final_w = std::max(init_w, new_w);
    double final_h = std::max(init_h, new_h);

    double xmin = init_center.x - final_w/2.0;
    double xmax = init_center.x + final_w/2.0;
    double ymin = init_center.y - final_h/2.0;
    double ymax = init_center.y + final_h/2.0;

    return {Vec2{xmin,ymin}, Vec2{xmax,ymax}};
}

static Mat3 K_dummy_real(double Lpix, double Hpix) {
    // 这里给一个和 K_simple 同构的占位（你实际如有真实K，可在请求里扩展传入）
    return make_K_simple(Lpix,Hpix);
}

static std::pair<Vec3, LMInfo> optimize_pan_tilt_LM_v2(
        const Context& ctx,
        double Lpix, double Hpix,
        const Vec3& raw_pos1,      // [pan, tilt, zoom]
        const Vec2& point1,        // [x,y]
        const Mat3& K_real,
        const Vec2& target_coord,  // [x,y]
        P1P2Mode forward_mode,
        bool K_simple_mode,
        const std::string& init_mode, // "C" / "raw" / "second_time"
        const Vec3& second_pos,
        double tol_per_axis=2.0,
        int max_iters=12,
        double jac_step_deg=0.18,
        double lambda_init=5e-2,
        double lambda_up=6.0,
        double lambda_down=0.25,
        double tiny_step_stop_deg=1e-4
) {
    Vec3 pos1 = raw_pos1;
    Vec2 p1 = point1;
    Vec2 target = target_coord;

    const double zoom_fixed = pos1.z;

    double pan0=0, tilt0=0;
    if (init_mode == "C") {
        Vec3 ptzC = point1_to_point2_get_pt_D_value(pos1, p1, target, SENSOR_X, SENSOR_Y, Lpix, Hpix, FOCAL_F);
        pan0 = wrap_pan_0_360(ptzC.x);
        tilt0 = clamp_tilt(ptzC.y);
    } else if (init_mode == "raw") {
        pan0 = wrap_pan_0_360(pos1.x);
        tilt0 = clamp_tilt(pos1.y);
    } else if (init_mode == "second_time") {
        pan0 = wrap_pan_0_360(second_pos.x);
        tilt0 = clamp_tilt(second_pos.y);
    } else {
        throw std::runtime_error("Unknown init_mode");
    }

    auto build_ptz_cont = [&](double pan_deg, double tilt_deg)->Vec3{
        return Vec3{ wrap_pan_0_360(pan_deg), clamp_tilt(tilt_deg), zoom_fixed };
    };

    auto f_cont = [&](double pan_deg, double tilt_deg)->Vec2{
        Vec3 raw_pos2 = build_ptz_cont(pan_deg, tilt_deg);
        return point1_to_point2_get_p2_new_new_add_xg(ctx, pos1, raw_pos2, p1, K_real,
                                                      forward_mode, K_simple_mode, Lpix, Hpix);
    };

    auto round1_angles = [&](double pan_deg, double tilt_deg)->std::pair<double,double>{
        double pan = wrap_pan_0_360(pan_deg);
        double tilt = clamp_tilt(tilt_deg);
        return { round_to_1(pan), round_to_1(tilt) };
    };

    double best_pan = pan0, best_tilt = tilt0;
    Vec2 best_xy = f_cont(best_pan, best_tilt);
    double best_err_max = std::max(std::abs(best_xy.x - target.x), std::abs(best_xy.y - target.y));

    LMInfo info;
    if (best_err_max <= tol_per_axis) {
        auto [pr,tr] = round1_angles(best_pan,best_tilt);
        info.iters = 0;
        info.reason = "init_within_tolerance";
        info.final_err_max_axis_cont = best_err_max;
        return { Vec3{pr,tr,zoom_fixed}, info };
    }

    double lam = lambda_init;
    int iters_done=0;

    // 返回列堆叠： [dfdpan.x, dfdpan.y, dfdtilt.x, dfdtilt.y]
    auto jacobian_forward_bounded_cached =
            [&](double pan_deg, double tilt_deg, const Vec2& base_xy, double h_deg)
                    -> std::array<double, 4>
            {
                const double inv_h = 1.0 / h_deg;

                // d f / d pan
                double pan_p = wrap_pan_0_360(pan_deg + h_deg);
                Vec2 xy_pan = f_cont(pan_p, tilt_deg);
                Vec2 dfdpan{ (xy_pan.x - base_xy.x)*inv_h, (xy_pan.y - base_xy.y)*inv_h };

                // d f / d tilt （优先 forward，否则 backward）
                Vec2 dfdtilt{0,0};
                if (tilt_deg + h_deg <= TILT_MAX) {
                    double tilt_s = tilt_deg + h_deg;
                    Vec2 xy_tilt = f_cont(pan_deg, tilt_s);
                    dfdtilt = Vec2{ (xy_tilt.x - base_xy.x)*inv_h, (xy_tilt.y - base_xy.y)*inv_h };
                } else if (tilt_deg - h_deg >= TILT_MIN) {
                    double tilt_s = tilt_deg - h_deg;
                    Vec2 xy_tilt = f_cont(pan_deg, tilt_s);
                    dfdtilt = Vec2{ (base_xy.x - xy_tilt.x)*inv_h, (base_xy.y - xy_tilt.y)*inv_h };
                } else {
                    dfdtilt = Vec2{0,0};
                }

                return { dfdpan.x, dfdpan.y, dfdtilt.x, dfdtilt.y };
            };

    for (int it=1; it<=max_iters; ++it) {
        iters_done = it;

        // ✅ 这里改成接 std::array
        const auto Jv = jacobian_forward_bounded_cached(best_pan, best_tilt, best_xy, jac_step_deg);

        // J matrix rows:
        // [Jv0  Jv2]
        // [Jv1  Jv3]
        double J00 = Jv[0], J10 = Jv[1];
        double J01 = Jv[2], J11 = Jv[3];

        // r = best_xy - target
        double r0 = best_xy.x - target.x;
        double r1 = best_xy.y - target.y;

        // Hm = J^T J + lam I
        double H00 = J00*J00 + J10*J10 + lam;
        double H01 = J00*J01 + J10*J11;
        double H10 = H01;
        double H11 = J01*J01 + J11*J11 + lam;

        // g = J^T r
        double g0 = J00*r0 + J10*r1;
        double g1 = J01*r0 + J11*r1;

        // solve delta = -inv(H)*g  (2x2)
        double det = H00*H11 - H01*H10;
        if (std::abs(det) < 1e-18) { lam *= lambda_up; continue; }

        double inv00 =  H11/det;
        double inv01 = -H01/det;
        double inv10 = -H10/det;
        double inv11 =  H00/det;

        double d0 = -(inv00*g0 + inv01*g1);
        double d1 = -(inv10*g0 + inv11*g1);

        if (std::max(std::abs(d0), std::abs(d1)) < tiny_step_stop_deg) break;

        double cand_pan  = wrap_pan_0_360(best_pan  + d0);
        double cand_tilt = clamp_tilt(best_tilt + d1);

        Vec2 cand_xy = f_cont(cand_pan, cand_tilt);
        double cand_err_max = std::max(std::abs(cand_xy.x - target.x), std::abs(cand_xy.y - target.y));

        bool accepted = cand_err_max < best_err_max - 1e-9;
        if (accepted) {
            best_pan = cand_pan;
            best_tilt = cand_tilt;
            best_xy = cand_xy;
            best_err_max = cand_err_max;
            lam = std::max(lam*lambda_down, 1e-12);
        } else {
            lam = lam*lambda_up;
        }

        if (best_err_max <= tol_per_axis) break;
    }

    auto [pr,tr] = round1_angles(best_pan,best_tilt);
    info.iters = iters_done;
    info.reason = (best_err_max <= tol_per_axis) ? "target_reached" : "best_effort";
    info.final_err_max_axis_cont = best_err_max;
    return { Vec3{pr,tr,zoom_fixed}, info };
}

// ===================== ours_pt_method / ours_roi_zoom_only_pt（与 Python 一致） =====================
static std::pair<Vec3, LMInfo> ours_pt_method(
        const Context& ctx,
        double Lpix, double Hpix,
        const ZoomTable& df_zoom,
        const Vec3& pos1,
        const Vec2& point1,
        double zoom,
        const Mat3& K_real,
        const std::pair<Vec2,Vec2>& ROI_points,
        double rate,
        const std::string& init_mode,
        P1P2Mode forward_mode,
        bool K_simple_mode,
        const std::string& method_mode
) {
    if (method_mode == "steady_zoom") {
        Vec2 point2 = final_method(df_zoom, pos1.z, zoom, Lpix/2.0, Hpix/2.0);
        auto [ptz, info] = optimize_pan_tilt_LM_v2(
                ctx, Lpix, Hpix, pos1, point1, K_real, point2,
                forward_mode, K_simple_mode,
                init_mode, Vec3{0,0,0},
                1.0, 33
        );
        Vec3 out = ptz;
        out.z = zoom;
        return {out, info};
    }
    else if (method_mode == "ROI_zoom") {
        double Z = ROI_to_Zoom(Lpix, Hpix, ROI_points.first, ROI_points.second, rate, FOCAL_F, SENSOR_X, SENSOR_Y, 23.0);
        Vec2 point2 = final_method(df_zoom, pos1.z, Z, Lpix/2.0, Hpix/2.0);

        auto [ptz1, info1] = optimize_pan_tilt_LM_v2(
                ctx, Lpix, Hpix, pos1, point1, K_real, point2,
                forward_mode, K_simple_mode,
                init_mode, Vec3{0,0,0},
                1.0, 33
        );

        auto new_roi = ROI_trans(ctx, Lpix, Hpix, ROI_points.first, ROI_points.second, pos1, point2, ptz1,
                                 K_real, forward_mode, K_simple_mode);
        double new_Z = ROI_to_Zoom(Lpix, Hpix, new_roi.first, new_roi.second, rate, FOCAL_F, SENSOR_X, SENSOR_Y, 23.0);
        Vec2 point2_new = final_method(df_zoom, pos1.z, new_Z, Lpix/2.0, Hpix/2.0);

        auto [ptz2, info2] = optimize_pan_tilt_LM_v2(
                ctx, Lpix, Hpix, pos1, point1, K_real, point2_new,
                forward_mode, K_simple_mode,
                "second_time", ptz1,
                1.0, 33
        );

        Vec3 out = ptz2;
        out.z = new_Z;
        LMInfo merged = info2;
        merged.iters = info1.iters + info2.iters; // 仅用于统计
        return {out, merged};
    }
    else if (method_mode == "no_second_trans") {
        double Z = ROI_to_Zoom(Lpix, Hpix, ROI_points.first, ROI_points.second, rate, FOCAL_F, SENSOR_X, SENSOR_Y, 23.0);
        Vec2 point2 = final_method(df_zoom, pos1.z, Z, Lpix/2.0, Hpix/2.0);

        auto [ptz, info] = optimize_pan_tilt_LM_v2(
                ctx, Lpix, Hpix, pos1, point1, K_real, point2,
                forward_mode, K_simple_mode,
                init_mode, Vec3{0,0,0},
                1.0, 33
        );
        Vec3 out = ptz;
        out.z = Z;
        return {out, info};
    }
    else if (method_mode == "no_LM") {
        double Z = ROI_to_Zoom(Lpix, Hpix, ROI_points.first, ROI_points.second, rate, FOCAL_F, SENSOR_X, SENSOR_Y, 23.0);
        Vec2 point2 = final_method(df_zoom, pos1.z, Z, Lpix/2.0, Hpix/2.0);
        Vec3 result_ptz = point1_to_point2_get_pt_D_value(pos1, point1, point2, SENSOR_X, SENSOR_Y, Lpix, Hpix, FOCAL_F);
        result_ptz.z = Z;

        auto new_roi = ROI_trans(ctx, Lpix, Hpix, ROI_points.first, ROI_points.second, pos1, point2, result_ptz,
                                 K_real, forward_mode, K_simple_mode);
        double new_Z = ROI_to_Zoom(Lpix, Hpix, new_roi.first, new_roi.second, rate, FOCAL_F, SENSOR_X, SENSOR_Y, 23.0);

        Vec2 point2_new = final_method(df_zoom, pos1.z, new_Z, Lpix/2.0, Hpix/2.0);
        Vec3 result_ptz_new = point1_to_point2_get_pt_D_value(pos1, point1, point2_new, SENSOR_X, SENSOR_Y, Lpix, Hpix, FOCAL_F);
        result_ptz_new.z = new_Z;

        LMInfo info; info.iters=0; info.reason="no_LM"; info.final_err_max_axis_cont=0;
        return {result_ptz_new, info};
    }
    else {
        throw std::runtime_error("Unknown method_mode");
    }
}

static Vec2 ours_roi_zoom_only_pt(
        const Context& ctx,
        double Lpix, double Hpix,
        const ZoomTable& df_zoom,
        const Vec3& pos_now,
        double c_x, double c_y,
        const Mat3& K_real,
        const std::pair<Vec2,Vec2>& ROI_points,
        double rate,
        double pan_offset_value,
        double tilt_offset_value,
        const std::string& init_mode,
        P1P2Mode forward_mode,
        bool K_simple_mode
) {
    auto start = std::chrono::high_resolution_clock::now();

    auto [ptz, info] = ours_pt_method(
            ctx, Lpix, Hpix,
            df_zoom,
            pos_now,
            Vec2{c_x,c_y},
            pos_now.z, // 占位
            K_real,
            ROI_points,
            rate,
            init_mode,
            forward_mode,
            K_simple_mode,
            "ROI_zoom"
    );

    double pan = ptz.x + pan_offset_value;
    double tilt = ptz.y + tilt_offset_value;

    (void)info;

    auto end = std::chrono::high_resolution_clock::now();
    (void)start; (void)end;

    return Vec2{pan, tilt};
}

// ===================== 3D 质心（与 Python 一致） =====================
static std::tuple<double,double,double> polygon_area_and_centroid(const std::vector<Vec2>& vertices) {
    if (vertices.size() < 3) {
        double mx=0,my=0;
        for (auto& v: vertices){ mx+=v.x; my+=v.y; }
        if (!vertices.empty()){ mx/=vertices.size(); my/=vertices.size(); }
        return {0.0, mx, my};
    }

    double double_area = 0.0;
    for (size_t i=0;i<vertices.size();++i){
        const auto& p = vertices[i];
        const auto& q = vertices[(i+1)%vertices.size()];
        double_area += p.x*q.y - q.x*p.y;
    }
    double signed_area = 0.5 * double_area;
    double area = std::abs(signed_area);

    if (area == 0.0) {
        double mx=0,my=0;
        for (auto& v: vertices){ mx+=v.x; my+=v.y; }
        mx/=vertices.size(); my/=vertices.size();
        return {0.0, mx, my};
    }

    double cx=0.0, cy=0.0;
    for (size_t i=0;i<vertices.size();++i){
        const auto& p = vertices[i];
        const auto& q = vertices[(i+1)%vertices.size()];
        double cross = p.x*q.y - q.x*p.y;
        cx += (p.x + q.x) * cross;
        cy += (p.y + q.y) * cross;
    }
    cx /= (6.0 * signed_area);
    cy /= (6.0 * signed_area);
    return {area, cx, cy};
}

static double compute_prism_volume(double A1, double A2, double z1, double z2) {
    return (1.0/3.0) * std::abs(A1 + A2 + std::sqrt(std::max(0.0, A1*A2))) * std::abs(z2 - z1);
}

static Vec3 compute_3d_centroid(const std::vector<std::pair<double,std::vector<Vec2>>>& slices) {
    if (slices.empty()) return Vec3{0,0,0};
    if (slices.size() == 1) {
        auto [z, verts] = slices[0];
        auto [A, Cx, Cy] = polygon_area_and_centroid(verts);
        (void)A;
        return Vec3{Cx, Cy, z};
    }

    double total_volume=0.0;
    double wcx=0.0, wcy=0.0, wcz=0.0;
    std::vector<Vec2> all_points;
    all_points.reserve(1024);

    for (size_t i=0;i+1<slices.size();++i){
        double z1 = slices[i].first;
        double z2 = slices[i+1].first;
        const auto& v1 = slices[i].second;
        const auto& v2 = slices[i+1].second;

        auto [A1, Cx1, Cy1] = polygon_area_and_centroid(v1);
        auto [A2, Cx2, Cy2] = polygon_area_and_centroid(v2);

        if (A1==0.0 && A2==0.0) {
            all_points.insert(all_points.end(), v1.begin(), v1.end());
            all_points.insert(all_points.end(), v2.begin(), v2.end());
            continue;
        }

        double V = compute_prism_volume(A1, A2, z1, z2);
        total_volume += V;

        double weight1=0.0, weight2=0.0;
        if (A1==0.0) { weight1=0.0; weight2=1.0; }
        else if (A2==0.0) { weight1=1.0; weight2=0.0; }
        else {
            double denom = (A1 + A2 + std::sqrt(std::max(0.0, A1*A2)));
            weight1 = (A1 + std::sqrt(std::max(0.0, A1*A2))) / (denom + 1e-30);
            weight2 = 1.0 - weight1;
        }

        double Cz = z1*weight1 + z2*weight2;
        double Cx = Cx1*weight1 + Cx2*weight2;
        double Cy = Cy1*weight1 + Cy2*weight2;

        wcx += V*Cx;
        wcy += V*Cy;
        wcz += V*Cz;

        all_points.insert(all_points.end(), v1.begin(), v1.end());
        all_points.insert(all_points.end(), v2.begin(), v2.end());
    }

    if (total_volume == 0.0) {
        if (all_points.empty()) {
            // fallback
            double mz=0;
            for (auto& s: slices) mz += s.first;
            mz /= slices.size();
            return Vec3{0,0,mz};
        }
        double mx=0,my=0;
        for (auto& p: all_points){ mx+=p.x; my+=p.y; }
        mx/=all_points.size(); my/=all_points.size();
        double mz=0;
        for (auto& s: slices) mz += s.first;
        mz /= slices.size();
        return Vec3{mx,my,mz};
    }

    return Vec3{ wcx/total_volume, wcy/total_volume, wcz/total_volume };
}

// ===================== generate_sequence（与 Python 一致） =====================
static std::vector<double> generate_sequence(double x1, double x2, double step) {
    std::vector<double> seq;
    if (step <= 0) { seq.push_back(x2); return seq; }

    double cur = x1;
    int guard=0;
    while (cur <= x2 + step + 1e-12 && guard++ < 100000) {
        double v = std::round(cur * 1e10) / 1e10;
        seq.push_back(v);
        cur += step;
    }
    if (seq.empty()) { seq.push_back(x2); return seq; }

    if (seq.back() > x2) seq.pop_back();
    if (seq.empty() || std::abs(seq.back() - x2) > 1e-12) seq.push_back(x2);
    return seq;
}

// ===================== shrink（与 Python 一致） =====================
static void shrink_towards_center(std::vector<Vec2>& vs, double factor) {
    if (factor == 0.0 || vs.empty()) return;
    double cx=0, cy=0;
    for (auto& p: vs){ cx += p.x; cy += p.y; }
    cx /= vs.size(); cy /= vs.size();
    for (auto& p: vs) {
        p.x = cx + (p.x - cx) * (1.0 - factor);
        p.y = cy + (p.y - cy) * (1.0 - factor);
    }
}

// ===================== Geometric（与 Python 逻辑一致） =====================
struct Geometric {
    int g_index{0};
    Vec3 pos_now;
    int geometric_type{0}; // 0/1/4/5
    double min_z{0}, max_z{0};
    Vec3 centroid{0,0,0};

    // hierarchical_point_set: list of [z, h_ps] where h_ps are 8 points [pan,tilt]
    std::vector<std::pair<double,std::vector<Vec2>>> hierarchical_point_set;

    Geometric(
            const Context& ctx,
            const ZoomTable& df_zoom,
            int g_index_,
            const Vec3& pos_now_,
            const Box& box,
            double proportion,
            double fov_w, double fov_h,
            double z_max,
            const ViewingAngles& va,
            double pan_offset_value,
            double tilt_offset_value,
            double shrink_factor,
            double Lpix, double Hpix,
            P1P2Mode forward_mode,
            bool K_simple_mode
    ) : g_index(g_index_), pos_now(pos_now_) {
        // 目标框宽高
        double box_w = box.x2 - box.x1;
        double box_h = box.y2 - box.y1;
        double box_area = box_w * box_h;
        double fov_area = fov_w * fov_h;

        double fov_aspect_ratio = fov_h / fov_w;
        double box_aspect_ratio = box_h / box_w;

        double now_p = box_area / fov_area;

        Mat3 K_real = K_dummy_real(Lpix,Hpix);

        if (now_p >= proportion) {
            geometric_type = 0;
            return;
        }

        // 最大视野面积
        double fov_max_area = box_area / proportion;
        double fov_max_w = std::sqrt(fov_max_area / fov_aspect_ratio);
        double fov_max_h = fov_max_w * fov_aspect_ratio;

        // ======= helper: 生成一层 8 点（与 Python 中每层调用 8 次 ours_roi_zoom_only_pt 对齐） =======
        auto build_layer_8pts = [&](double z, double l, double h)->std::vector<Vec2>{
            double l_h = l/2.0;
            double h_h = h/2.0;
            double f_c_x1 = box.x2 - l_h;
            double f_c_y1 = box.y2 - h_h;
            double f_c_x2 = box.x1 + l_h;
            double f_c_y2 = box.y1 + h_h;
            double f_c_h_x = (f_c_x1 + f_c_x2)/2.0;
            double f_c_h_y = (f_c_y1 + f_c_y2)/2.0;

            auto mk_roi = [&](double cx, double cy)->std::pair<Vec2,Vec2>{
                return roi_points_from_mid_center(cx, cy, cx, cy, l, h, Lpix, Hpix);
            };

            // 8 个中心点：1,12,2,23,3,34,4,41
            auto ROI_1  = mk_roi(f_c_x1, f_c_y1);
            auto ROI_2  = mk_roi(f_c_x2, f_c_y1);
            auto ROI_3  = mk_roi(f_c_x2, f_c_y2);
            auto ROI_4  = mk_roi(f_c_x1, f_c_y2);
            auto ROI_12 = mk_roi(f_c_h_x, f_c_y1);
            auto ROI_23 = mk_roi(f_c_x2, f_c_h_y);
            auto ROI_34 = mk_roi(f_c_h_x, f_c_y2);
            auto ROI_41 = mk_roi(f_c_x1, f_c_h_y);

            Vec2 pt1  = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x1, f_c_y1, K_real, ROI_1,  0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt12 = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_h_x, f_c_y1, K_real, ROI_12, 0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt2  = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x2, f_c_y1, K_real, ROI_2,  0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt23 = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x2, f_c_h_y, K_real, ROI_23, 0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt3  = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x2, f_c_y2, K_real, ROI_3,  0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt34 = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_h_x, f_c_y2, K_real, ROI_34, 0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt4  = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x1, f_c_y2, K_real, ROI_4,  0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);
            Vec2 pt41 = ours_roi_zoom_only_pt(ctx, Lpix, Hpix, df_zoom, pos_now, f_c_x1, f_c_h_y, K_real, ROI_41, 0.9, pan_offset_value, tilt_offset_value, "C", forward_mode, K_simple_mode);

            std::vector<Vec2> hps{pt1,pt12,pt2,pt23,pt3,pt34,pt4,pt41};
            if (shrink_factor > 0) shrink_towards_center(hps, shrink_factor);
            return hps;
        };

        // ======= thin case: fov_max smaller than box => line types =======
        if (fov_max_h < box_h || fov_max_w < box_w) {
            if (box_aspect_ratio > fov_aspect_ratio) {
                // 横向直线 type=4
                geometric_type = 4;

                double fov_min_h = box_h;
                double fov_min_w = fov_min_h / fov_aspect_ratio;

                double fov_min_l_x1 = box.x1;
                double fov_min_l_y1 = box.y1;
                double fov_min_center_x2 = fov_min_l_x1 + (fov_min_w/2.0);
                double fov_min_center_y2 = fov_min_l_y1 + (fov_min_h/2.0);

                double fov_min_r_x2 = box.x2;
                double fov_min_r_y2 = box.y2;
                double fov_min_center_x1 = fov_min_r_x2 - (fov_min_w/2.0);
                double fov_min_center_y1 = fov_min_r_y2 - (fov_min_h/2.0);

                auto ROI = roi_points_from_mid_center(
                        fov_min_center_x1, fov_min_center_y1,
                        fov_min_center_x2, fov_min_center_y2,
                        fov_min_w, fov_min_h, Lpix, Hpix
                );
                double z2 = ROI_to_Zoom_fov(Lpix, Hpix, ROI.first, ROI.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
                if (z2 > z_max || z2 < 0) z2 = z_max;

                if (z2 > 1.0) {
                    geometric_type = 1;
                    double z1 = z2 - 1.0;
                    min_z = z1;
                    max_z = z2;

                    auto zs = generate_sequence(z1, z2, 0.5);
                    for (double z : zs) {
                        auto [l,h] = inverse_translate_Z(z, va, Lpix, Hpix);
                        auto hps = build_layer_8pts(z, l, h);
                        hierarchical_point_set.push_back({z, hps});
                    }
                    centroid = compute_3d_centroid(hierarchical_point_set);
                } else {
                    // 保留 type=4，cross_sections 可能为空（与 Python 行为一致）
                    centroid = Vec3{ (0.0), (0.0), z2 };
                }
                return;
            } else if (box_aspect_ratio < fov_aspect_ratio) {
                // 竖向直线 type=5
                geometric_type = 5;

                double fov_min_w = box_w;
                double fov_min_h = fov_min_w * fov_aspect_ratio;

                double fov_min_t_x1 = box.x1;
                double fov_min_t_y1 = box.y1;
                double fov_min_center_x2 = fov_min_t_x1 + (fov_min_w/2.0);
                double fov_min_center_y2 = fov_min_t_y1 + (fov_min_h/2.0);

                double fov_min_b_x2 = box.x2;
                double fov_min_b_y2 = box.y2;
                double fov_min_center_x1 = fov_min_b_x2 - (fov_min_w/2.0);
                double fov_min_center_y1 = fov_min_b_y2 - (fov_min_h/2.0);

                auto ROI = roi_points_from_mid_center(
                        fov_min_center_x1, fov_min_center_y1,
                        fov_min_center_x2, fov_min_center_y2,
                        fov_min_w, fov_min_h, Lpix, Hpix
                );
                double z2 = ROI_to_Zoom_fov(Lpix, Hpix, ROI.first, ROI.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
                if (z2 > z_max || z2 < 0) z2 = z_max;

                if (z2 > 1.0) {
                    geometric_type = 1;
                    double z1 = z2 - 1.0;
                    min_z = z1;
                    max_z = z2;

                    auto zs = generate_sequence(z1, z2, 0.5);
                    for (double z : zs) {
                        auto [l,h] = inverse_translate_Z(z, va, Lpix, Hpix);
                        auto hps = build_layer_8pts(z, l, h);
                        hierarchical_point_set.push_back({z, hps});
                    }
                    centroid = compute_3d_centroid(hierarchical_point_set);
                } else {
                    centroid = Vec3{ (0.0), (0.0), z2 };
                }
                return;
            }
        }

        // ======= general case: 类锥体 geometric_type=1 =======
        geometric_type = 1;

        double box_cx = (box.x1 + box.x2)/2.0;
        double box_cy = (box.y1 + box.y2)/2.0;

        auto ROI_max = roi_points_from_mid_center(box_cx, box_cy, box_cx, box_cy, fov_max_w, fov_max_h, Lpix, Hpix);
        double z1 = ROI_to_Zoom_fov(Lpix, Hpix, ROI_max.first, ROI_max.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
        if (z1 < 0 || z1 > z_max) z1 = z_max;

        double z2 = 0.0;
        double fov_min_w=0.0, fov_min_h=0.0;

        if (box_aspect_ratio > fov_aspect_ratio) {
            fov_min_h = box_h;
            fov_min_w = fov_min_h / fov_aspect_ratio;
            auto ROI_min = roi_points_from_mid_center(box_cx, box_cy, box_cx, box_cy, fov_min_w, fov_min_h, Lpix, Hpix);
            z2 = ROI_to_Zoom_fov(Lpix, Hpix, ROI_min.first, ROI_min.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
        } else if (box_aspect_ratio < fov_aspect_ratio) {
            fov_min_w = box_w;
            fov_min_h = fov_min_w * fov_aspect_ratio;
            auto ROI_min = roi_points_from_mid_center(box_cx, box_cy, box_cx, box_cy, fov_min_w, fov_min_h, Lpix, Hpix);
            z2 = ROI_to_Zoom_fov(Lpix, Hpix, ROI_min.first, ROI_min.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
        } else {
            // box_aspect == fov_aspect
            auto ROI_min = std::pair<Vec2,Vec2>{
                    Vec2{(double)std::lround(box.x1),(double)std::lround(box.y1)},
                    Vec2{(double)std::lround(box.x2),(double)std::lround(box.y2)}
            };
            z2 = ROI_to_Zoom_fov(Lpix, Hpix, ROI_min.first, ROI_min.second, va, 1.0, SENSOR_X, SENSOR_Y, 23.0);
        }

        if (z2 < 0 || z2 > z_max) z2 = z_max;
        if (z2 == z1 && z2 == z_max) {
            // Python: print warning then z1=z2-1
            z1 = z2 - 1.0;
        }

        min_z = z1;
        max_z = z2;

        if (z2 - z1 > 0.5) z2 = z2 - 0.5;

        auto zs = generate_sequence(z1, z2, 0.5);
        for (double z : zs) {
            auto [l,h] = inverse_translate_Z(z, va, Lpix, Hpix); // shrink=0.9 默认内置
            auto hps = build_layer_8pts(z, l, h);
            hierarchical_point_set.push_back({z, hps});
        }

        centroid = compute_3d_centroid(hierarchical_point_set);
    }

    std::string get_gosa_input_json_cs() const {
        // {"centroid":[...], "cross_sections":[ [z, [[p,t],...]], ... ]}
        std::ostringstream oss;
        oss << std::setprecision(15);
        oss << "{\"centroid\":["
            << centroid.x << "," << centroid.y << "," << centroid.z
            << "],\"cross_sections\":[";
        for (size_t i=0;i<hierarchical_point_set.size();++i){
            const double z = hierarchical_point_set[i].first;
            const auto& pts = hierarchical_point_set[i].second;
            if (i) oss << ",";
            oss << "[" << z << ",[";
            for (size_t k=0;k<pts.size();++k){
                if (k) oss << ",";
                oss << "[" << pts[k].x << "," << pts[k].y << "]";
            }
            oss << "]]";
        }
        oss << "]}";
        return oss.str();
    }
};

// ===================== box 工具（与 Python 一致） =====================
static Box scale_box(const Box& box, double scale_factor, double img_width, double img_height) {
    double x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2;

    double cx = (x1+x2)/2.0;
    double cy = (y1+y2)/2.0;
    double w  = (x2-x1);
    double h  = (y2-y1);

    double max_scale_x = std::min( (cx)/(w/2.0), (img_width - cx)/(w/2.0) );
    double max_scale_y = std::min( (cy)/(h/2.0), (img_height - cy)/(h/2.0) );
    double max_scale = std::min(max_scale_x, max_scale_y);

    if (scale_factor > max_scale) scale_factor = max_scale;

    double new_w = w * scale_factor;
    double new_h = h * scale_factor;

    double nx1 = std::max(0.0, cx - new_w/2.0);
    double ny1 = std::max(0.0, cy - new_h/2.0);
    double nx2 = std::min(img_width, cx + new_w/2.0);
    double ny2 = std::min(img_height, cy + new_h/2.0);

    return Box{nx1,ny1,nx2,ny2};
}

static std::vector<Geometric> boxs_to_gs(
        const Context& ctx,
        const ZoomTable& df_zoom,
        const std::vector<Box>& boxes,
        const Vec3& pos_now,
        double proportion,
        double scale_factor,
        double shrink_factor,
        double canvas_width,
        double canvas_height,
        double z_max,
        const ViewingAngles& va,
        double p_offset,
        double t_offset,
        P1P2Mode forward_mode,
        bool K_simple_mode
) {
    std::vector<Geometric> geometrics;
    geometrics.reserve(boxes.size());

    for (size_t i=0;i<boxes.size();++i){
        Box n_box = scale_box(boxes[i], scale_factor, canvas_width, canvas_height);

        Geometric g(
                ctx,
                df_zoom,
                (int)i,
                pos_now,
                n_box,
                proportion,
                canvas_width,
                canvas_height,
                z_max,
                va,
                p_offset,
                t_offset,
                shrink_factor,
                canvas_width,
                canvas_height,
                forward_mode,
                K_simple_mode
        );
        geometrics.push_back(std::move(g));
    }
    return geometrics;
}

static std::string g_gs_str(const std::vector<Geometric>& geometrics, const Vec3& pos_now) {
    std::ostringstream oss;
    oss << std::setprecision(15);

    oss << "{\"targets\":[";
    // 起点
    oss << "{\"centroid\":["
        << pos_now.x << "," << pos_now.y << "," << pos_now.z
        << "]}";

    for (const auto& g : geometrics) {
        if (g.geometric_type == 1 || g.geometric_type == 4 || g.geometric_type == 5) {
            oss << "," << g.get_gosa_input_json_cs();
        }
    }
    oss << "]}";
    return oss.str();
}

// ===================== API 参数映射 =====================
static P1P2Mode forward_mode_from_int(int mode) {
    // 你给的示例备注 0/1/2/3，这里按：
    // 0=all, 1=only_y, 2=only_x, 3=no
    switch(mode){
        case 0: return P1P2Mode::ALL;
        case 1: return P1P2Mode::ONLY_Y;
        case 2: return P1P2Mode::ONLY_X;
        case 3: return P1P2Mode::NO;
        default: return P1P2Mode::ALL;
    }
}

// ===================== main + Crow HTTP =====================
static std::string getenv_or(const char* k, const std::string& defv) {
    const char* v = std::getenv(k);
    if (!v || !*v) return defv;
    return std::string(v);
}
static int getenv_int_or(const char* k, int defv) {
    const char* v = std::getenv(k);
    if (!v || !*v) return defv;
    try { return std::stoi(v); } catch(...) { return defv; }
}

int main() {
    // ===== 启动加载 CSV + XGBoost 模型 =====
    const std::string csv_path    = getenv_or("CSV_PATH", "zoom_to_P2.csv");
    const std::string modelx_path = getenv_or("MODEL_X_PATH", "train_data_x_new_new_new_no_K_no_feature.json");
    const std::string modely_path = getenv_or("MODEL_Y_PATH", "train_data_y_new_new_new_no_K_no_feature.json");
    const int port                = getenv_int_or("PORT", 18080);

    static Context ctx;
    try {
        ctx.table   = ZoomTable::load(csv_path);
        ctx.model_x.load(modelx_path);
        ctx.model_y.load(modely_path);
    } catch (const std::exception& e) {
        std::cerr << "[Startup] Failed: " << e.what() << std::endl;
        return 1;
    }

    crow::SimpleApp app;

    CROW_ROUTE(app, "/model").methods(crow::HTTPMethod::Post)
            ([&](const crow::request& req){
                auto body = crow::json::load(req.body);
                if (!body) {
                    crow::json::wvalue err;
                    err["error"] = "invalid json";
                    return crow::response(400, err);
                }

                try {
                    // ===== 解析标量参数 =====
                    double proportion    = body["proportion"].d();
                    double n             = body["n"].d();   // scale_factor
                    double s             = body["s"].d();   // shrink
                    double canvas_width  = body["canvas_width"].d();
                    double canvas_height = body["canvas_height"].d();
                    double z_max         = body["z_max"].d();
                    double p_offset      = body["p_offset"].d();
                    double t_offset      = body["t_offset"].d();
                    int mode             = body["mode"].i(); // 0/1/2/3 -> forward_mode

                    // ===== pos_now =====
                    auto pos_arr = body["pos_now"];
                    if (!pos_arr || pos_arr.size() != 3) {
                        crow::json::wvalue err;
                        err["error"] = "pos_now must be array of length 3";
                        return crow::response(400, err);
                    }
                    Vec3 pos_now{ pos_arr[0].d(), pos_arr[1].d(), pos_arr[2].d() };

                    // ===== va =====
                    auto va_arr = body["va"];
                    if (!va_arr || va_arr.size() != 2) {
                        crow::json::wvalue err;
                        err["error"] = "va must be array of length 2";
                        return crow::response(400, err);
                    }
                    ViewingAngles va{ va_arr[0].d(), va_arr[1].d() };

                    // ===== boxes =====
                    std::vector<Box> boxes;
                    auto boxes_arr = body["boxes"];
                    if (!boxes_arr) {
                        crow::json::wvalue err;
                        err["error"] = "boxes is required";
                        return crow::response(400, err);
                    }
                    boxes.reserve(boxes_arr.size());
                    for (size_t i=0;i<boxes_arr.size();++i){
                        auto b = boxes_arr[i];
                        if (!b || b.size() != 4) {
                            crow::json::wvalue err;
                            err["error"] = "each box must be [x1,y1,x2,y2]";
                            return crow::response(400, err);
                        }
                        boxes.push_back(Box{ b[0].d(), b[1].d(), b[2].d(), b[3].d() });
                    }

                    // ===== 计时 & 计算 =====
                    auto start = std::chrono::high_resolution_clock::now();

                    P1P2Mode forward_mode = forward_mode_from_int(mode);
                    bool K_simple_mode = true; // 与 Python 默认一致（simple）

                    auto geometrics = boxs_to_gs(
                            ctx, ctx.table,
                            boxes, pos_now,
                            proportion,
                            n,     // scale_factor
                            s,     // shrink
                            canvas_width, canvas_height,
                            z_max,
                            va,
                            p_offset, t_offset,
                            forward_mode,
                            K_simple_mode
                    );

                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                    double duration_ms = (double)duration_ns / 1'000'000.0;

                    std::string json_str = g_gs_str(geometrics, pos_now);

                    crow::json::wvalue result;
                    result["t"] = duration_ms;
                    result["json"] = json_str;
                    return crow::response(result);
                }
                catch (const std::exception& e) {
                    crow::json::wvalue err;
                    err["error"] = std::string("exception: ") + e.what();
                    return crow::response(500, err);
                }
            });

    std::cout << "Server listening on http://0.0.0.0:" << port << "/model" << std::endl;
    app.port((uint16_t)port).multithreaded().run();
    return 0;
}
