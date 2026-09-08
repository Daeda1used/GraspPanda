// Python array interface to the BSD-licensed Grasp Pose Generator.
#include <gpg/candidates_generator.h>
#include <gpg/cloud_camera.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <mutex>
namespace py = pybind11;

py::array_t<double> generate(py::array_t<float, py::array::c_style | py::array::forcecast> points,
                            int samples, int threads, unsigned int seed) {
    if (points.ndim() != 2 || points.shape(1) != 3 || points.shape(0) < 5)
        throw std::invalid_argument("GPG requires a finite Nx3 cloud with at least five points");
    if (samples < 1 || samples > 1000000 || threads < 1 || threads > 64)
        throw std::invalid_argument("Invalid GPG sample or thread count");
    const auto input = points.unchecked<2>();
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    cloud->reserve(points.shape(0));
    for (py::ssize_t i = 0; i < points.shape(0); ++i) {
        pcl::PointXYZRGBA p;
        for (int j = 0; j < 3; ++j)
            if (!std::isfinite(input(i, j))) throw std::invalid_argument("GPG input contains nonfinite points");
        p.x = input(i, 0); p.y = input(i, 1); p.z = input(i, 2);
        cloud->push_back(p);
    }
    HandSearch::Parameters hand;
    hand.finger_width_ = .01; hand.hand_outer_diameter_ = .12;
    hand.hand_depth_ = .04; hand.hand_height_ = .02; hand.init_bite_ = .01;
    hand.nn_radius_frames_ = .01; hand.num_orientations_ = 8;
    hand.num_samples_ = samples; hand.num_threads_ = threads; hand.rotation_axis_ = 2;
    CandidatesGenerator::Parameters settings;
    settings.num_samples_ = samples; settings.num_threads_ = threads;
    settings.max_grasp_num_ = 0; settings.plot_grasps_ = false; settings.plot_normals_ = false;
    settings.remove_statistical_outliers_ = false; settings.voxelize_ = true;
    settings.workspace_ = {-10., 10., -10., 10., -10., 10.};
    Eigen::Matrix3Xd viewpoint = Eigen::Matrix3Xd::Zero(3, 1);
    CloudCamera observation(cloud, 0, viewpoint);
    CandidatesGenerator generator(settings, hand);
    std::vector<Grasp> candidates;
    {
        py::gil_scoped_release release;
        static std::mutex random_state;
        std::lock_guard<std::mutex> guard(random_state);
        std::srand(seed);
        generator.preprocessPointCloud(observation);
        candidates = generator.generateGraspCandidates(observation);
    }
    py::array_t<double> output({static_cast<py::ssize_t>(candidates.size()), py::ssize_t(17)});
    auto result = output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(candidates.size()); ++i) {
        const auto &g = candidates[i];
        result(i, 0) = 1.; result(i, 1) = g.getGraspWidth();
        result(i, 2) = .02; result(i, 3) = .04;
        for (int axis = 0; axis < 3; ++axis) {
            result(i, 4 + axis * 3) = g.getApproach()(axis);
            result(i, 5 + axis * 3) = g.getBinormal()(axis);
            result(i, 6 + axis * 3) = g.getAxis()(axis);
            result(i, 13 + axis) = g.getGraspBottom()(axis);
        }
        result(i, 16) = -1.;
    }
    return output;
}

PYBIND11_MODULE(_grasppanda_gpg, module) {
    module.def("generate", &generate, py::arg("points"), py::arg("samples") = 1000000,
               py::arg("threads") = 4, py::arg("seed") = 0);
}
