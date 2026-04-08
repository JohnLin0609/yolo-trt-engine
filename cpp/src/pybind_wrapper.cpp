#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/core.hpp>
#include "yolo_engine.h"

namespace py = pybind11;

// Convert numpy array (HWC, uint8) to cv::Mat without copying data
static cv::Mat numpy_to_mat(py::array_t<uint8_t>& arr)
{
    auto buf = arr.request();
    if (buf.ndim != 3)
        throw std::runtime_error("Expected 3D array (H, W, C)");

    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = buf.shape[2];
    if (c != 3)
        throw std::runtime_error("Expected 3-channel BGR image");

    // Wrap numpy data directly — no copy
    return cv::Mat(h, w, CV_8UC3, buf.ptr);
}

// Convert Detection results to Python list[dict]
// Matches the same format as the Python YoloEngine
static py::list detections_to_py(const std::vector<Detection>& dets)
{
    py::list results;
    for (const auto& d : dets) {
        py::dict item;
        item["classid"] = d.class_id;
        item["score"] = d.score;

        // bbox as numpy array [x1, y1, x2, y2] — same as Python version
        py::array_t<float> bbox(4);
        auto ptr = bbox.mutable_data();
        ptr[0] = d.x1; ptr[1] = d.y1;
        ptr[2] = d.x2; ptr[3] = d.y2;
        item["bbox"] = bbox;

        results.append(item);
    }
    return results;
}

PYBIND11_MODULE(yolo_engine_cpp, m)
{
    m.doc() = "YOLO TensorRT C++ inference engine with fused CUDA preprocessing";

    py::class_<YoloEngine>(m, "YoloEngine")
        .def(py::init<const std::string&, float, float>(),
             py::arg("engine_path"),
             py::arg("conf_thresh") = 0.7f,
             py::arg("iou_thresh") = 0.1f,
             "Load a TensorRT engine file. All GPU memory is allocated once here.")

        .def("inference",
             [](YoloEngine& self, py::array_t<uint8_t>& image) -> py::list {
                 cv::Mat mat = numpy_to_mat(image);
                 auto dets = self.inference(mat);
                 return detections_to_py(dets);
             },
             py::arg("image"),
             "Run inference on a BGR uint8 numpy image. Returns list[dict] "
             "with keys: classid (int), score (float), bbox (ndarray[4]).")

        .def_property_readonly("input_width", &YoloEngine::input_width)
        .def_property_readonly("input_height", &YoloEngine::input_height);
}
