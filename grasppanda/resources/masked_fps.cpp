// Deterministic masked FPS for point-row preserving sampling adapters.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>

void masked_fps_launch(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&);

at::Tensor masked_fps(at::Tensor points, at::Tensor eligible, at::Tensor starts,
                     int64_t count) {
  TORCH_CHECK(points.is_cuda() && points.scalar_type() == at::kFloat &&
              points.is_contiguous(), "points must be contiguous CUDA float32");
  TORCH_CHECK(points.dim() == 3 && points.size(2) == 3 &&
              points.size(0) > 0 && points.size(1) > 0,
              "points must have shape [B, N, 3] with nonempty B and N");
  const c10::cuda::CUDAGuard guard(points.device());
  const auto batch = points.size(0), n = points.size(1);
  TORCH_CHECK(batch <= 65535 && n <= std::numeric_limits<int>::max() / 3 &&
              count >= 1 && count <= n, "invalid batch or sample count");
  TORCH_CHECK(eligible.device() == points.device() && eligible.scalar_type() == at::kBool &&
              eligible.is_contiguous() && eligible.dim() == 2 &&
              eligible.size(0) == batch && eligible.size(1) == n,
              "eligible must be contiguous CUDA bool [B, N] on the point device");
  TORCH_CHECK(starts.device() == points.device() && starts.scalar_type() == at::kLong &&
              starts.is_contiguous() && starts.dim() == 1 && starts.size(0) == batch,
              "starts must be contiguous CUDA int64 [B] on the point device");
  TORCH_CHECK(at::isfinite(points).all().item<bool>() && points.abs().max().item<float>() < 1e18f,
              "point coordinates must be finite and have representable squared distances");
  TORCH_CHECK((eligible.sum(1) >= count).all().item<bool>(), "not enough eligible rows");
  TORCH_CHECK((starts >= 0).all().item<bool>() && (starts < n).all().item<bool>(),
              "start index outside point rows");
  TORCH_CHECK(eligible.gather(1, starts.unsqueeze(1)).all().item<bool>(), "start row is filtered out");
  auto output = at::empty({batch, count}, points.options().dtype(at::kLong));
  auto distance = at::full({batch, n}, std::numeric_limits<float>::infinity(), points.options());
  masked_fps_launch(points, eligible, starts, distance, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &masked_fps);
  m.attr("api_version") = 1;
}
