"""Static checks for D5's source map. These do NOT compile or execute CUDA."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/cuda/ntt_kernels.cu").read_text()
HEADER = (ROOT / "include/ntt/cuda_ntt.hpp").read_text()


def body_after(marker):
    """Extract a simple balanced source body, not a C++ parser or compiler."""
    start = SOURCE.index(marker)
    start = SOURCE.index("{", start)
    depth = 1
    end = start + 1
    while depth:
        depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
        end += 1
    return SOURCE[start:end]


def method(name):
    return body_after("CudaNttTransform::" + name + "(")


class CudaDataflowSourceTests(unittest.TestCase):
    def test_six_allocation_sizes_and_normal_cleanup(self):
        expected = {
            "device_operand_a": "value_count",
            "device_operand_b": "value_count",
            "device_twiddle_powers": "twiddle_count",
            "device_inverse_twiddle_powers": "twiddle_count",
            "device_moduli": "impl.limb_count",
            "device_inverse_transform_lengths": "impl.limb_count",
        }
        allocations = re.findall(
            r"cudaMalloc\(&impl\.(device_\w+),\s*([\w.]+) \* sizeof\(uint64_t\)\)",
            SOURCE,
        )
        self.assertEqual(dict(allocations), expected)
        self.assertEqual(len(allocations), 6)
        frees = re.findall(r"cudaFree\((device_\w+)\)", SOURCE)
        self.assertEqual(sorted(frees), sorted(expected))
        self.assertIn("value_count = shape.values", SOURCE)
        self.assertIn("twiddle_count = shape.twiddles", SOURCE)
        self.assertIn("std::unique_ptr<Impl> impl_", HEADER)
        self.assertIn("impl_(std::make_unique<Impl>())", SOURCE)

    def test_constructor_uploads_only_constants_before_public_methods(self):
        constructor = SOURCE[SOURCE.index("  Impl &impl = *impl_;"):SOURCE.index("CudaNttTransform::~")]
        self.assertEqual(constructor.count("cudaMemcpy("), 4)
        self.assertEqual(constructor.count("cudaMemcpyHostToDevice"), 4)
        for buffer in ("device_twiddle_powers", "device_inverse_twiddle_powers",
                       "device_moduli", "device_inverse_transform_lengths"):
            self.assertRegex(constructor, r"cudaMemcpy\(impl\." + buffer + r",")
        self.assertLess(constructor.index("cpu_references.emplace_back"),
                        constructor.index("cudaMalloc("))

    def test_public_copy_counts_and_fused_operation_order(self):
        for name, uploads in (("forward", 1), ("inverse", 1),
                              ("pointwise_multiply", 2), ("multiply", 2)):
            body = method(name)
            self.assertEqual(body.count("cudaMemcpyHostToDevice"), uploads, name)
            self.assertEqual(body.count("cudaMemcpyDeviceToHost"), 1, name)
            checked_operands = re.findall(r"detail::validate_values\([^;]*,\s*(\w+)\)", body)
            self.assertEqual(checked_operands,
                             ["values"] if uploads == 1 else
                             ["left_values", "right_values"], name)
            self.assertNotIn("cudaMalloc", body)
            self.assertLess(body.index("DeviceGuard guard"), body.index("cudaMemcpy("))
        body = method("multiply")
        operations = re.findall(
            r"launch_forward_transform\(\*impl_, impl_->(device_operand_[ab])\)|"
            r"(pointwise_multiply_kernel)<<<|(launch_inverse_transform)\(", body)
        self.assertEqual([next(value for value in group if value) for group in operations],
                         ["device_operand_a", "device_operand_b",
                          "pointwise_multiply_kernel", "launch_inverse_transform"])
        self.assertLess(body.rindex("cudaMemcpyHostToDevice"),
                        body.index("launch_forward_transform"))
        self.assertGreater(body.index("cudaMemcpyDeviceToHost"),
                           body.index("launch_inverse_transform"))

    def test_host_interface_shape_and_no_async_api(self):
        self.assertIn("CudaNttTransform(const CudaNttTransform &) = delete", HEADER)
        self.assertIn("std::vector<uint64_t> multiply(std::vector<uint64_t> left_values", HEADER)
        for name in ("n", "limbs", "q", "ref"):
            self.assertNotIn("cudaMemcpy", method(name))
        self.assertNotIn("cudaMemcpyAsync", SOURCE)
        self.assertNotIn("cudaStream_t", HEADER)

    def test_launch_composition_and_limb_selection(self):
        forward = body_after("void launch_forward_transform(")
        inverse = body_after("void launch_inverse_transform(")
        self.assertEqual(re.findall(r"(\w+_kernel)<<<", forward),
                         ["forward_strided_group_kernel", "forward_contiguous_tile_kernel"])
        self.assertEqual(re.findall(r"(\w+_kernel)<<<", inverse),
                         ["inverse_contiguous_tile_kernel", "inverse_strided_group_kernel",
                          "inverse_scale_kernel"])
        for name in ("pointwise_multiply", "multiply"):
            self.assertEqual(re.findall(r"(\w+_kernel)<<<", method(name)),
                             ["pointwise_multiply_kernel"])
        for name in ("pointwise_multiply_kernel", "inverse_scale_kernel"):
            body = body_after("void " + name + "(")
            self.assertIn("value_index / transform_length", body)
            self.assertNotIn("blockIdx.y", body)

    def test_65536_allocation_and_transfer_arithmetic(self):
        for n, operand, constants, total in (
            (65536, 1572864, 3145824, 6291552),
        ):
            k = 3
            self.assertEqual(8*k*n, operand)
            self.assertEqual(2*8*k*(n+1) + 2*8*k, constants)
            self.assertEqual(2*operand + constants, total)
            self.assertEqual(total, 32*k*(n+1))
        self.assertEqual(9*1572864, 14155776)
        self.assertEqual(3*1572864, 4718592)
