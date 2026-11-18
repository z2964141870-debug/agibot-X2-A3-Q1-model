"""Build original carticulate dynamics with its original reduced Python binding."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def selected_binding(text):
    kinematic = text[text.index("    py::class_<KinematicArmature>"):text.index("    py::enum_<RobustKernel>")]
    dynamic = text[text.index("    py::class_<DynamicArmature>"):text.index("#ifndef FULL_ESKF_WITH_POS_VEL")]
    return ("#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <pybind11/eigen.h>\n"
        '#include "utils.h"\n#include "../kinematics/kinematic_model.h"\n#include "../dynamics/dynamic_model.h"\n'
        'namespace py = pybind11;\nPYBIND11_MODULE(carticulate, m) {\n'
        + kinematic + dynamic + "\n}\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cart-source", type=Path, required=True)
    parser.add_argument("--sophus-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new build output; preserve previous attempts")
    args.output.mkdir(parents=True)
    original = args.cart_source / "pybind/pybind.cpp"
    text = original.read_text(encoding="utf-8-sig")
    binding = args.output / "binding.cpp"
    # Original includes are relative to the upstream binding directory.
    generated = selected_binding(text).replace('"../kinematics/', '"' + str(args.cart_source / "kinematics") + "/").replace('"../dynamics/', '"' + str(args.cart_source / "dynamics") + "/")
    binding.write_text(generated)
    native = args.output / "module"
    cmake_dir = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    before = time.perf_counter()
    subprocess.run(["cmake", "-S", str(Path(__file__).parent), "-B", str(args.output / "build"),
        "-DCMAKE_BUILD_TYPE=Release", "-DPYBIND11_FINDPYTHON=ON", "-DPython3_EXECUTABLE="+sys.executable,
        "-Dpybind11_DIR="+cmake_dir, "-DCART_SOURCE="+str(args.cart_source),
        "-DSOPHUS_SOURCE="+str(args.sophus_source), "-DBINDING_FILE="+str(binding),
        "-DNATIVE_OUTPUT="+str(native)], check=True, timeout=60)
    subprocess.run(["cmake", "--build", str(args.output / "build"), "--parallel", "2"], check=True, timeout=420)
    modules = list(native.glob("carticulate*.so"))
    if len(modules) != 1:
        raise RuntimeError("Expected one native extension")
    result = dict(status="CARTICULATE_REDUCED_NATIVE_BUILD_PASSED", wall_s=time.perf_counter()-before,
        module=str(modules[0]), bytes=modules[0].stat().st_size,
        sha256=hashlib.sha256(modules[0].read_bytes()).hexdigest(),
        upstream_binding_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        generated_binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),
        binding_scope="Original KinematicArmature/KinematicModel and DynamicArmature/DynamicModel blocks only. No optimizer or ESKF bindings. Original dynamics and kinematics implementation files unchanged.",
        windows_binary_equivalence="NOT_ESTABLISHED", hardware_commands=0)
    (args.output / "build_receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
