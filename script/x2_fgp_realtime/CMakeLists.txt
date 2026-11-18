cmake_minimum_required(VERSION 3.18)
project(carticulate_fgp_native LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
pybind11_add_module(carticulate MODULE
  "${BINDING_FILE}"
  "${CART_SOURCE}/pybind/utils.cpp"
  "${CART_SOURCE}/kinematics/kinematic_armature.cpp"
  "${CART_SOURCE}/kinematics/kinematic_model.cpp"
  "${CART_SOURCE}/dynamics/dynamic_armature.cpp"
  "${CART_SOURCE}/dynamics/dynamic_model.cpp")
target_include_directories(carticulate PRIVATE "${CART_SOURCE}/pybind" "${SOPHUS_SOURCE}")
target_link_libraries(carticulate PRIVATE Eigen3::Eigen fmt::fmt)
target_compile_features(carticulate PRIVATE cxx_std_17)
set_target_properties(carticulate PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUTPUT}")
