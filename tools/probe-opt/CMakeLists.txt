add_llvm_executable(probe-opt
  probe-opt.cpp
)

llvm_update_compile_flags(probe-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(probe-opt
  PRIVATE
  MLIRProbeDialect
  MLIRProbeTransforms

  # MLIR dependencies
  MLIRMlirOptMain
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_link_libraries(mlir-opt)
