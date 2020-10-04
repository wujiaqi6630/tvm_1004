# DPU Module
if(USE_DPU)
  message(STATUS "Build with DPU support")
  #file(GLOB RUNTIME_DPU_SRCS src/runtime/dpu/*.cc)
  #list(APPEND RUNTIME_SRCS ${RUNTIME_DPU_SRCS})
  #list(APPEND COMPILER_SRCS src/codegen/opt/build_dpu_on.cc)
#else(USE_DPU)
  #list(APPEND COMPILER_SRCS src/codegen/opt/build_dpu_off.cc)
endif(USE_DPU)

































