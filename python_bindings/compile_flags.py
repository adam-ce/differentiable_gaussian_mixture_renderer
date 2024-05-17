import platform
import os
import gmc.config

source_dir = os.path.dirname(__file__)
print(f"compile_flags.py: source_dir={source_dir}")
extra_include_paths = [source_dir + "/../glm/", source_dir + "/", source_dir + "/../yamc/include", source_dir + "/../gcem/include", source_dir + "/.."]  # source_dir + "/../cub/",

profiler_sync_extra = ""
if gmc.config.enable_profiler_synchronisation:
    profiler_sync_extra = "-DGPE_PROFILER_BUILD"

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--expt-extended-lambda", "-std=c++17", " --expt-relaxed-constexpr", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG", profiler_sync_extra]  # , "-DNDEBUG"
if platform.system() == "Windows":
    cuda_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/DGPE_NO_CUDA_ERROR_CHECKING", "/DNDEBUG", "/DGPE_LIMIT_N_REDUCTION"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++17", "/DNDEBUG", "/DGPE_LIMIT_N_REDUCTION"]
    cuda_extra_cuda_cflags.append("-Xcompiler=/openmp,/O2,/fp:fast,/DGPE_NO_CUDA_ERROR_CHECKING")
else:
    cuda_extra_cflags = ["-O3", "-ffast-math", "-march=native", "-std=c++17", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG", profiler_sync_extra]
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++17", "-DGPE_LIMIT_N_REDUCTION", "-DNDEBUG", profiler_sync_extra]  # , "-DNDEBUG", "-DGPE_NO_CUDA_ERROR_CHECKING"
    cuda_extra_cuda_cflags.append(f"-Xcompiler -fopenmp -DNDEBUG {profiler_sync_extra}")  #  -DNDEBUG"

