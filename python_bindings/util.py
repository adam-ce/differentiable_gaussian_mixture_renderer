import platform
import os
import subprocess

def find_source_files(source_dir):
    source_files = []
    extensions = ('.cpp', '.cu')

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extensions):
                source_files.append(os.path.join(root, file))

    return source_files

def clone_and_checkout_repos(repos, target_dir):
    if os.path.exists(target_dir):
        return
    
    os.makedirs(target_dir)

    for repo, ref in repos:
        repo_name = os.path.basename(repo).replace('.git', '')
        repo_path = os.path.join(target_dir, repo_name)

        try:
            clone_result = subprocess.run(['git', 'clone', repo], cwd=target_dir, check=True, capture_output=True, text=True)
            print(f"Successfully cloned {repo}")
            print(clone_result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo}")
            print(e.stderr)
            continue

        try:
            checkout_result = subprocess.run(['git', 'checkout', ref], cwd=repo_path, check=True, capture_output=True, text=True)
            print(f"Successfully checked out {ref} in {repo_name}")
            print(checkout_result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to checkout {ref} in {repo_name}")
            print(e.stderr)

repos = [
    ('https://github.com/g-truc/glm.git', '673a963a0f1eb82f5fcef00b7b873371555e5814'),
    ('https://github.com/kthohr/gcem.git', 'master'),
    ('git@github.com:cg-tuwien/cuda_stroke.git', 'main'),
    ('git@github.com:cg-tuwien/cuda_whack.git', 'main')
]


source_dir = os.path.dirname(__file__)
print(f"util.py: source_dir={source_dir}")

if False:
    # this function is not good, but good enough for now.
    # if you have any problems, try deleting all files in ./extern
    clone_and_checkout_repos(repos, target_dir=f"{source_dir}/extern")
    extra_include_paths = [
        source_dir + "/extern/cuda_stroke/src/", 
        source_dir + "/extern/cuda_whack/src", 
        source_dir + "/extern/gcem/include", 
        source_dir + "/extern/glm/", 
        source_dir + "/../src"]
else:
    extern_dir = source_dir + "/../../SIBR_viewers/3rdparty"
    extra_include_paths = [
        extern_dir + "/stroke/src/", 
        extern_dir + "/whack/src", 
        extern_dir + "/gcem/include", 
        extern_dir + "/glm/", 
        source_dir + "/../src"]

profiler_sync_extra = ""
# if gmc.config.enable_profiler_synchronisation:
#     profiler_sync_extra = "-DGPE_PROFILER_BUILD"

cuda_extra_cuda_cflags = ["-O3",  "--use_fast_math", "--expt-extended-lambda", "-std=c++20", " -DNDEBUG -DGLM_FORCE_LEFT_HANDED -DGLM_FORCE_QUAT_DATA_WXYZ --expt-relaxed-constexpr",  profiler_sync_extra]  # 
if platform.system() == "Windows":
    cuda_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/DNDEBUG"]
    cpp_extra_cflags = ["/openmp", "/O2", "/fp:fast", "/std:c++20", "/DNDEBUG"]
    cuda_extra_cuda_cflags.append("-Xcompiler=/openmp,/O2,/fp:fast,/DGPE_NO_CUDA_ERROR_CHECKING")
else:
    # cuda_extra_cflags = ["-O3", "-ffast-math", "-march=native", "--expt-extended-lambda --expt-relaxed-constexpr", "-std=c++20", "-DNDEBUG", profiler_sync_extra]
    cuda_extra_cflags = ["-O3 -ffast-math -march=native --expt-extended-lambda --expt-relaxed-constexpr -std=c++20 -DNDEBUG -DGLM_FORCE_XYZW_ONLY -DGLM_ENABLE_EXPERIMENTAL", profiler_sync_extra]
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++20", " -DNDEBUG -DGLM_FORCE_XYZW_ONLY -DGLM_ENABLE_EXPERIMENTAL", profiler_sync_extra]  # , "-DNDEBUG", "-DGPE_NO_CUDA_ERROR_CHECKING"
    cuda_extra_cuda_cflags.append(f"-Xcompiler -fopenmp {profiler_sync_extra} -DNDEBUG -DGLM_FORCE_XYZW_ONLY -DGLM_ENABLE_EXPERIMENTAL")  #  -DNDEBUG"

