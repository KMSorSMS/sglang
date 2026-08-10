#!/bin/bash
# 容器内执行：sgl-kernel wheel 缓存命中则直接 pip install，否则编译并写回缓存
# 用法: env/build_sgl_kernel.sh <EVENT_TYPE> <BASE_IMAGE_TAG> <CI_WORK_DIR> <WHEEL_CACHE_ROOT> <KERNEL_DIR> [FETCHCONTENT_CACHE_ROOT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sanitize_fetchcontent_cache() {
    local cache_root=$1
    local build_dirs=()
    local subbuild_dir

    if [[ -z "${cache_root}" || ! -d "${cache_root}" ]]; then
        echo "[SGLang CI] ERROR: invalid FetchContent cache directory: ${cache_root:-<empty>}" >&2
        return 2
    fi

    shopt -s nullglob
    build_dirs=("${cache_root}"/*-build)
    if (( ${#build_dirs[@]} > 0 )); then
        rm -rf -- "${build_dirs[@]}"
    fi

    for subbuild_dir in "${cache_root}"/*-subbuild; do
        rm -rf -- \
            "${subbuild_dir}/CMakeCache.txt" \
            "${subbuild_dir}/CMakeFiles" \
            "${subbuild_dir}/Makefile" \
            "${subbuild_dir}/cmake_install.cmake" \
            "${subbuild_dir}/build.ninja" \
            "${subbuild_dir}/rules.ninja" \
            "${subbuild_dir}/.ninja_deps" \
            "${subbuild_dir}/.ninja_log"
    done
    shopt -u nullglob

    echo "[SGLang CI] FetchContent cache build state sanitized: ${cache_root}"
}

record_fetchcontent_revisions() {
    local kernel_dir=$1

    python3 - "${kernel_dir}" <<'PY'
import re
import sys
from pathlib import Path

kernel_dir = Path(sys.argv[1])
cmake_files = [kernel_dir / "CMakeLists.txt", *sorted((kernel_dir / "cmake").glob("*.cmake"))]
declaration_pattern = re.compile(r"FetchContent_Declare\(\s*([^\s)]+)(.*?)\n\s*\)", re.DOTALL)
revision_count = 0

for cmake_file in cmake_files:
    if not cmake_file.is_file():
        continue
    content = cmake_file.read_text(encoding="utf-8")
    for match in declaration_pattern.finditer(content):
        dependency, body = match.groups()
        url_match = re.search(r"^\s*URL\s+(\S+)", body, re.MULTILINE)
        tag_match = re.search(r"^\s*GIT_TAG\s+(\S+)", body, re.MULTILINE)
        hash_match = re.search(r"^\s*URL_HASH\s+(\S+)", body, re.MULTILINE)
        if tag_match:
            revision = tag_match.group(1)
        elif url_match and "/archive/" in url_match.group(1):
            revision = url_match.group(1).split("/archive/", 1)[1]
            revision = re.sub(r"\.(?:tar\.gz|tgz|zip)$", "", revision)
        else:
            continue
        source = cmake_file.relative_to(kernel_dir)
        checksum = hash_match.group(1) if hash_match else "<none>"
        print(
            "[SGLang CI] FetchContent "
            f"dependency={dependency} revision={revision} "
            f"checksum={checksum} declaration={source}"
        )
        revision_count += 1

if revision_count == 0:
    raise SystemExit(f"no immutable FetchContent revisions found under {kernel_dir}")
PY
}

if [[ "${1:-}" == "--sanitize-fetchcontent-cache" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 --sanitize-fetchcontent-cache <fetchcontent-cache-dir>" >&2
        exit 2
    fi
    sanitize_fetchcontent_cache "$2"
    exit 0
fi

if [[ "${1:-}" == "--list-fetchcontent-revisions" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 --list-fetchcontent-revisions <sgl-kernel-dir>" >&2
        exit 2
    fi
    record_fetchcontent_revisions "$2"
    exit 0
fi

EVENT_TYPE=$1
BASE_IMAGE_TAG=$2
CI_WORK_DIR=$3
WHEEL_CACHE_ROOT=$4
KERNEL_DIR=$5
FETCHCONTENT_CACHE_ROOT="${6:-${WHEEL_CACHE_ROOT}}"

# Copy sgl-kernel source from workspace mount to editable install target
if [ -d "${KERNEL_DIR}" ] && [ -d python/sglang/kernels/aot ]; then
    echo "[SGLang CI] 拷贝 sgl-kernel 源码到 ${KERNEL_DIR} ..."
    \cp -rf python/sglang/kernels/aot/* "${KERNEL_DIR}/"
fi


version_ge() {
    local current=$1
    local minimum=$2
    [[ "$(printf '%s\n%s\n' "${minimum}" "${current}" | sort -V | head -n1)" == "${minimum}" ]]
}

find_installed_sgl_kernel_so() {
    local relative_pattern=$1
    python3 - "${relative_pattern}" <<'PY'
import glob
import sys
import sysconfig

relative_pattern = sys.argv[1]
roots = []
for key in ("platlib", "purelib"):
    value = sysconfig.get_paths().get(key)
    if value and value not in roots:
        roots.append(value)

for root in roots:
    matches = sorted(glob.glob(f"{root}/sgl_kernel/{relative_pattern}"))
    if matches:
        print(matches[0])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

require_cubin_arch() {
    local so_path=$1
    local expected_arch=$2
    local elf_list
    if [[ ! -f "${so_path}" ]]; then
        echo "[SGLang CI] ERROR: missing ${so_path}"
        exit 1
    fi
    echo "[SGLang CI] 验证 ${so_path} 包含 ${expected_arch}"
    elf_list=$(cuobjdump --list-elf "${so_path}")
    if ! grep -Fq ".${expected_arch}.cubin" <<< "${elf_list}"; then
        echo "[SGLang CI] ERROR: ${so_path} does not contain ${expected_arch}"
        echo "${elf_list}"
        exit 1
    fi
}

CUDA_TOOLKIT_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
if [[ -z "${CUDA_TOOLKIT_VERSION}" ]]; then
    echo "[SGLang CI] ERROR: failed to detect CUDA Toolkit version from nvcc"
    nvcc --version || true
    exit 1
fi

TARGET_ARCHS_RAW="${SGLANG_KERNEL_TARGET_ARCHS:-sm90,sm90a,sm100a,sm103a}"
TARGET_ARCHS=$(echo "${TARGET_ARCHS_RAW}" | tr -d '[:space:]')
IFS=',' read -r -a TARGET_ARCH_ARRAY <<< "${TARGET_ARCHS}"
read -r JD_CI_SGL_KERNEL_BUILD_MAX_JOBS JD_CI_SGL_KERNEL_NVCC_THREADS < <(
    JD_CI_SGL_KERNEL_BUILD_MAX_JOBS="${JD_CI_SGL_KERNEL_BUILD_MAX_JOBS:-}" \
    JD_CI_SGL_KERNEL_NVCC_THREADS="${JD_CI_SGL_KERNEL_NVCC_THREADS:-}" \
        bash "${SCRIPT_DIR}/sgl_kernel_parallelism.sh" "$(nproc)"
)

REQUIRE_SM90=0
REQUIRE_SM90A=0
REQUIRE_SM100A=0
REQUIRE_SM103A=0
for arch in "${TARGET_ARCH_ARRAY[@]}"; do
    case "${arch}" in
        sm90|sm_90)
            REQUIRE_SM90=1
            ;;
        h20|h100|h200|hopper)
            REQUIRE_SM90=1
            REQUIRE_SM90A=1
            ;;
        sm90a|sm_90a)
            REQUIRE_SM90A=1
            ;;
        sm100a|sm_100a|b200)
            REQUIRE_SM100A=1
            ;;
        sm103a|sm_103a|b300)
            REQUIRE_SM103A=1
            ;;
        blackwell)
            REQUIRE_SM100A=1
            REQUIRE_SM103A=1
            ;;
        *)
            echo "[SGLang CI] ERROR: unsupported SGLANG_KERNEL_TARGET_ARCHS item: ${arch}"
            exit 1
            ;;
    esac
done

if [[ ${REQUIRE_SM90A} -eq 1 ]] && ! version_ge "${CUDA_TOOLKIT_VERSION}" "12.4"; then
    echo "[SGLang CI] ERROR: sm90a requires CUDA Toolkit >= 12.4, got ${CUDA_TOOLKIT_VERSION}"
    exit 1
fi
if [[ ${REQUIRE_SM100A} -eq 1 ]] && ! version_ge "${CUDA_TOOLKIT_VERSION}" "12.8"; then
    echo "[SGLang CI] ERROR: sm100a requires CUDA Toolkit >= 12.8, got ${CUDA_TOOLKIT_VERSION}"
    exit 1
fi
if [[ ${REQUIRE_SM103A} -eq 1 ]] && ! version_ge "${CUDA_TOOLKIT_VERSION}" "13.0"; then
    echo "[SGLang CI] ERROR: sm103a requires CUDA Toolkit >= 13.0, got ${CUDA_TOOLKIT_VERSION}"
    exit 1
fi

TARGET_ARCH_KEY=$(echo "${TARGET_ARCHS}" | tr ',' '-')
WHEEL_CACHE_DIR="${WHEEL_CACHE_ROOT}/cuda${CUDA_TOOLKIT_VERSION}-${TARGET_ARCH_KEY}/wheels"
CACHE_BUILD_INFO="${WHEEL_CACHE_DIR}/build_info.txt"
SGL_KERNEL_FETCHCONTENT_BASE_DIR="${FETCHCONTENT_CACHE_ROOT%/}/_deps"
VERSION=$(awk -F'"' '/^__version__/{print $2}' "${KERNEL_DIR}/python/sgl_kernel/version.py")
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_BRANCH=$(git branch --show-current)
KERNEL_TREE=$(git rev-parse HEAD:python/sglang/kernels/aot)
EXPECTED_CACHE_BRANCH="JD-${BASE_IMAGE_TAG}"
EXPECTED_RELEASE_SOURCE_COMMIT="${SGL_KERNEL_EXPECTED_SOURCE_COMMIT:-}"
EXPECTED_RELEASE_TREE="${SGL_KERNEL_EXPECTED_TREE:-}"
mkdir -p "${WHEEL_CACHE_DIR}"
mkdir -p "${SGL_KERNEL_FETCHCONTENT_BASE_DIR}"

CLEAN_SGL_KERNEL_BUILD_TMPDIR=0
if [[ -z "${SGL_KERNEL_BUILD_TMPDIR:-}" ]]; then
    if [[ -n "${CI_TMP_DIR:-}" ]]; then
        SGL_KERNEL_BUILD_TMPDIR="${CI_TMP_DIR%/}/sgl-kernel"
    else
        SGL_KERNEL_TMP_ROOT="${CI_WORK_DIR}/ci/sglang/tmp/${BASE_IMAGE_TAG}/sgl-kernel"
        mkdir -p "${SGL_KERNEL_TMP_ROOT}"
        SGL_KERNEL_BUILD_TMPDIR=$(mktemp -d "${SGL_KERNEL_TMP_ROOT}/cuda${CUDA_TOOLKIT_VERSION}-${TARGET_ARCH_KEY}.XXXXXX")
        CLEAN_SGL_KERNEL_BUILD_TMPDIR=1
    fi
else
    mkdir -p "${SGL_KERNEL_BUILD_TMPDIR}"
fi
mkdir -p "${SGL_KERNEL_BUILD_TMPDIR}"
export TMPDIR="${SGL_KERNEL_BUILD_TMPDIR}"
export TMP="${SGL_KERNEL_BUILD_TMPDIR}"
export TEMP="${SGL_KERNEL_BUILD_TMPDIR}"
export UV_BUILD_DIR="${SGL_KERNEL_BUILD_TMPDIR}/build"
if [[ ${CLEAN_SGL_KERNEL_BUILD_TMPDIR} -eq 1 ]]; then
    trap 'rm -rf "${SGL_KERNEL_BUILD_TMPDIR}" || true' EXIT
fi

FETCHCONTENT_REVISION_LOG=$(record_fetchcontent_revisions "${KERNEL_DIR}")
echo "${FETCHCONTENT_REVISION_LOG}"
FETCHCONTENT_REVISIONS_SHA256=$(
    printf '%s\n' "${FETCHCONTENT_REVISION_LOG}" | sha256sum | awk '{print $1}'
)

echo "[SGLang CI] CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION}"
echo "[SGLang CI] SGLANG_KERNEL_TARGET_ARCHS=${TARGET_ARCHS}"
echo "[SGLang CI] SGL_KERNEL_BUILD_TMPDIR=${SGL_KERNEL_BUILD_TMPDIR}"
echo "[SGLang CI] SGL_KERNEL_FETCHCONTENT_BASE_DIR=${SGL_KERNEL_FETCHCONTENT_BASE_DIR}"
echo "[SGLang CI] UV_BUILD_DIR=${UV_BUILD_DIR}"
echo "[SGLang CI] JD_CI_SGL_KERNEL_BUILD_MAX_JOBS=${JD_CI_SGL_KERNEL_BUILD_MAX_JOBS}"
echo "[SGLang CI] JD_CI_SGL_KERNEL_NVCC_THREADS=${JD_CI_SGL_KERNEL_NVCC_THREADS}"
echo "[SGLang CI] SOURCE_COMMIT=${SOURCE_COMMIT}"
echo "[SGLang CI] SOURCE_BRANCH=${SOURCE_BRANCH:-<detached>}"
echo "[SGLang CI] KERNEL_TREE=${KERNEL_TREE}"
echo "[SGLang CI] EXPECTED_RELEASE_SOURCE_COMMIT=${EXPECTED_RELEASE_SOURCE_COMMIT:-<not-required>}"
echo "[SGLang CI] EXPECTED_RELEASE_TREE=${EXPECTED_RELEASE_TREE:-<not-required>}"
echo "[SGLang CI] FETCHCONTENT_REVISIONS_SHA256=${FETCHCONTENT_REVISIONS_SHA256}"

CMAKE_EXTRA_ARGS=""
if [[ ${REQUIRE_SM100A} -eq 1 ]]; then
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DSGL_KERNEL_ENABLE_SM100A=ON"
fi
if [[ ${REQUIRE_SM103A} -eq 1 ]]; then
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DSGL_KERNEL_ENABLE_SM103A=ON"
fi
CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DSGL_KERNEL_COMPILE_THREADS=${JD_CI_SGL_KERNEL_NVCC_THREADS}"
CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DFETCHCONTENT_BASE_DIR=${SGL_KERNEL_FETCHCONTENT_BASE_DIR} -DENABLE_BELOW_SM90=OFF"

if [[ "${EVENT_TYPE}" == "merge_request__merged" ]]; then
    if [[ ! "${EXPECTED_RELEASE_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ \
        || ! "${EXPECTED_RELEASE_TREE}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "[SGLang CI] ERROR: exact release source commit/tree are required for cache-only mode" >&2
        exit 1
    fi
    if [[ "$(git rev-parse "${EXPECTED_RELEASE_SOURCE_COMMIT}:python/sglang/kernels/aot" 2>/dev/null || true)" \
        != "${EXPECTED_RELEASE_TREE}" ]]; then
        echo "[SGLang CI] ERROR: host-provided release commit/tree cannot be verified" >&2
        exit 1
    fi
    if [[ ! -f "${CACHE_BUILD_INFO}" ]]; then
        echo "[SGLang CI] ERROR: SGL-Kernel cache metadata missing: ${CACHE_BUILD_INFO}" >&2
        exit 1
    fi
    CACHED_WHEEL_NAME=$(sed -n 's/^WHEEL=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_WHEEL_SHA256=$(sed -n 's/^WHEEL_SHA256=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_SOURCE_COMMIT=$(sed -n 's/^SOURCE_COMMIT=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_SOURCE_BRANCH=$(sed -n 's/^SOURCE_BRANCH=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_KERNEL_TREE=$(sed -n 's/^KERNEL_TREE=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_BASE_IMAGE_TAG=$(sed -n 's/^BASE_IMAGE_TAG=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_CUDA=$(sed -n 's/^CUDA_TOOLKIT_VERSION=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_ARCHS=$(sed -n 's/^TARGET_ARCHS=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED_DEPS_SHA256=$(sed -n 's/^FETCHCONTENT_REVISIONS_SHA256=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    CACHED="${WHEEL_CACHE_DIR}/${CACHED_WHEEL_NAME}"
    if [[ ! -f "${CACHED}" ]]; then
        echo "[SGLang CI] ERROR: SGL-Kernel wheel cache miss: ${WHEEL_CACHE_DIR}" >&2
        exit 1
    fi
    if [[ "${CACHED_SOURCE_BRANCH}" != "${EXPECTED_CACHE_BRANCH}" \
        || "${CACHED_SOURCE_COMMIT}" != "${EXPECTED_RELEASE_SOURCE_COMMIT}" \
        || "${CACHED_KERNEL_TREE}" != "${EXPECTED_RELEASE_TREE}" \
        || "${CACHED_BASE_IMAGE_TAG}" != "${BASE_IMAGE_TAG}" \
        || "${CACHED_CUDA}" != "${CUDA_TOOLKIT_VERSION}" \
        || "${CACHED_ARCHS}" != "${TARGET_ARCHS}" \
        || "${CACHED_DEPS_SHA256}" != "${FETCHCONTENT_REVISIONS_SHA256}" ]]; then
        echo "[SGLang CI] ERROR: SGL-Kernel cache provenance mismatch: ${CACHE_BUILD_INFO}" >&2
        exit 1
    fi
    if [[ "$(git rev-parse "${CACHED_SOURCE_COMMIT}^{commit}" 2>/dev/null || true)" \
        != "${CACHED_SOURCE_COMMIT}" \
        || "$(git rev-parse "${CACHED_SOURCE_COMMIT}:python/sglang/kernels/aot" 2>/dev/null || true)" \
        != "${CACHED_KERNEL_TREE}" ]]; then
        echo "[SGLang CI] ERROR: cached SGL-Kernel source commit/tree cannot be verified" >&2
        exit 1
    fi
    WHEEL_SHA256=$(sha256sum "${CACHED}" | awk '{print $1}')
    if [[ "${WHEEL_SHA256}" != "${CACHED_WHEEL_SHA256}" ]]; then
        echo "[SGLang CI] ERROR: cached SGL-Kernel wheel sha256 mismatch" >&2
        exit 1
    fi
    echo "[SGLang CI] SGL-Kernel cache provenance: source=${CACHED_SOURCE_COMMIT} tree=${CACHED_KERNEL_TREE} wheel_sha256=${WHEEL_SHA256}"
    echo "[SGLang CI] 合入事件 + 缓存命中，直接 pip install"
    pip install --force-reinstall --no-deps "${CACHED}"
else
    sanitize_fetchcontent_cache "${SGL_KERNEL_FETCHCONTENT_BASE_DIR}"
    echo "[SGLang CI] 编译 sgl-kernel..."
    make -C "${KERNEL_DIR}" rebuild \
        MAX_JOBS="${JD_CI_SGL_KERNEL_BUILD_MAX_JOBS}" \
        CMAKE_BUILD_PARALLEL_LEVEL="${JD_CI_SGL_KERNEL_BUILD_MAX_JOBS}" \
        CMAKE_ARGS="-DBASE_IMAGE_TAG=${BASE_IMAGE_TAG} -DCI_WORK_DIR=${CI_WORK_DIR} ${CMAKE_EXTRA_ARGS}" 2>&1 \
        | python3 "${SCRIPT_DIR}/filter_sgl_kernel_build_log.py"

    BUILT=$(ls -t "${KERNEL_DIR}"/dist/*.whl 2>/dev/null | head -1 || true)
    if [[ -f "${BUILT}" ]]; then
        rm -f "${WHEEL_CACHE_DIR}"/sglang_kernel-"${VERSION}"-*.whl
        cp -f "${BUILT}" "${WHEEL_CACHE_DIR}/"
        CACHED="${WHEEL_CACHE_DIR}/$(basename "${BUILT}")"
        WHEEL_SHA256=$(sha256sum "${CACHED}" | awk '{print $1}')
        cat > "${CACHE_BUILD_INFO}.tmp.$$" <<EOF
SOURCE_COMMIT=${SOURCE_COMMIT}
SOURCE_BRANCH=${SOURCE_BRANCH}
KERNEL_TREE=${KERNEL_TREE}
BASE_IMAGE_TAG=${BASE_IMAGE_TAG}
CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION}
TARGET_ARCHS=${TARGET_ARCHS}
FETCHCONTENT_REVISIONS_SHA256=${FETCHCONTENT_REVISIONS_SHA256}
WHEEL=$(basename "${CACHED}")
WHEEL_SHA256=${WHEEL_SHA256}
BUILT_AT=$(date -Iseconds)
EOF
        mv -f "${CACHE_BUILD_INFO}.tmp.$$" "${CACHE_BUILD_INFO}"
        echo "[SGLang CI] wheel 已写入缓存: ${CACHED}"
        echo "[SGLang CI] wheel metadata: ${CACHE_BUILD_INFO}"
    else
        echo "[SGLang CI] ERROR: 未找到编译产物 dist/*.whl" >&2
        exit 1
    fi
fi

SM90_COMMON=$(find_installed_sgl_kernel_so "sm90/common_ops*.so")
SM100_COMMON=$(find_installed_sgl_kernel_so "sm100/common_ops*.so")
if [[ ${REQUIRE_SM90} -eq 1 ]]; then
    require_cubin_arch "${SM90_COMMON}" "sm_90"
fi
if [[ ${REQUIRE_SM90A} -eq 1 ]]; then
    require_cubin_arch "${SM90_COMMON}" "sm_90a"
fi
if [[ ${REQUIRE_SM100A} -eq 1 ]]; then
    require_cubin_arch "${SM100_COMMON}" "sm_100a"
fi
if [[ ${REQUIRE_SM103A} -eq 1 ]]; then
    require_cubin_arch "${SM100_COMMON}" "sm_103a"
fi

FLASHMLA_OPS=$(find_installed_sgl_kernel_so "flashmla_ops*.so")
if [[ ${REQUIRE_SM90A} -eq 1 ]]; then
    require_cubin_arch "${FLASHMLA_OPS}" "sm_90a"
fi
if [[ ${REQUIRE_SM100A} -eq 1 ]]; then
    require_cubin_arch "${FLASHMLA_OPS}" "sm_100a"
fi
if [[ ${REQUIRE_SM103A} -eq 1 ]]; then
    require_cubin_arch "${FLASHMLA_OPS}" "sm_103a"
fi

INFLLM_OPS=$(find_installed_sgl_kernel_so "infllm_ops*.so")
require_cubin_arch "${INFLLM_OPS}" "sm_90"
if version_ge "${CUDA_TOOLKIT_VERSION}" "12.8"; then
    require_cubin_arch "${INFLLM_OPS}" "sm_120a"
fi
