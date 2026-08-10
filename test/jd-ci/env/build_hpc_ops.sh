#!/bin/bash
set -euo pipefail

verify_source_revision() {
    local source_dir=$1
    local expected_revision=$2
    local source_revision

    if ! source_revision=$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null); then
        echo "[JD CI] ERROR: hpc-ops source is not a Git checkout: ${source_dir}" >&2
        return 1
    fi
    if [[ "${source_revision}" != "${expected_revision}" ]]; then
        echo "[JD CI] ERROR: hpc-ops revision mismatch: expected=${expected_revision}, actual=${source_revision}" >&2
        return 1
    fi
    echo "[JD CI] HPC_OPS_SOURCE_REVISION=${source_revision}"
}

if [[ "${1:-}" == "--verify-source-revision" ]]; then
    if [[ $# -ne 3 ]]; then
        echo "Usage: $0 --verify-source-revision <source-dir> <expected-revision>" >&2
        exit 2
    fi
    verify_source_revision "$2" "$3"
    exit 0
fi

SOURCE_DIR="$1"
COMPILE_DIR="$2"
INSTALL_DIR="$3"
CACHE_ROOT="$4"
FORCE_REBUILD="$5"
REQUIRE_CACHE="$6"
EXPECTED_REVISION="${7:?expected hpc-ops revision is required}"

# 按 CUDA Toolkit 版本分桶，避免不同 CUDA 的 wheel 互相覆盖
CUDA_TOOLKIT_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
if [[ -z "${CUDA_TOOLKIT_VERSION}" ]]; then
    echo "[JD CI] ERROR: failed to detect CUDA Toolkit version from nvcc" >&2
    nvcc --version || true
    exit 1
fi

CACHE_DIR="${CACHE_ROOT}/cuda${CUDA_TOOLKIT_VERSION}/wheels"
CACHE_BUILD_INFO="${CACHE_DIR}/build_info.txt"
SOURCE_REVISION=""

echo "[JD CI] CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION}"
echo "[JD CI] hpc-ops wheel cache bucket: ${CACHE_DIR}"

mkdir -p "${INSTALL_DIR}" "${CACHE_DIR}"
rm -rf "${INSTALL_DIR:?}/"*

WHEEL_FILE=""

# merge 模式：只使用已有正式缓存
if [[ "${REQUIRE_CACHE}" == "1" ]]; then
    if [[ ! -f "${CACHE_BUILD_INFO}" ]]; then
        echo "[JD CI] ERROR: hpc-ops cache has no revision metadata: ${CACHE_BUILD_INFO}" >&2
        exit 1
    fi
    SOURCE_REVISION=$(sed -n 's/^SOURCE_REVISION=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    WHEEL_NAME=$(sed -n 's/^WHEEL=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    WHEEL_SHA256=$(sed -n 's/^WHEEL_SHA256=//p' "${CACHE_BUILD_INFO}" | head -n 1)
    if [[ "${SOURCE_REVISION}" != "${EXPECTED_REVISION}" ]]; then
        echo "[JD CI] ERROR: hpc-ops cache revision mismatch: expected=${EXPECTED_REVISION}, actual=${SOURCE_REVISION:-<missing>}" >&2
        exit 1
    fi
    WHEEL_FILE="${CACHE_DIR}/${WHEEL_NAME}"
    if [[ ! -f "${WHEEL_FILE}" ]]; then
        echo "[JD CI] ERROR: hpc-ops metadata wheel missing: ${WHEEL_FILE}" >&2
        exit 1
    fi
    ACTUAL_WHEEL_SHA256=$(sha256sum "${WHEEL_FILE}" | awk '{print $1}')
    if [[ "${ACTUAL_WHEEL_SHA256}" != "${WHEEL_SHA256}" ]]; then
        echo "[JD CI] ERROR: hpc-ops cached wheel sha256 mismatch" >&2
        exit 1
    fi

    echo "[JD CI] 使用 hpc-ops wheel 缓存: ${WHEEL_FILE}"
    echo "[JD CI] HPC_OPS_SOURCE_REVISION=${SOURCE_REVISION}"
else
    if [[ ! -d "${SOURCE_DIR}" ]]; then
        echo "[JD CI] ERROR: hpc-ops 源码目录不存在: ${SOURCE_DIR}" >&2
        exit 1
    fi

    verify_source_revision "${SOURCE_DIR}" "${EXPECTED_REVISION}"
    SOURCE_REVISION=$(git -C "${SOURCE_DIR}" rev-parse HEAD)

    if [[ ! -f "${SOURCE_DIR}/Makefile" ]]; then
        echo "[JD CI] ERROR: hpc-ops 中未找到 Makefile: ${SOURCE_DIR}" >&2
        exit 1
    fi

    # 从已核验的本地 checkout 克隆 exact commit；不复制工作树中的未跟踪文件、
    # 本地修改或历史构建产物，同时保留 .git 供 setup.py 解析版本。
    rm -rf "${COMPILE_DIR}"
    echo "[JD CI] 克隆 hpc-ops exact revision 到隔离目录"
    git clone --quiet --no-hardlinks --no-checkout "${SOURCE_DIR}" "${COMPILE_DIR}"
    git -C "${COMPILE_DIR}" checkout --quiet --detach "${EXPECTED_REVISION}"
    if [[ "$(git -C "${COMPILE_DIR}" rev-parse HEAD)" != "${EXPECTED_REVISION}" ]]; then
        echo "[JD CI] ERROR: isolated hpc-ops checkout revision mismatch" >&2
        exit 1
    fi

    cd "${COMPILE_DIR}"

    echo "[JD CI] 在干净源码副本中执行 make wheel: ${COMPILE_DIR}"
    make wheel

    mapfile -t BUILT_WHEELS < <(
        find "${COMPILE_DIR}" \
            -type f \
            -path '*/dist/*.whl' \
            -print |
            sort
    )

    if (( ${#BUILT_WHEELS[@]} == 0 )); then
        echo "[JD CI] ERROR: make wheel 后未找到 dist/*.whl" >&2
        exit 1
    fi

    WHEEL_FILE="${BUILT_WHEELS[$((${#BUILT_WHEELS[@]} - 1))]}"
    echo "[JD CI] hpc-ops 构建产物: ${WHEEL_FILE}"

    if [[ "${FORCE_REBUILD}" == "1" ]]; then
        rm -f "${CACHE_DIR}"/*.whl
    fi

    cp -f "${WHEEL_FILE}" "${CACHE_DIR}/"
    WHEEL_FILE="${CACHE_DIR}/$(basename "${WHEEL_FILE}")"
    WHEEL_SHA256=$(sha256sum "${WHEEL_FILE}" | awk '{print $1}')
    cat > "${CACHE_BUILD_INFO}" <<EOF
SOURCE_REVISION=${SOURCE_REVISION}
WHEEL=$(basename "${WHEEL_FILE}")
WHEEL_SHA256=${WHEEL_SHA256}
BUILT_AT=$(date -Iseconds)
EOF

    echo "[JD CI] wheel 已保存到缓存: ${WHEEL_FILE}"
    echo "[JD CI] hpc-ops revision metadata: ${CACHE_BUILD_INFO}"
fi

# 复制到容器内部临时目录再安装
INSTALL_WHEEL="${INSTALL_DIR}/$(basename "${WHEEL_FILE}")"
cp -f "${WHEEL_FILE}" "${INSTALL_WHEEL}"

echo "[JD CI] 安装 hpc-ops wheel: ${INSTALL_WHEEL}"

python3 -m pip install \
    --force-reinstall \
    --no-deps \
    "${INSTALL_WHEEL}"

echo "[JD CI] hpc-ops 安装完成"

python3 -m pip show hpc-ops 2>/dev/null ||
    python3 -m pip show hpc_ops 2>/dev/null ||
    true
