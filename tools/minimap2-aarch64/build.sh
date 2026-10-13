#!/usr/bin/env bash
#
# Build a static aarch64 minimap2 (with zlib) using an aarch64-linux-gnu
# cross toolchain. Reproduces tools/minimap2-aarch64/minimap2.
#
# Requirements: aarch64-linux-gnu-gcc/ar/strip (e.g. Xilinx Vitis), curl, make.
#
set -euo pipefail

TOOLCHAIN_PREFIX="${TOOLCHAIN_PREFIX:-aarch64-linux-gnu}"
MINIMAP2_VERSION="2.24"
ZLIB_VERSION="1.3.1"

WORK="${WORK:-$(mktemp -d)}"
echo "> work dir: $WORK"

# 1. zlib (static, for aarch64)
cd "$WORK"
curl -sSL "https://zlib.net/fossils/zlib-${ZLIB_VERSION}.tar.gz" -o zlib.tar.gz
tar xzf zlib.tar.gz
cd "zlib-${ZLIB_VERSION}"
CHOST="${TOOLCHAIN_PREFIX}" \
CC="${TOOLCHAIN_PREFIX}-gcc" \
AR="${TOOLCHAIN_PREFIX}-ar" \
RANLIB="${TOOLCHAIN_PREFIX}-ranlib" \
    ./configure --static --prefix="$WORK/zlib-arm"
make -j4
make install

# 2. minimap2 (ARM NEON kernels, static)
cd "$WORK"
curl -sSL "https://github.com/lh3/minimap2/archive/refs/tags/v${MINIMAP2_VERSION}.tar.gz" -o minimap2.tar.gz
tar xzf minimap2.tar.gz
cd "minimap2-${MINIMAP2_VERSION}"
make clean >/dev/null 2>&1 || true
make \
    CC="${TOOLCHAIN_PREFIX}-gcc" \
    AR="${TOOLCHAIN_PREFIX}-ar" \
    arm_neon=1 aarch64=1 \
    CPPFLAGS="-DHAVE_KALLOC -I$WORK/zlib-arm/include" \
    LIBS="-lm $WORK/zlib-arm/lib/libz.a -lpthread" \
    CFLAGS="-O2 -static -Wc++-compat -D_FILE_OFFSET_BITS=64 -fsigned-char"

"${TOOLCHAIN_PREFIX}-strip" minimap2
file minimap2
echo "> built $WORK/minimap2-${MINIMAP2_VERSION}/minimap2"
