#pragma once

#if defined(__cplusplus)
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#include <nnpack/fxdiv.h>
#include <nnpack/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void(*pthreadpool_function_1d_t)(void*, const size_t);
typedef void(*pthreadpool_function_1d_tiled_t)(void*, const size_t, const size_t);
typedef void(*pthreadpool_function_2d_t)(void*, const size_t, const size_t);
typedef void(*pthreadpool_function_2d_tiled_t)(void*, const size_t, const size_t, const size_t, const size_t);
typedef void(*pthreadpool_function_3d_tiled_t)(void*, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t);
typedef void(*pthreadpool_function_4d_tiled_t)(void*, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t);

struct compute_1d_tiled_context {
	pthreadpool_function_1d_tiled_t function;
	void* argument;
	const size_t range;
	const size_t tile;
};

struct compute_2d_context {
	pthreadpool_function_2d_t function;
	void* argument;
	const struct fxdiv_divisor_size_t range_j;
};

struct compute_2d_tiled_context {
	pthreadpool_function_2d_tiled_t function;
	void* argument;
	const struct fxdiv_divisor_size_t tile_range_j;
	const size_t range_i;
	const size_t range_j;
	const size_t tile_i;
	const size_t tile_j;
};

struct compute_3d_tiled_context {
	pthreadpool_function_3d_tiled_t function;
	void* argument;
	struct fxdiv_divisor_size_t tile_range_j;
	struct fxdiv_divisor_size_t tile_range_k;
	size_t range_i;
	size_t range_j;
	size_t range_k;
	size_t tile_i;
	size_t tile_j;
	size_t tile_k;
};

void pthreadpool_compute_1d(
	pthreadpool_function_1d_t function,
	void* argument,
	const size_t range);

void pthreadpool_compute_1d_tiled(
	pthreadpool_function_1d_tiled_t function,
	void* argument,
	const size_t range,
	const size_t tile);

void pthreadpool_compute_2d(
	pthreadpool_function_2d_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j);

void pthreadpool_compute_2d_tiled(
	pthreadpool_function_2d_tiled_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j,
	const size_t tile_i,
	const size_t tile_j);

void pthreadpool_compute_3d_tiled(
	pthreadpool_function_3d_tiled_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j,
	const size_t range_k,
	const size_t tile_i,
	const size_t tile_j,
	const size_t tile_k);

void pthreadpool_compute_4d_tiled(
	pthreadpool_function_4d_tiled_t function,
	void* argument,
	const size_t range_i,
	const size_t range_j,
	const size_t range_k,
	const size_t range_l,
	const size_t tile_i,
	const size_t tile_j,
	const size_t tile_k,
	const size_t tile_l);

#ifdef __cplusplus
}
#endif
