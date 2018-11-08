#pragma once

#if defined(__cplusplus)
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

#include <nnpack/fxdiv.h>
#include <nnpack/utils.h>

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
	const size_t tile_k)
{
	for (size_t i = 0; i < range_i; i += tile_i) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				function(argument, i, j, k,
					min(range_i - i, tile_i), min(range_j - j, tile_j), min(range_k - k, tile_k));
			}
		}
	}
}

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
	const size_t tile_l)
{
	for (size_t i = 0; i < range_i; i += tile_i) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				for (size_t l = 0; l < range_l; l += tile_l) {
					function(argument, i, j, k, l,
						min(range_i - i, tile_i), min(range_j - j, tile_j), min(range_k - k, tile_k), min(range_l - l, tile_l));
				}
			}
		}
	}
}
#ifdef __cplusplus
}
#endif