#ifndef UTILITY_H_INCLUDED
#define UTILITY_H_INCLUDED

#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <driver_functions.h>


typedef float VolumeType;
#define CACHE_SIZE 200


int isVisible(float x, float y, float z, float dx, float dy, float dz, float angle);

bool isPresent(std::vector<int> cache, int index, int &hit_position);
void tpd2xyz(float theta, float phi, float dis, float &x, float &y, float &z);
void xyz2tpd(float x, float y, float z, float &theta, float &phi, float &dis);
void getCurrentVisibleBlocks(float theta, float phi, float dis,
							 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 int block_size,
							 float angle,
							 std::vector<int> &visible_blocks);

void getCurrentVisibleBlocksFastV2(float theta, float phi, float dis,
							 	 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							     int block_size,
							 	 float angle,
							 	 std::vector<int> &visible_blocks,
								 int *cornerBlockMapArray);

void getCurrentVisibleBlocksFast(float theta, float phi, float dis,
							 	 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							     int block_size,
							 	 float angle,
							 	 std::vector<int> &visible_blocks,
								 // std::map<int, std::vector<std::vector<int>>> &cornerBlockMap);
								 int *cornerBlockMapArray,
								 int *blockCheck);

void getCurrentVisibleBlocksShift(float theta, float phi, float dis,
							 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 int block_size,
							 float angle,
							 std::vector<int> &visible_blocks,
							 float shift);

void getCurrentVisibleBlocksRange(float theta, float phi, float dis, int index,
							 	  int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 	  int block_size,
							 	  float angle,
							 	  std::vector<int> &visible_blocks,
								  std::map<int, std::vector<std::vector<float>>> &range,
								  int *cornerBlockMapArray,
								  int *blockCheck);

void getCurrentVisibleBlocksRangeRealtime(float theta, float phi, float dis,
							 	  		  int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 	  		  int block_size,
							 	  		  float angle,
							 	  		  std::vector<int> &visible_blocks,
								  		  std::vector<float> thetas, std::vector<float> phis,
								  		  int *cornerBlockMapArray,
										  int *blockCheck);

/*
void getCurrentVisibleBlocksLUT(float theta, float phi, float dis,
							 	std::vector<int> &visible_blocks);
								*/
int updateCacheAndCacheData(std::vector<int> &cache,
							std::vector<int> &cache_mem,
							void *cache_data,
							int visible_block_index,
							cudaExtent microBlockSize,
							std::string method,
							const char *volumeFilePath,
							int camera_index);

void updateCache(std::vector<int> &cache, std::vector<int> visible_blocks, std::string method, int &hit, int &miss);
void updateCacheData(void *cache_data, std::vector<int> &cache_mem, std::vector<int> cache, int cache_size, cudaExtent microBlockSize, const char *volumeFilePath);
void getVisibleBlocksData(void *cache_data, std::vector<int> cache_mem, void *h_visible_blocks_data, std::vector<int> visible_blocks, cudaExtent microBlockSize);

#endif
