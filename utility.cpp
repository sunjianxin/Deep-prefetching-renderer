#include "utility.hpp"
#include <iostream>
#include <chrono>
#include <math.h>
#include <assert.h>

// Load raw data from disk
void *loadRawFile(char *filename, size_t size) {
    // printf("filename %s\n", filename);
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

// #if defined(_MSC_VER_)
//     printf("Read '%s', %Iu bytes\n", filename, read);
// #else
//     printf("Read '%s', %zu bytes\n", filename, read);
// #endif

    return data;
}

int isVisible(float x, float y, float z, float dx, float dy, float dz, float angle) {
	// testing position T: x, y, z; camera position C: dx, dy, dz; volume center position O: 0, 0, 0
	// CT: (x - dx), (y - dy), (z - dz)
	// CO: (0 - dx), (0 - dy), (0 - dz)
	// Handle when testing position is on volume center
	if (x == 0 && y == 0 && z == 0) {
		// The angle is 0 degree, should alway count
		return 1;
	}
	float CT_norm = sqrt((x - dx)*(x - dx) + (y - dy)*(y - dy) + (z - dz)*(z - dz));
	float CO_norm = sqrt(dx*dx + dy*dy + dz*dz);
	float CT_dot_CO = (x - dx)*(-dx) + (y - dy)*(-dy) + (z - dz)*(-dz); 
	if (CT_norm == 0) {
		return 1;
	}
	assert(CT_norm != 0);
	assert(CO_norm != 0);

	// std::cout << "acos argument: " << CT_dot_CO/CT_norm/CO_norm << std::endl;
	float result;
	if (CT_dot_CO/CT_norm/CO_norm > 1) {
		result = 0.0;
		// result = acos(1.0) * 180.0 / M_PI; // 0.0 degree
	} else if (CT_dot_CO/CT_norm/CO_norm < -1) {
		result = 180.0;
		// result = acos(-1.0) * 180.0 / M_PI; // 180.0 degree
	} else {
		result = acos(CT_dot_CO/CT_norm/CO_norm) * 180.0 / M_PI;
	}
	if (isnan(result)) {
		std::cout << "get nan-------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        exit(EXIT_SUCCESS);
	}
	if (result <= angle) {
		return 1;
	}
	else {
		return 0;
	}
}

bool isPresent(std::vector<int> cache, int index, int &hit_position) {

	for (int i = 0; i < cache.size(); i++) {
		if (cache.at(i) == index) {
			hit_position = i;
			return true;
		}
	}
	return false;
}

void tpd2xyz(float theta, float phi, float dis, float &x, float &y, float &z) {
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	x = dis * sin_theta * cos_phi;
	y = dis * sin_theta * sin_phi;
	z = dis * cos_theta;
}

void xyz2tpd(float x, float y, float z, float &theta, float &phi, float &dis) {
    dis = sqrt(x*x + y*y + z*z);
    theta = atan(sqrt(x*x + y*y)/z)/M_PI*180.0;
    phi = atan(y/x)/M_PI*180.0;
        
    if (dis >= 2.0f){
        dis= 2.0f;
    }else if (dis <= 1.0f){
        dis= 1.0f;
    }


    if (theta < 0) {
        theta = 180 + theta;
    }
    if (phi > 0) {
        if (x > 0) {
            phi = phi;
        } else {
            phi = 180 + phi;
        }
    } else {
        if (x < 0) {
            phi = 180 + phi;
        } else {
            phi = 360 + phi;
        }
    }
}

float getDistance(float x_center, float y_center, float z_center, float camera_x, float camera_y, float camera_z) {
	float x_distance = camera_x - x_center;
	float y_distance = camera_y - y_center;
	float z_distance = camera_z - z_center;
	return sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
}

float getDistanceFromBlock(int block_index,
						   float camera_x, float camera_y, float camera_z,
						   int block_num,
						   float unit) {
	int z = block_index/(block_num*block_num);
	int remain = block_index%(block_num*block_num);
	int y = remain/block_num;
	int x = remain%block_num;

	float x_lower = -1 + x*unit;
	float y_lower = -1 + y*unit;
	float z_lower = -1 + z*unit;
	float x_upper = x_lower + unit;
	float y_upper = y_lower + unit;
	float z_upper = z_lower + unit;
				
	return getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
}




void getCurrentVisibleBlocksFastV2(float theta, float phi, float dis,
							 	 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							     int block_size,
							 	 float angle,
							 	 std::vector<int> &visible_blocks,
								 int *cornerBlockMapArray) {
	// translate from Spherical Coordinate System to Cartesian Coordinate System for the camera location
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	float camera_x = dis * sin_theta * cos_phi;
	float camera_y = dis * sin_theta * sin_phi;
	float camera_z = dis * cos_theta;

	// int block_num_1 = (volume_size_1 - 1)/(block_size - 1); // 2
	// int block_num_2 = (volume_size_2 - 1)/(block_size - 1); // 4
	// int block_num_3 = (volume_size_3 - 1)/(block_size - 1); // 8
	// int block_num_4 = (volume_size_4 - 1)/(block_size - 1); // 16
	int block_num_1 = (volume_size_1 - 1)/(block_size - 2); // 2
	int block_num_2 = (volume_size_2 - 1)/(block_size - 2); // 4
	int block_num_3 = (volume_size_3 - 1)/(block_size - 2); // 8
	int block_num_4 = (volume_size_4 - 1)/(block_size - 2); // 16

	int all_block_num_1 = block_num_1*block_num_1*block_num_1; // 8
	int all_block_num_2 = block_num_2*block_num_2*block_num_2; // 64
	int all_block_num_3 = block_num_3*block_num_3*block_num_3; // 512
	int all_block_num_4 = block_num_4*block_num_4*block_num_4; // 4096

	float unit_1 = 2.0/(float)block_num_1;
	float unit_2 = 2.0/(float)block_num_2;
	float unit_3 = 2.0/(float)block_num_3;
	float unit_4 = 2.0/(float)block_num_4;

	float x_start = -1.0;
	float y_start = -1.0;
	float z_start = -1.0;


	int sample_num = block_num_4 + 1;  // 17
	std::vector<int> level_1_blocks;
	std::vector<int> level_2_blocks;
	std::vector<int> level_3_blocks;
	std::vector<int> level_4_blocks;

	std::vector<int> selected_blocks_level_1;
	std::vector<int> selected_blocks_level_2;
	std::vector<int> selected_blocks_level_3;
	std::vector<int> selected_blocks_level_4;

	std::chrono::high_resolution_clock::time_point t_start, t_end;
	t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 17*17*17; i++) {
		int z = i/(sample_num*sample_num); // 1/(17*17) integer portion
		int remain = i%(sample_num*sample_num); // 1%(17*17) remainding portion
		int y = remain/(sample_num); 
		int x = remain%(sample_num);

		float x_corner = x_start + x*unit_4;
		float y_corner = y_start + y*unit_4;
		float z_corner = z_start + z*unit_4;

		int b = isVisible(x_corner, y_corner, z_corner, camera_x, camera_y, camera_z, angle/2.0);
	
		/* find all visible blocks for all levels */
#if 1
		int hit_position;
		if (b == 1) {
			/* Level 1 */
			int begin = i*4*9;
			int begin_level_1 = begin;
			int begin_level_2 = begin + 9;
			int begin_level_3 = begin + 9 + 9;
			int begin_level_4 = begin + 9 + 9 + 9;
			for (int j = 0; j < cornerBlockMapArray[begin_level_1]; j++) {
				float distance = getDistanceFromBlock(cornerBlockMapArray[begin_level_1 + j + 1], camera_x, camera_y, camera_z, block_num_1, unit_1);
				if (distance >= 2.4) {
					level_1_blocks.push_back(cornerBlockMapArray[begin_level_1 + j + 1]);
				}
			}
			/* Level 2 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_2]; j++) {
				float distance = getDistanceFromBlock(cornerBlockMapArray[begin_level_2 + j + 1] - 8, camera_x, camera_y, camera_z, block_num_2, unit_2);
				if (distance >= 1.6) {
					level_2_blocks.push_back(cornerBlockMapArray[begin_level_2 + j + 1]);
				}
			}
			/* Level 3 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_3]; j++) {
				float distance = getDistanceFromBlock(cornerBlockMapArray[begin_level_3 + j + 1] - 8 - 64, camera_x, camera_y, camera_z, block_num_3, unit_3);
				if (distance >= 0.8) {
					level_3_blocks.push_back(cornerBlockMapArray[begin_level_3 + j + 1]);
				}
			}
			/* Level 4 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_4]; j++) {
				float distance = getDistanceFromBlock(cornerBlockMapArray[begin_level_4 + j + 1] - 8 - 64 - 512, camera_x, camera_y, camera_z, block_num_4, unit_4);
				if (distance >= 0.0) {
					level_4_blocks.push_back(cornerBlockMapArray[begin_level_4 + j + 1]);
				}
			}
		}
#endif
	}
	t_end = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;
	// std::cout << "'''''''Finding corner visibility time: " << time << std::endl;


	for (int i = 0; i < level_1_blocks.size(); i++) {
		int block_index = level_1_blocks.at(i) - 0;
		selected_blocks_level_1.push_back(block_index);
		visible_blocks.push_back(block_index);
	}
	for (int i = 0; i < level_2_blocks.size(); i++) {
		int block_index = level_2_blocks.at(i) - 8;
		int z = block_index/(block_num_2*block_num_2);
		int remain = block_index%(block_num_2*block_num_2);
		int y = remain/block_num_2;
		int x = remain%block_num_2;

		// skip current level 2 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/2; // 0~2
		int i_y_up_level_1 = y/2;
		int i_z_up_level_1 = z/2;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		if (!found_level_1) {
			selected_blocks_level_2.push_back(block_index);
			visible_blocks.push_back(block_index + 8);
		}
	}
	for (int i = 0; i < level_3_blocks.size(); i++) {
		int block_index = level_3_blocks.at(i) - 8 - 64;
		int z = block_index/(block_num_3*block_num_3);
		int remain = block_index%(block_num_3*block_num_3);
		int y = remain/block_num_3;
		int x = remain%block_num_3;

		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/4; // 0~2
		int i_y_up_level_1 = y/4;
		int i_z_up_level_1 = z/4;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/2; // 0~4
		int i_y_up_level_2 = y/2;
		int i_z_up_level_2 = z/2;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {			
				found_level_2 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2) {
			selected_blocks_level_3.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64);
		}
	}
	for (int i = 0; i < level_4_blocks.size(); i++) {
		int block_index = level_4_blocks.at(i) - 8 - 64 - 512;
		int z = block_index/(block_num_4*block_num_4);
		int remain = block_index%(block_num_4*block_num_4);
		int y = remain/block_num_4;
		int x = remain%block_num_4;

		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/8; // 0~2
		int i_y_up_level_1 = y/8;
		int i_z_up_level_1 = z/8;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/4; // 0~4
		int i_y_up_level_2 = y/4;
		int i_z_up_level_2 = z/4;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {
				found_level_2 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 3 blocks
		int i_x_up_level_3 = x/2; // 0~8
		int i_y_up_level_3 = y/2;
		int i_z_up_level_3 = z/2;
		int index_up_level_3 = i_z_up_level_3*8*8 + i_y_up_level_3*8 + i_x_up_level_3;
		bool found_level_3 = false;
		for (int k = 0; k < selected_blocks_level_3.size(); k++) {
			if (index_up_level_3 == selected_blocks_level_3.at(k)) {
				found_level_3 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2 && !found_level_3) {
			selected_blocks_level_4.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64 + 512);
		}
	}
}

void getCurrentVisibleBlocksFast(float theta, float phi, float dis,
							 	 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							     int block_size,
							 	 float angle,
							 	 std::vector<int> &visible_blocks,
								 // std::map<int, std::vector<std::vector<int>>> &cornerBlockMap) {
								 int *cornerBlockMapArray,
								 int *blockCheck) {
	// translate from Spherical Coordinate System to Cartesian Coordinate System for the camera location
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	float camera_x = dis * sin_theta * cos_phi;
	float camera_y = dis * sin_theta * sin_phi;
	float camera_z = dis * cos_theta;

	// int block_num_1 = (volume_size_1 - 1)/(block_size - 1); // 2
	// int block_num_2 = (volume_size_2 - 1)/(block_size - 1); // 4
	// int block_num_3 = (volume_size_3 - 1)/(block_size - 1); // 8
	// int block_num_4 = (volume_size_4 - 1)/(block_size - 1); // 16
	int block_num_1 = (volume_size_1 - 1)/(block_size - 2); // 2
	int block_num_2 = (volume_size_2 - 1)/(block_size - 2); // 4
	int block_num_3 = (volume_size_3 - 1)/(block_size - 2); // 8
	int block_num_4 = (volume_size_4 - 1)/(block_size - 2); // 16

	int all_block_num_1 = block_num_1*block_num_1*block_num_1; // 8
	int all_block_num_2 = block_num_2*block_num_2*block_num_2; // 64
	int all_block_num_3 = block_num_3*block_num_3*block_num_3; // 512
	int all_block_num_4 = block_num_4*block_num_4*block_num_4; // 4096

	float unit_1 = 2.0/(float)block_num_1;
	float unit_2 = 2.0/(float)block_num_2;
	float unit_3 = 2.0/(float)block_num_3;
	float unit_4 = 2.0/(float)block_num_4;

	float x_start = -1.0;
	float y_start = -1.0;
	float z_start = -1.0;


	int sample_num = block_num_4 + 1;  // 17
	std::vector<int> level_1_blocks;
	std::vector<int> level_2_blocks;
	std::vector<int> level_3_blocks;
	std::vector<int> level_4_blocks;

	std::vector<int> selected_blocks_level_1;
	std::vector<int> selected_blocks_level_2;
	std::vector<int> selected_blocks_level_3;
	std::vector<int> selected_blocks_level_4;

	std::chrono::high_resolution_clock::time_point t_start, t_end;
	t_start = std::chrono::high_resolution_clock::now();

	memset(blockCheck, 0, sizeof(int)*(8 + 64 + 512 + 4096));

	// for (int i = 0; i < cornerBlockMap.size(); i++) {
	for (int i = 0; i < 17*17*17; i++) {
		int z = i/(sample_num*sample_num); // 1/(17*17) integer portion
		int remain = i%(sample_num*sample_num); // 1%(17*17) remainding portion
		int y = remain/(sample_num); 
		int x = remain%(sample_num);

		float x_corner = x_start + x*unit_4;
		float y_corner = y_start + y*unit_4;
		float z_corner = z_start + z*unit_4;

		int b = isVisible(x_corner, y_corner, z_corner, camera_x, camera_y, camera_z, angle/2.0);
	
		/* find all visible blocks for all levels */
		int hit_position;
#if 0
		if (b == 1) {
			/* Level 1 */
			for (int j = 0; j < cornerBlockMap[i].at(0).size(); j++) {
				if (!isPresent(level_1_blocks, cornerBlockMap[i].at(0).at(j), hit_position)) {
					level_1_blocks.push_back(cornerBlockMap[i].at(0).at(j));
				}
			}
			/* Level 2 */
			for (int j = 0; j < cornerBlockMap[i].at(1).size(); j++) {
				if (!isPresent(level_2_blocks, cornerBlockMap[i].at(1).at(j), hit_position)) {
					level_2_blocks.push_back(cornerBlockMap[i].at(1).at(j));
				}
			}
			/* Level 3 */
			for (int j = 0; j < cornerBlockMap[i].at(2).size(); j++) {
				if (!isPresent(level_3_blocks, cornerBlockMap[i].at(2).at(j), hit_position)) {
					level_3_blocks.push_back(cornerBlockMap[i].at(2).at(j));
				}
			}
			/* Level 4 */
			for (int j = 0; j < cornerBlockMap[i].at(3).size(); j++) {
				if (!isPresent(level_4_blocks, cornerblockmap[i].at(3).at(j), hit_position)) {
					level_4_blocks.push_back(cornerblockmap[i].at(3).at(j));
				}
			}
		}
#endif
		if (b == 1) {
			/* Level 1 */
			int begin = i*4*9;
			int begin_level_1 = begin;
			int begin_level_2 = begin + 9;
			int begin_level_3 = begin + 9 + 9;
			int begin_level_4 = begin + 9 + 9 + 9;
			for (int j = 0; j < cornerBlockMapArray[begin_level_1]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_1 + j + 1]] == 0)
				//if (!isPresent(level_1_blocks, cornerBlockMapArray[begin_level_1 + j + 1], hit_position)) {
					level_1_blocks.push_back(cornerBlockMapArray[begin_level_1 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_1 + j + 1]] = 1;
				//}
			}
			/* Level 2 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_2]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_2 + j + 1]] == 0)
				//if (!isPresent(level_2_blocks, cornerBlockMapArray[begin_level_2 + j + 1], hit_position)) {
					level_2_blocks.push_back(cornerBlockMapArray[begin_level_2 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_2 + j + 1]] = 1;
				//}
			}
			/* Level 3 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_3]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_3 + j + 1]] == 0)
				//if (!isPresent(level_3_blocks, cornerBlockMapArray[begin_level_3 + j + 1], hit_position)) {
					level_3_blocks.push_back(cornerBlockMapArray[begin_level_3 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_3 + j + 1]] = 1;
				//}
			}
			/* Level 4 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_4]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_4 + j + 1]] == 0)
				//if (!isPresent(level_4_blocks, cornerBlockMapArray[begin_level_4 + j + 1], hit_position)) {
					level_4_blocks.push_back(cornerBlockMapArray[begin_level_4 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_4 + j + 1]] = 1;
				//}
			}
		}

	}

	t_end = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;
	// std::cout << "'''''''Finding corner visibility time: " << time << std::endl;


	for (int i = 0; i < level_1_blocks.size(); i++) {
		int block_index = level_1_blocks.at(i) - 0;
		int z = block_index/(block_num_1*block_num_1);
		int remain = block_index%(block_num_1*block_num_1);
		int y = remain/block_num_1;
		int x = remain%block_num_1;

		float x_lower = x_start + x*unit_1;
		float y_lower = y_start + y*unit_1;
		float z_lower = z_start + z*unit_1;
		float x_upper = x_lower + unit_1;
		float y_upper = y_lower + unit_1;
		float z_upper = z_lower + unit_1;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		if (distance >= 2.4) {
			selected_blocks_level_1.push_back(block_index);
			visible_blocks.push_back(block_index);
		}
	}
	for (int i = 0; i < level_2_blocks.size(); i++) {
		int block_index = level_2_blocks.at(i) - 8;
		int z = block_index/(block_num_2*block_num_2);
		int remain = block_index%(block_num_2*block_num_2);
		int y = remain/block_num_2;
		int x = remain%block_num_2;

		float x_lower = x_start + x*unit_2;
		float y_lower = y_start + y*unit_2;
		float z_lower = z_start + z*unit_2;
		float x_upper = x_lower + unit_2;
		float y_upper = y_lower + unit_2;
		float z_upper = z_lower + unit_2;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
	
		// skip current level 2 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/2; // 0~2
		int i_y_up_level_1 = y/2;
		int i_z_up_level_1 = z/2;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		if (!found_level_1 && distance >= 1.6) {
			selected_blocks_level_2.push_back(block_index);
			visible_blocks.push_back(block_index + 8);
		}
	}
	for (int i = 0; i < level_3_blocks.size(); i++) {
		int block_index = level_3_blocks.at(i) - 8 - 64;
		int z = block_index/(block_num_3*block_num_3);
		int remain = block_index%(block_num_3*block_num_3);
		int y = remain/block_num_3;
		int x = remain%block_num_3;

		float x_lower = x_start + x*unit_3;
		float y_lower = y_start + y*unit_3;
		float z_lower = z_start + z*unit_3;
		float x_upper = x_lower + unit_3;
		float y_upper = y_lower + unit_3;
		float z_upper = z_lower + unit_3;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/4; // 0~2
		int i_y_up_level_1 = y/4;
		int i_z_up_level_1 = z/4;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/2; // 0~4
		int i_y_up_level_2 = y/2;
		int i_z_up_level_2 = z/2;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {			
				found_level_2 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2 && distance >= 0.8) {
			selected_blocks_level_3.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64);
		}
	}
	for (int i = 0; i < level_4_blocks.size(); i++) {
		int block_index = level_4_blocks.at(i) - 8 - 64 - 512;
		int z = block_index/(block_num_4*block_num_4);
		int remain = block_index%(block_num_4*block_num_4);
		int y = remain/block_num_4;
		int x = remain%block_num_4;

		float x_lower = x_start + x*unit_4;
		float y_lower = y_start + y*unit_4;
		float z_lower = z_start + z*unit_4;
		float x_upper = x_lower + unit_4;
		float y_upper = y_lower + unit_4;
		float z_upper = z_lower + unit_4;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/8; // 0~2
		int i_y_up_level_1 = y/8;
		int i_z_up_level_1 = z/8;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/4; // 0~4
		int i_y_up_level_2 = y/4;
		int i_z_up_level_2 = z/4;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {
				found_level_2 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 3 blocks
		int i_x_up_level_3 = x/2; // 0~8
		int i_y_up_level_3 = y/2;
		int i_z_up_level_3 = z/2;
		int index_up_level_3 = i_z_up_level_3*8*8 + i_y_up_level_3*8 + i_x_up_level_3;
		bool found_level_3 = false;
		for (int k = 0; k < selected_blocks_level_3.size(); k++) {
			if (index_up_level_3 == selected_blocks_level_3.at(k)) {
				found_level_3 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2 && !found_level_3) {
			selected_blocks_level_4.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64 + 512);
		}
	}
}



void getCurrentVisibleBlocks(float theta, float phi, float dis,
							 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 int block_size,
							 float angle,
							 std::vector<int> &visible_blocks) {
	
    // translate from Spherical Coordinate System to Cartesian Coordinate System for the camera location
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	float camera_x = dis * sin_theta * cos_phi;
	float camera_y = dis * sin_theta * sin_phi;
	float camera_z = dis * cos_theta;

	// int block_num_1 = (volume_size_1 - 1)/(block_size - 1); // 2
	// int block_num_2 = (volume_size_2 - 1)/(block_size - 1); // 4
	// int block_num_3 = (volume_size_3 - 1)/(block_size - 1); // 8
	// int block_num_4 = (volume_size_4 - 1)/(block_size - 1); // 16
	int block_num_1 = (volume_size_1 - 1)/(block_size - 2); // 2
	int block_num_2 = (volume_size_2 - 1)/(block_size - 2); // 4
	int block_num_3 = (volume_size_3 - 1)/(block_size - 2); // 8
	int block_num_4 = (volume_size_4 - 1)/(block_size - 2); // 16

	int all_block_num_1 = block_num_1*block_num_1*block_num_1;
	int all_block_num_2 = block_num_2*block_num_2*block_num_2;
	int all_block_num_3 = block_num_3*block_num_3*block_num_3;
	int all_block_num_4 = block_num_4*block_num_4*block_num_4;

	float unit_1 = 2.0/(float)block_num_1;
	float unit_2 = 2.0/(float)block_num_2;
	float unit_3 = 2.0/(float)block_num_3;
	float unit_4 = 2.0/(float)block_num_4;

	float x_start = -1.0;
	float y_start = -1.0;
	float z_start = -1.0;
	
	// std::cout << block_num_1 << ", " << block_num_2 << ", " << block_num_3 << ", " << block_num_4 << std::endl;

	std::vector<int> selected_blocks_level_1;
	std::vector<int> selected_blocks_level_2;
	std::vector<int> selected_blocks_level_3;
	std::vector<int> selected_blocks_level_4;

	// Check all level 1 blocks
	for (int i = 0; i < all_block_num_1; i++) {
		int z = i/(block_num_1*block_num_1);
		int remain = i%(block_num_1*block_num_1);
		int y = remain/block_num_1;
		int x = remain%block_num_1;
		
		float x_lower = x_start + x*unit_1;
		float y_lower = y_start + y*unit_1;
		float z_lower = z_start + z*unit_1;
		float x_upper = x_lower + unit_1;
		float y_upper = y_lower + unit_1;
		float z_upper = z_lower + unit_1;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			if (distance >= 2.4) {
				selected_blocks_level_1.push_back(i);
				visible_blocks.push_back(i); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 2 blocks
	for (int i = 0; i < all_block_num_2; i++) {
		int z = i/(block_num_2*block_num_2);
		int remain = i%(block_num_2*block_num_2);
		int y = remain/block_num_2;
		int x = remain%block_num_2;
		
		float x_lower = x_start + x*unit_2;
		float y_lower = y_start + y*unit_2;
		float z_lower = z_start + z*unit_2;
		float x_upper = x_lower + unit_2;
		float y_upper = y_lower + unit_2;
		float z_upper = z_lower + unit_2;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 2 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/2; // 0~2
			int i_y_up_level_1 = y/2;
			int i_z_up_level_1 = z/2;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			if (!found_level_1 && distance >= 1.6) {
				selected_blocks_level_2.push_back(i);
				visible_blocks.push_back(i + 8); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 3 blocks
	for (int i = 0; i < all_block_num_3; i++) {
		int z = i/(block_num_3*block_num_3);
		int remain = i%(block_num_3*block_num_3);
		int y = remain/block_num_3;
		int x = remain%block_num_3;
		
		float x_lower = x_start + x*unit_3;
		float y_lower = y_start + y*unit_3;
		float z_lower = z_start + z*unit_3;
		float x_upper = x_lower + unit_3;
		float y_upper = y_lower + unit_3;
		float z_upper = z_lower + unit_3;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 3 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/4; // 0~2
			int i_y_up_level_1 = y/4;
			int i_z_up_level_1 = z/4;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 2 blocks
			int i_x_up_level_2 = x/2; // 0~4
			int i_y_up_level_2 = y/2;
			int i_z_up_level_2 = z/2;
			int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
			bool found_level_2 = false;
			for (int k = 0; k < selected_blocks_level_2.size(); k++) {
				if (index_up_level_2 == selected_blocks_level_2.at(k)) {			
					found_level_2 = true;
					break;
				}
			}
			if (!found_level_1 && !found_level_2 && distance >= 0.8) {
				selected_blocks_level_3.push_back(i);
				visible_blocks.push_back(i + 8 + 64); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 4 blocks
	for (int i = 0; i < all_block_num_4; i++) {
		int z = i/(block_num_4*block_num_4);
		int remain = i%(block_num_4*block_num_4);
		int y = remain/block_num_4;
		int x = remain%block_num_4;
		
		float x_lower = x_start + x*unit_4;
		float y_lower = y_start + y*unit_4;
		float z_lower = z_start + z*unit_4;
		float x_upper = x_lower + unit_4;
		float y_upper = y_lower + unit_4;
		float z_upper = z_lower + unit_4;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 3 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/8; // 0~2
			int i_y_up_level_1 = y/8;
			int i_z_up_level_1 = z/8;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 2 blocks
			int i_x_up_level_2 = x/4; // 0~4
			int i_y_up_level_2 = y/4;
			int i_z_up_level_2 = z/4;
			int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
			bool found_level_2 = false;
			for (int k = 0; k < selected_blocks_level_2.size(); k++) {
				if (index_up_level_2 == selected_blocks_level_2.at(k)) {
					found_level_2 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 3 blocks
			int i_x_up_level_3 = x/2; // 0~8
			int i_y_up_level_3 = y/2;
			int i_z_up_level_3 = z/2;
			int index_up_level_3 = i_z_up_level_3*8*8 + i_y_up_level_3*8 + i_x_up_level_3;
			bool found_level_3 = false;
			for (int k = 0; k < selected_blocks_level_3.size(); k++) {
				if (index_up_level_3 == selected_blocks_level_3.at(k)) {
					found_level_3 = true;
					break;
				}
			}
			if (!found_level_1 && !found_level_2 && !found_level_3) {
				selected_blocks_level_4.push_back(i);
				visible_blocks.push_back(i + 8 + 64 + 512); // push to the list include all 4 levels
			}
		}
	}
}

void getCurrentVisibleBlocksShift(float theta, float phi, float dis,
							 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 int block_size,
							 float angle,
							 std::vector<int> &visible_blocks, 
							 float shift) {
	
    // translate from Spherical Coordinate System to Cartesian Coordinate System for the camera location
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	float camera_x = dis * sin_theta * cos_phi;
	float camera_y = dis * sin_theta * sin_phi;
	float camera_z = dis * cos_theta;

	int block_num_1 = (volume_size_1 - 1)/(block_size - 1); // 2
	int block_num_2 = (volume_size_2 - 1)/(block_size - 1); // 4
	int block_num_3 = (volume_size_3 - 1)/(block_size - 1); // 8
	int block_num_4 = (volume_size_4 - 1)/(block_size - 1); // 16

	int all_block_num_1 = block_num_1*block_num_1*block_num_1;
	int all_block_num_2 = block_num_2*block_num_2*block_num_2;
	int all_block_num_3 = block_num_3*block_num_3*block_num_3;
	int all_block_num_4 = block_num_4*block_num_4*block_num_4;

	float unit_1 = 2.0/(float)block_num_1;
	float unit_2 = 2.0/(float)block_num_2;
	float unit_3 = 2.0/(float)block_num_3;
	float unit_4 = 2.0/(float)block_num_4;

	float x_start = -1.0;
	float y_start = -1.0;
	float z_start = -1.0;
	
	// std::cout << block_num_1 << ", " << block_num_2 << ", " << block_num_3 << ", " << block_num_4 << std::endl;

	std::vector<int> selected_blocks_level_1;
	std::vector<int> selected_blocks_level_2;
	std::vector<int> selected_blocks_level_3;
	std::vector<int> selected_blocks_level_4;

	// Check all level 1 blocks
	for (int i = 0; i < all_block_num_1; i++) {
		int z = i/(block_num_1*block_num_1);
		int remain = i%(block_num_1*block_num_1);
		int y = remain/block_num_1;
		int x = remain%block_num_1;
		
		float x_lower = x_start + x*unit_1;
		float y_lower = y_start + y*unit_1;
		float z_lower = z_start + z*unit_1;
		float x_upper = x_lower + unit_1;
		float y_upper = y_lower + unit_1;
		float z_upper = z_lower + unit_1;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			if (distance >= 2.4 + shift) {
				selected_blocks_level_1.push_back(i);
				visible_blocks.push_back(i); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 2 blocks
	for (int i = 0; i < all_block_num_2; i++) {
		int z = i/(block_num_2*block_num_2);
		int remain = i%(block_num_2*block_num_2);
		int y = remain/block_num_2;
		int x = remain%block_num_2;
		
		float x_lower = x_start + x*unit_2;
		float y_lower = y_start + y*unit_2;
		float z_lower = z_start + z*unit_2;
		float x_upper = x_lower + unit_2;
		float y_upper = y_lower + unit_2;
		float z_upper = z_lower + unit_2;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 2 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/2; // 0~2
			int i_y_up_level_1 = y/2;
			int i_z_up_level_1 = z/2;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			if (!found_level_1 && distance >= 1.6 + shift) {
				selected_blocks_level_2.push_back(i);
				visible_blocks.push_back(i + 8); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 3 blocks
	for (int i = 0; i < all_block_num_3; i++) {
		int z = i/(block_num_3*block_num_3);
		int remain = i%(block_num_3*block_num_3);
		int y = remain/block_num_3;
		int x = remain%block_num_3;
		
		float x_lower = x_start + x*unit_3;
		float y_lower = y_start + y*unit_3;
		float z_lower = z_start + z*unit_3;
		float x_upper = x_lower + unit_3;
		float y_upper = y_lower + unit_3;
		float z_upper = z_lower + unit_3;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 3 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/4; // 0~2
			int i_y_up_level_1 = y/4;
			int i_z_up_level_1 = z/4;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 2 blocks
			int i_x_up_level_2 = x/2; // 0~4
			int i_y_up_level_2 = y/2;
			int i_z_up_level_2 = z/2;
			int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
			bool found_level_2 = false;
			for (int k = 0; k < selected_blocks_level_2.size(); k++) {
				if (index_up_level_2 == selected_blocks_level_2.at(k)) {			
					found_level_2 = true;
					break;
				}
			}
			if (!found_level_1 && !found_level_2 && distance >= 0.8 + shift) {
				selected_blocks_level_3.push_back(i);
				visible_blocks.push_back(i + 8 + 64); // push to the list include all 4 levels
			}
		}
	}
	// Check all level 4 blocks
	for (int i = 0; i < all_block_num_4; i++) {
		int z = i/(block_num_4*block_num_4);
		int remain = i%(block_num_4*block_num_4);
		int y = remain/block_num_4;
		int x = remain%block_num_4;
		
		float x_lower = x_start + x*unit_4;
		float y_lower = y_start + y*unit_4;
		float z_lower = z_start + z*unit_4;
		float x_upper = x_lower + unit_4;
		float y_upper = y_lower + unit_4;
		float z_upper = z_lower + unit_4;
		
		// checking vertices of the cube
		int b1 = isVisible(x_lower, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b2 = isVisible(x_lower, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b3 = isVisible(x_lower, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b4 = isVisible(x_lower, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b5 = isVisible(x_upper, y_lower, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b6 = isVisible(x_upper, y_lower, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		int b7 = isVisible(x_upper, y_upper, z_lower, camera_x, camera_y, camera_z, angle/2.0);
		int b8 = isVisible(x_upper, y_upper, z_upper, camera_x, camera_y, camera_z, angle/2.0);
		// checking mid point of each cube edge
		int b1b2 = isVisible(x_lower,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b1b3 = isVisible(x_lower,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b1b5 = isVisible((x_lower + x_upper)/2, y_lower,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b2 = isVisible(x_lower,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b4b3 = isVisible(x_lower,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b4b8 = isVisible((x_lower + x_upper)/2, y_upper,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b3 = isVisible((x_lower + x_upper)/2, y_upper,               z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b5 = isVisible(x_upper,               (y_lower + y_upper)/2, z_lower,               camera_x, camera_y, camera_z, angle/2.0);
		int b7b8 = isVisible(x_upper,               y_upper,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b2 = isVisible((x_lower + x_upper)/2, y_lower,               z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		int b6b5 = isVisible(x_upper,               y_lower,               (z_lower + z_upper)/2, camera_x, camera_y, camera_z, angle/2.0);
		int b6b8 = isVisible(x_upper,               (y_lower + y_upper)/2, z_upper,               camera_x, camera_y, camera_z, angle/2.0);
		if(b1||b2||b3||b4||b5||b6||b7||b8|b1b2|b1b3|b1b5|b4b2|b4b3|b4b8|b7b3|b7b5|b7b8|b6b2|b6b5|b6b8){
			// find blocks selected according to distance from this block to camera
			float x_center = (x_lower + x_upper)/2;
			float y_center = (y_lower + y_upper)/2;
			float z_center = (z_lower + z_upper)/2;
			float x_distance = camera_x - x_center;
			float y_distance = camera_y - y_center;
			float z_distance = camera_z - z_center;
			float distance = sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
			// skip current level 3 block if it is in the selected level 1 blocks
			int i_x_up_level_1 = x/8; // 0~2
			int i_y_up_level_1 = y/8;
			int i_z_up_level_1 = z/8;
			int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
			bool found_level_1 = false;
			for (int k = 0; k < selected_blocks_level_1.size(); k++) {
				if (index_up_level_1 == selected_blocks_level_1.at(k)) {
					found_level_1 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 2 blocks
			int i_x_up_level_2 = x/4; // 0~4
			int i_y_up_level_2 = y/4;
			int i_z_up_level_2 = z/4;
			int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
			bool found_level_2 = false;
			for (int k = 0; k < selected_blocks_level_2.size(); k++) {
				if (index_up_level_2 == selected_blocks_level_2.at(k)) {
					found_level_2 = true;
					break;
				}
			}
			// skip current level 3 block if it is in the selected level 3 blocks
			int i_x_up_level_3 = x/2; // 0~8
			int i_y_up_level_3 = y/2;
			int i_z_up_level_3 = z/2;
			int index_up_level_3 = i_z_up_level_3*8*8 + i_y_up_level_3*8 + i_x_up_level_3;
			bool found_level_3 = false;
			for (int k = 0; k < selected_blocks_level_3.size(); k++) {
				if (index_up_level_3 == selected_blocks_level_3.at(k)) {
					found_level_3 = true;
					break;
				}
			}
			if (!found_level_1 && !found_level_2 && !found_level_3) {
				selected_blocks_level_4.push_back(i);
				visible_blocks.push_back(i + 8 + 64 + 512); // push to the list include all 4 levels
			}
		}
	}
}

void combineVisibleBlocks(std::vector<int> &toVector, std::vector<int> &fromVector) {
	int hit_position;
	for (int i = 0; i < fromVector.size(); i++) {
		if (!isPresent(toVector, fromVector.at(i), hit_position)) {
			toVector.push_back(fromVector.at(i));
		}
	}
}



void getCurrentVisibleBlocksRange(float theta, float phi, float dis, int index,
							 	  int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 	  int block_size,
							 	  float angle,
							 	  std::vector<int> &visible_blocks,
								  std::map<int, std::vector<std::vector<float>>> &range,
								  int *cornerBlockMapArray,
								  int *blockCheck) {
	/*
	getCurrentVisibleBlocks(theta, phi, dis,
							volume_size_1, volume_size_2, volume_size_3, volume_size_4,
							block_size,
							angle,
							visible_blocks);
	*/
	// /*
	getCurrentVisibleBlocksFast(theta, phi, dis,
								volume_size_1, volume_size_2, volume_size_3, volume_size_4,
								block_size,
								angle,
								visible_blocks,
								cornerBlockMapArray,
								blockCheck);
	// */
	// std::cout << "prefetch points on ellipse number: " << range[index].size() << std::endl;
	float next_range_dis;   // current d
    float next_range_theta; // current theta
    float next_range_phi;   // current phi
	std::vector<int> visible_blocks_current;
	for (int i = 0; i < range[index].size(); i++) {
		next_range_dis = range[index].at(i).at(0);   // current d
        next_range_theta = range[index].at(i).at(1); // current theta
        next_range_phi = range[index].at(i).at(2);   // current phi
		visible_blocks_current.clear();
		/*
		getCurrentVisibleBlocks(next_range_theta, next_range_phi, next_range_dis,
								volume_size_1, volume_size_2, volume_size_3, volume_size_4,
								block_size,
								angle,
								visible_blocks_current);
		*/
		// /*
		getCurrentVisibleBlocksFast(next_range_theta, next_range_phi, next_range_dis,
									volume_size_1, volume_size_2, volume_size_3, volume_size_4,
									block_size,
									angle,
									visible_blocks_current,
									cornerBlockMapArray,
									blockCheck);
		// */
		combineVisibleBlocks(visible_blocks, visible_blocks_current);
	}
}

void getCurrentVisibleBlocksRangeRealtime(float theta, float phi, float dis,
							 	  		  int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							 	  		  int block_size,
							 	  		  float angle,
							 	  		  std::vector<int> &visible_blocks,
								  		  std::vector<float> thetas, std::vector<float> phis,
								  		  int *cornerBlockMapArray,
										  int *blockCheck) {
	/*
	getCurrentVisibleBlocks(theta, phi, dis,
							volume_size_1, volume_size_2, volume_size_3, volume_size_4,
							block_size,
							angle,
							visible_blocks);
	*/
	// /*
	getCurrentVisibleBlocksFast(theta, phi, dis,
								volume_size_1, volume_size_2, volume_size_3, volume_size_4,
								block_size,
								angle,
								visible_blocks,
								cornerBlockMapArray,
								blockCheck);
    // */
	// std::cout << "prefetch points on ellipse number: " << range[index].size() << std::endl;
	float next_range_theta; // current theta
    float next_range_phi;   // current phi
	std::vector<int> visible_blocks_current;
	int size = thetas.size(); // number of points on the ellipse
	for (int i = 0; i < size; i++) {
        next_range_theta = thetas.at(i); // current theta
        next_range_phi = phis.at(i);   // current phi
		visible_blocks_current.clear();
		/*
		getCurrentVisibleBlocks(next_range_theta, next_range_phi, dis,
								volume_size_1, volume_size_2, volume_size_3, volume_size_4,
								block_size,
								angle,
								visible_blocks_current);
		*/
		// /*
		getCurrentVisibleBlocksFast(next_range_theta, next_range_phi, dis,
									volume_size_1, volume_size_2, volume_size_3, volume_size_4,
									block_size,
									angle,
									visible_blocks_current,
									cornerBlockMapArray,
									blockCheck);
		// */
		combineVisibleBlocks(visible_blocks, visible_blocks_current);
	}
}

/*
void getCurrentVisibleBlocksLUT(float theta, float phi, float dis,
							 	std::vector<int> &visible_blocks) {
    float delta_d = 0.1;
    float delta_theta = 6.0;
    float delta_phi = 6.0;
	char view_pos[50]; 
    float norm_dis, norm_theta, norm_phi;


    double intpart, fractpart;
    fractpart = modf(double(dis/delta_d) , &intpart);
    norm_dis = float(intpart)*delta_d;
	fractpart = modf(double(theta/delta_theta) , &intpart);
    norm_theta = float(intpart)*delta_theta;
    fractpart = modf(double(phi/delta_phi) , &intpart);
    norm_phi = float(intpart)*delta_phi;
    sprintf(view_pos, "%4.2f%5.1f%5.1f", norm_dis, norm_theta, norm_phi);
	std::map<std::string, std::vector<int> >::iterator sample;
    sample = all_map->find(view_pos);
	std::cout << sample->second << std::endl;
}
*/
int updateCacheAndCacheData(std::vector<int> &cache, std::vector<int> &cache_mem, void *cache_data, int visible_block_index, cudaExtent microBlockSize, std::string method, const char *volumeFilePath, int camera_index) {
	size_t block_size = microBlockSize.width*microBlockSize.height*microBlockSize.depth*sizeof(VolumeType);
	int hit_miss = -1; // hit: 1; miss: 0
	if (method == "lru") {
		int hit_position = 0;
		if (cache.size() == 0) { // miss and cache is empty
			hit_miss = 0;
			cache.push_back(visible_block_index);
			cache_mem.push_back(visible_block_index);

			std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data, block, block_size);

		} else if (isPresent(cache, visible_block_index, hit_position)) { // hit
			hit_miss = 1;
			int temp = cache.at(hit_position);
			cache.erase(cache.begin() + hit_position);
			cache.push_back(temp);
		} else if (cache.size() < CACHE_SIZE) { // miss and cache is not full, do appending
			hit_miss = 0;
			cache.push_back(visible_block_index);

			int write_position = cache_mem.size();
			cache_mem.push_back(visible_block_index);
			std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data + block_size*write_position, block, block_size);

		} else { // miss and cache is full, do replacement
			hit_miss = 0;
			int lru_index = cache.at(0);
			cache.erase(cache.begin());
			cache.push_back(visible_block_index);

			int replace_position;
			if(isPresent(cache_mem, lru_index, replace_position)) {
				cache_mem.at(replace_position) = visible_block_index;
				std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
				void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
				memcpy(cache_data + block_size*replace_position, block, block_size);
			} else {
				std::cout << "Cannot find lru element in data_mem." << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return hit_miss;
}

void updateCache(std::vector<int> &cache, std::vector<int> visible_blocks, std::string method, int &hit, int &miss) {
	// std::cout << "Cache Size: " << cache.size() << std::endl;
	if (method == "lru") {
		for (int i = 0; i < visible_blocks.size(); i++) {
			int hit_position = 0;
			if (cache.size() == 0) { // miss and cache is empty
				miss++;
				cache.push_back(visible_blocks.at(i));
			} else if (isPresent(cache, visible_blocks.at(i), hit_position)) { // hit
				hit++;
				int temp = cache.at(hit_position);
				cache.erase(cache.begin() + hit_position);
				cache.push_back(temp);
			} else if (cache.size() < CACHE_SIZE) { // miss and cache is not full, do appending
				miss++;
				// std::cout << "miss on: " << visible_blocks.at(i) << std::endl;
				cache.push_back(visible_blocks.at(i));
			} else { // miss and cache is full, do replacement
				miss++;
				cache.erase(cache.begin());
				cache.push_back(visible_blocks.at(i));
				// std::cout << "miss on: " << visible_blocks.at(i) << std::endl;
			}

		}
	}
	// std::cout << "hit:  " << hit << std::endl;
	std::cout << "miss: " << miss << std::endl;
	std::cout << "cache size: " << cache.size() << std::endl;
}	

void updateCacheData(void *cache_data, std::vector<int> &cache_mem, std::vector<int> cache, int cache_size, cudaExtent microBlockSize, const char *volumeFilePath) {
	size_t block_size = microBlockSize.width*microBlockSize.height*microBlockSize.depth*sizeof(VolumeType);
	// int write_position = 0;
	int hit_position = 0;
	
	bool test = false;


	for (int i = 0; i < cache.size(); i++) {
		if (cache_mem.size() == 0) {
			cache_mem.push_back(cache.at(i));
			std::string block_file_name = volumeFilePath + std::to_string(cache.at(i)) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data, block, block_size);
			// write_position++;
			test = true;
		} else if (isPresent(cache_mem, cache.at(i), hit_position)) {
			// hit
		} else if (cache_mem.size() < cache_size) { // miss, cache not full
			int write_position = cache_mem.size();
			cache_mem.push_back(cache.at(i));
			std::string block_file_name = volumeFilePath + std::to_string(cache.at(i)) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data + block_size*write_position, block, block_size);
			// write_position++;
			test = true;
			// std::cout << "miss: " << cache.at(i) << "insert position: " << write_position << std::endl;
		} else { // miss, cache is full
			// Find blocks indexes in cache but not in cache_mem
			std::vector<int> new_blocks;
			for (int j = 0; j < cache.size(); j++) {
				if (!isPresent(cache_mem, cache.at(j), hit_position)) {
					new_blocks.push_back(cache.at(j));
				}
			}
			// Find blocks indexes in cache_mem but not in cache
			// std::vector<int> old_blocks;
			std::vector<int> old_blocks_positions;
			for (int j = 0; j < cache_mem.size(); j++) {
				if (!isPresent(cache, cache_mem.at(j), hit_position)) {
					// old_blocks.push_back(cache_mem.at(j));
					old_blocks_positions.push_back(j);
				}
			}
			// Do replacement
			for (int j = 0; j < old_blocks_positions.size(); j++) {
				cache_mem.at(old_blocks_positions.at(j)) = new_blocks.at(j);
				std::string block_file_name = volumeFilePath + std::to_string(new_blocks.at(j)) + ".blk";
    			void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
				memcpy(cache_data + block_size*old_blocks_positions.at(j), block, block_size);
			}
			test = true;
		}
	}
	if (test) {
		std::cout << ">>>>>>>miss mem operation happends" << std::endl;
	}
	/*
	std::cout << "============================" << std::endl;
	std::cout << "cache size: " << cache.size() << std::endl;
	for (int i = 0; i < cache.size(); i++) {
		std::cout << cache.at(i) << ",";
	}
	std::cout << std::endl;
	std::cout << "cache_mem size:" << cache_mem.size() << std::endl;
	for (int i = 0; i < cache_mem.size(); i++) {
		std::cout << cache_mem.at(i) << ",";
	}
	std::cout << std::endl;
	*/
}

void getVisibleBlocksData(void *cache_data, std::vector<int> cache_mem, void *h_visible_blocks_data, std::vector<int> visible_blocks, cudaExtent microBlockSize) {
	size_t block_size = microBlockSize.width*microBlockSize.height*microBlockSize.depth*sizeof(VolumeType);
	std::cout << "hit position: " << std::endl;
	for (int i = 0; i < visible_blocks.size(); i++) {
		int hit_position = 0;
		if (isPresent(cache_mem, visible_blocks.at(i), hit_position)) {
			std::cout << hit_position << " ";
			memcpy(h_visible_blocks_data + block_size*i, cache_data + block_size*hit_position, block_size);
		} else {
			std::cout << "Cannot find specific visible block in the current cached memory, wrong! Exits." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	std::cout << std::endl;
}

