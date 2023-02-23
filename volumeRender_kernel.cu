/*
 * Jianxin (Jason) Sun, sunjianxin66@gmail.com
 * Visualization Lab
 * School of Computing
 * University of Nebraska-Lincoln
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#define X_BLOCK_NUM_LEVEL_1	  2
#define Y_BLOCK_NUM_LEVEL_1   2
#define Z_BLOCK_NUM_LEVEL_1	  2
#define X_BLOCK_NUM_LEVEL_2	  4
#define Y_BLOCK_NUM_LEVEL_2	  4
#define Z_BLOCK_NUM_LEVEL_2	  4
#define X_BLOCK_NUM_LEVEL_3	  8
#define Y_BLOCK_NUM_LEVEL_3	  8
#define Z_BLOCK_NUM_LEVEL_3	  8
#define X_BLOCK_NUM_LEVEL_4	  16
#define Y_BLOCK_NUM_LEVEL_4	  16
#define Z_BLOCK_NUM_LEVEL_4	  16


typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
// cudaArray *d_volumeArrays[3];
cudaArray *d_transferFuncArray;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;         // 3D texture
cudaTextureObject_t texObj = 0;
cudaTextureObject_t *texObjList;
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

cudaExtent allBlockSize = make_cudaExtent(2, 2, 2); // dimension of sub blocks of the original volume
int* selectListKernel;
int* visibleBlocksListKernel;
int visibleBlocksSizeKernel;
int selectBlockSizeKernel;

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__
float3 getNormal(float px, float py, float pz, int size_x, int size_y, int size_z)
{
     float3 gradient;
     float3 theCentDiffSpace;

	 theCentDiffSpace.x = 1.0/size_x; // 1.0/77 
	 theCentDiffSpace.y = 1.0/size_y; // 1.0/77
	 theCentDiffSpace.z = 1.0/size_z; // 1.0/(77*vb_size)
/*
      gradient.x = tex3D(tex, px + theCentDiffSpace.x, py, pz)
           		  -tex3D(tex, px - theCentDiffSpace.x, py, pz);
      
      gradient.y = tex3D(tex, px, py + theCentDiffSpace.y, pz)
           		  -tex3D(tex, px, py - theCentDiffSpace.y, pz);
      
      gradient.z = tex3D(tex, px, py, pz + theCentDiffSpace.z)
		   		  -tex3D(tex, px, py, pz - theCentDiffSpace.z);
*/
	  gradient.x = tex3D(tex, px + theCentDiffSpace.x, py, pz)
           		  -tex3D(tex, px, py, pz);
      
      gradient.y = tex3D(tex, px, py + theCentDiffSpace.y, pz)
           		  -tex3D(tex, px, py, pz);
      
      gradient.z = tex3D(tex, px, py, pz + theCentDiffSpace.z)
		   		  -tex3D(tex, px, py, pz);

      
      gradient = gradient * 10.0;
      
      if(length(gradient) > 0.0) {
          gradient = normalize(gradient);
      }
      
      return gradient;
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, int* selectList, int* visibleBlocksList, int visibleBlocksSize,
		 int allBlockSizeWidth_level_1, int allBlockSizeHeight_level_1, int allBlockSizeDepth_level_1,
		 int allBlockSizeWidth_level_2, int allBlockSizeHeight_level_2, int allBlockSizeDepth_level_2,
		 int allBlockSizeWidth_level_3, int allBlockSizeHeight_level_3, int allBlockSizeDepth_level_3,
		 int allBlockSizeWidth_level_4, int allBlockSizeHeight_level_4, int allBlockSizeDepth_level_4,
		 int selectBlockSize, 
		 float sampleDistance, 
		 int size_x, int size_y, int size_z)
{
#if 0
    // const int maxSteps = 50;
    // const int maxSteps = 500;
    const int maxSteps = 5000;
    // const float tstep = 0.1f;
    // const float tstep = 0.01f;
    const float tstep = 0.001f;
#endif
	// printf("sample distance: %f\n", sampleDistance);
    // const int maxSteps = 50000;
    const int maxSteps = 500000000;
    // const float tstep = 0.1f;
    const float tstep = sampleDistance;
    // const float tstep = 0.02f;
	// ...

    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    // const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

	// if ((x == 0) && (y == 0)) {
	// 	printf("a");
	// 			// printf("%f, %f, %f\n", px, py, pz);
	// }



    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    // eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = normalize(make_float3(u, v, -8.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

	// bool flag = true;
    for (int i = 0; i < maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        // float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);


		float allBlockSizeWidth;
		float allBlockSizeHeight;
		float allBlockSizeDepth;
		bool found = false;
		float x_shift;
		float y_shift;
		float z_shift;
		float unit_x;
		float unit_y;
		float unit_z;
		int x_idx;
		int y_idx;
		int z_idx;
		int idx;
		int current_level;
		for (int level = 0; level < 4; level++) {
			if (level == 0) {
				allBlockSizeWidth = allBlockSizeWidth_level_1;
				allBlockSizeHeight = allBlockSizeHeight_level_1;
				allBlockSizeDepth = allBlockSizeDepth_level_1;
			}
			if (level == 1) {
				allBlockSizeWidth = allBlockSizeWidth_level_2;
				allBlockSizeHeight = allBlockSizeHeight_level_2;
				allBlockSizeDepth = allBlockSizeDepth_level_2;
			}
			if (level == 2) {
				allBlockSizeWidth = allBlockSizeWidth_level_3;
				allBlockSizeHeight = allBlockSizeHeight_level_3;
				allBlockSizeDepth = allBlockSizeDepth_level_3;
			}
			if (level == 3) {
				allBlockSizeWidth = allBlockSizeWidth_level_4;
				allBlockSizeHeight = allBlockSizeHeight_level_4;
				allBlockSizeDepth = allBlockSizeDepth_level_4;
			};

			// find the block containing the current sample position
			x_shift = pos.x*0.5f+0.5f; // from (-1, 1) to (0, 1)
			y_shift = pos.y*0.5f+0.5f; // from (-1, 1) to (0, 1)
			z_shift = pos.z*0.5f+0.5f; // from (-1, 1) to (0, 1)
			unit_x = 1/(float)allBlockSizeWidth;
			unit_y = 1/(float)allBlockSizeHeight;
			unit_z = 1/(float)allBlockSizeDepth;
			x_idx = x_shift/unit_x;
			y_idx = y_shift/unit_y;
			z_idx = z_shift/unit_z;
			idx = z_idx*allBlockSizeWidth*allBlockSizeHeight + y_idx*allBlockSizeWidth + x_idx;
			if (level == 0) {
				idx = idx;
				current_level = level;
			}
			if (level == 1) {
				idx = idx + 8;
				current_level = level;
			}
			if (level == 2) {
				idx = idx + 8 + 64;
				current_level = level;
			}
			if (level == 3) {
				idx = idx + 8 + 64 + 512;
				current_level = level;
			}

			if (selectList[idx] != 0) {
				found = true;
				break;
			}
		}

		float sample = 0.0;
		float px, py, pz;
		float diffuse = 1.0; 

		if (found) {
			// find index in select block list
			int counter = 0;
			// for (int j = 0; j < idx; j++) {
			// 	if (selectList[j] == 1) {
			// 		counter++;
			// 	}
			// }
			for (int j = 0; j < visibleBlocksSize; j++) {
				if (visibleBlocksList[j] == idx) {
					break;
				} else {
					counter++;
				}
			}
			// float unit = 1.0/((float)selectBlockSize*75 + (float)selectBlockSize - 1); // for initCuda
			// float unit = 1.0/((float)visibleBlocksSize*75 + (float)visibleBlocksSize - 1); //  for initCuda2
			// float unit = 1.0/((float)visibleBlocksSize*76 + (float)visibleBlocksSize - 1); //  for initCuda2
			float unit = 1.0/((float)visibleBlocksSize*76 + (float)visibleBlocksSize - 1 + 1); //  for initCuda2
			// printf("vb size: %d\n", visibleBlocksSize);
			float z_shift_unit = unit*(76 + 1);
			// float z_unit = unit*76;

			float x_left = x_shift - unit_x*(float)x_idx;
			float y_left = y_shift - unit_y*(float)y_idx;
			float z_left = z_shift - unit_z*(float)z_idx;
			px = x_left*(float)allBlockSizeWidth;
			py = y_left*(float)allBlockSizeHeight;
			pz = z_left*(float)allBlockSizeDepth;
			// if ((px > 1.0) || (px < 0.0) || (py > 1.0) || (py < 0.0) || (pz > 1.0) || (pz < 0.0)) {
			// 	printf("jump: %f, %f, %f\n", px, py, pz);
			// }
			// if ((px > 1.0) || (py > 1.0) || (pz > 1.0)) {
			// 	printf("jump: %f, %f, %f\n", px, py, pz);
			// }
			
			if (px > 1.0) {px = 1.0;}
			if (py > 1.0) {py = 1.0;}
			if (pz > 1.0) {pz = 1.0;}

			if (px < 0.0) {px = 0.0;}
			if (py < 0.0) {py = 0.0;}
			if (pz < 0.0) {pz = 0.0;}

			// Adjustment for gradient extended block size, extended block size = traditional block size + 1
			
			float uunit = 1.0/77.0;
			px = px*75.0*uunit + uunit/2.0;
			py = py*75.0*uunit + uunit/2.0;
        	pz = (float)counter*z_shift_unit + unit/2.0 + pz*75.0*unit;

			// px = px*74.5/76.0;
			// px = px*38.0/76.0;
			// px = px*38.5/76.0;
			// px = px*38.5/76.0 + 1.5/77.0;
			// py = py*74.5/76.0;
			// pz = pz*76.0/76.0;
        	// pz = (float)counter*z_shift_unit + pz*z_unit;
			// if (current_level == 0 || current_level == 1 || current_level == 2 || current_level == 3) {
			// if ((pos.x < 0.01 && pos.x > -0.01) || (pos.y < 0.01 && pos.y > -0.01)) {
			// if ((pos.x < 0.01) || (pos.y < 0.01)) {
			// if (pos.y < 0.1 && pos.y > -0.1) {
		 		sample = tex3D(tex, px, py, pz);
			// }
			
			

			if (idx == 28) {
				// printf(" %d", visibleBlocksSize);
			}

			// sample = tex3D(tex, px, py, pz) + 0.5;
			// }
#if 0
			if (current_level == 0) {
				sample = 0.1;
			}
			if (current_level == 1) {
				sample = 0.3;
			}
			if (current_level == 2) {
				sample = 0.5;
			}
			if (current_level == 3) {
				sample = 0.7;
			}
#endif
		}
#if 0
		// find the block containing the current sample positionmember
		float x_shift = pos.x*0.5f+0.5f; // from (-1, 1) to (0, 1)
		float y_shift = pos.y*0.5f+0.5f; // from (-1, 1) to (0, 1)
		float z_shift = pos.z*0.5f+0.5f; // from (-1, 1) to (0, 1)
		float unit_x = 1/(float)allBlockSizeWidth_level_1;
		float unit_y = 1/(float)allBlockSizeHeight_level_1;
		float unit_z = 1/(float)allBlockSizeDepth_level_1; int x_idx = x_shift/unit_x;
		int y_idx = y_shift/unit_y;
		int z_idx = z_shift/unit_z;
		int idx = z_idx*allBlockSizeWidth_level_1*allBlockSizeHeight_level_1 + y_idx*allBlockSizeWidth_level_1 + x_idx;
		// float z_unit = 1.0/((float)selectBlockSize);
		float unit = 1.0/((float)selectBlockSize*75 + (float)selectBlockSize - 1);
		float z_shift_unit = unit*(75 + 1);
		float z_unit = unit*75;
		
		float sample = 0.0;
		float px, py, pz;

		if (selectList[idx] != 0) {
			// printf("%d\n", idx);
			// find index in select block list
			int counter = 0;
			for (int j = 0; j < idx; j++) {
				if (selectList[j] == 1) {
					counter++;
				}
			}
		member	
			float x_left = x_shift - unit_x*(float)x_idx;
			float y_left = y_shift - unit_y*(float)y_idx;
			float z_left = z_shift - unit_z*(float)z_idx;
			px = x_left*(float)allBlockSizeWidth_level_1;
			py = y_left*(float)allBlockSizeHeight_level_1;
			pz = z_left*(float)allBlockSizeDepth_level_1;
        	pz = (float)counter*z_shift_unit + pz*z_unit;
			sample = tex3D(tex, px, py, pz);
			// printf("%f, %f, %f\n", px, py, pz);
			// sample = tex3D(tex, x_shift, y_shift, z_shift);
			// sample = tex3D(tex, 0.5, 0.5, 0.5);
			// if (sample > 0.1) {
			// 		printf("%d, %f\n", idx, sample);
			// }
			// sample = 0.5;
			// printf("%f\n", sample);

#if 0tex3D
			float px = pos.x;
			float py = pos.y;
			float pz = pos.z;
			if (px < 0) { px = px + 1; } 
			if (py < 0) { py = py + 1; } 
			if (pz < 0) { pz = pz + 1; } 
        	pz = (float)counter*z_unit + pz*z_unit;
			// sample = tex3D(tex, px, py, pz);
			// sample = tex3D(tex, z_shift, y_shift, x_shift);
			sample = tex3D(tex, x_shift, y_shift, z_shift);
#endif
		}

		// if ((pos.x >= 0 && pos.x <= 1) && (pos.y >= 0 && pos.y <= 1) && (pos.z >= 0 && pos.z <= 1)) {
			/*
			if (flag) {
				printf("A\n");
				flag = false;
			}
			*/
			// printf("lala");
	        // float sample = tex3D(tex, pos.x, pos.y, pos.z);
	        // sample = tex3D(tex, pos.x, pos.y, pos.z/2);
	        // sample = tex3D<float>(texObj, pos.x, pos.y, pos.z);
	        // sample = tex3D<float>(texObjList[0], pos.x, pos.y, pos.z);
		// } else {
			// break;
			// return;
		// }
		// if ((pos.x >= -1 && pos.x <= 0) && (pos.y >= -1 && pos.y <= 0) && (pos.z >= -1 && pos.z <= 0)) {
			/*
			if (flag) {
				printf("A\n");
				flag = false;
			}
			*/
			// printf("lala");
	        // float sample = tex3D(tex, pos.x, pos.y, pos.z);
	        // sample = tex3D(tex, pos.x + 1, pos.y + 1, (pos.z + 1)/2 + 0.5);
	        // sample = tex3D<float>(texObj_1, -pos.x, -pos.y, -pos.z);
	        // sample = tex3D<float>(texObjList[1], -pos.x, -pos.y, -pos.z);
		// } else {
			// break;
			// return;
		// }
#endif
        // sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);

		// if (idx <= 3) {
			// col = {0.0, 0.0, pz, 1.0};
		// }
        col.w *= density;

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
	    if (col.w > 0.001) {
			float3 normal = getNormal(px, py, pz, size_x, size_y, size_z);
			// diffuse = max(dot(normal, eyeRay.d), dot(eyeRay, -normal));
			diffuse = 0.5 +  max(normal.x*eyeRay.d.x + normal.y*eyeRay.d.y + normal.z*eyeRay.d.z, 
						  (-normal.x*eyeRay.d.x) + (-normal.y*eyeRay.d.y) + (-normal.z*eyeRay.d.z));
			col.x *= diffuse;
			col.y *= diffuse;
			col.z *= diffuse;
		}

        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize, int* selectList, int* visibleBlocksList, int visibleBlocksSize, int selectBlockSize)
{
#if 1
	checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));

	selectListKernel = selectList;
	visibleBlocksListKernel = visibleBlocksList;
	visibleBlocksSizeKernel = visibleBlocksSize;
	selectBlockSizeKernel = selectBlockSize;
    // create 3D array
	clock_t start, end;
	start = clock();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
	end = clock();
	// printf("Loading from CPU Mem to GPU Mem: %4.6f\n", (double)((double)(end - start)/CLOCKS_PER_SEC));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    // tex.filterMode = cudaFilterModePoint;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

#endif
    // create transfer function texture
#if 1
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };
#endif
	// RGBA
#if 0
	float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 1.0, 1.0, },
    };
#endif
    // printf("transferFunc: %f\n", transferFunc[1].x);
    // printf("transferFunc: %f\n", transferFunc[1].y);
    // printf("transferFunc: %f\n", transferFunc[1].z);
    // printf("transferFunc: %f\n", transferFunc[1].w);

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

float getValue(float start, float startValue, float end, float endValue, float ref) {
	if (startValue > endValue) {
		float l = end - ref;
		float ll = end - start;
		float hh = startValue - endValue;
		return l/ll*hh + endValue;
	} else if (startValue < endValue) {
		float l = ref;
		float ll = end - start;
		float hh = endValue - startValue;
		return l/ll*hh + startValue;
	} else {
		return startValue;
	}
}


void getTransferFunc(float4* transferFunc, int colorTFSize, float4* colorTransferFunc, int opacityTFSize, float2* opacityTransferFunc) {
	float r[101];
	float g[101];
	float b[101];
	float a[101];
	// fill color
	for (int i = 0; i < 101; i++) {
		float currentValue = (float)i/100; // [0, 1]
		// printf("%f\n", currentValue);
		for (int j = 0; j < colorTFSize - 1; j++) {
			float lowerBound = colorTransferFunc[j].x;
			float upperBound = colorTransferFunc[j + 1].x;
			// printf("%f\n", lowerBound);
			// printf("%f\n", upperBound);
			if (currentValue >= lowerBound && currentValue <= upperBound) {
				r[i] = getValue(lowerBound, colorTransferFunc[j].y, upperBound, colorTransferFunc[j + 1].y, currentValue);
				g[i] = getValue(lowerBound, colorTransferFunc[j].z, upperBound, colorTransferFunc[j + 1].z, currentValue);
				b[i] = getValue(lowerBound, colorTransferFunc[j].w, upperBound, colorTransferFunc[j + 1].w, currentValue);
			}
		}
	}
	// fill opacity
	for (int i = 0; i < 101; i++) {
		float currentValue = (float)i/100; // [0, 1]
		for (int j = 0; j < opacityTFSize - 1; j++) {
			float lowerBound = opacityTransferFunc[j].x;
			float upperBound = opacityTransferFunc[j + 1].x;
			if (currentValue >= lowerBound && currentValue <= upperBound) {
				a[i] = getValue(lowerBound, opacityTransferFunc[j].y, upperBound, opacityTransferFunc[j + 1].y, currentValue);
			}
		}
	}
	for (int i = 0; i < 101; i++) {
		transferFunc[i] = {r[i], g[i], b[i], a[i]};
		// printf("%f", r[i]);
	}
}

extern "C"
void initCuda2(void *h_volume, cudaExtent volumeSize, int* selectList, int* visibleBlocksList, int visibleBlocksSize, int selectBlockSize)
{
#if 1
	checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));

	selectListKernel = selectList;
	visibleBlocksListKernel = visibleBlocksList;
	visibleBlocksSizeKernel = visibleBlocksSize; // cache size = 200
	selectBlockSizeKernel = selectBlockSize; // visible block number
    // create 3D array
	clock_t start, end;
	start = clock();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
	end = clock();
	// printf("Loading from CPU Mem to GPU Mem: %4.6f\n", (double)((double)(end - start)/CLOCKS_PER_SEC));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

#endif
    // create transfer function texture
#if 0
	float4 transferFunc[101];
	int colorTFSize = 3;
	float4 colorTransferFunc[] = {
		{ 0, 0.231373, 0.298039, 0.752941 },
		{ 0.5, 0.865003, 0.865003, 0.865003 }, 
		{ 1.0, 0.705882, 0.0156863, 0.14902 },
	};
	int opacityTFSize = 9;
	float2 opacityTransferFunc[] = {
		{0, 0},
		{0.05, 0.0},
		{0.06, 0.6},
		{0.07, 0.0},
		{0.23, 0.0},
		{0.27, 0.8},
		{0.31, 0.0},
		{0.5, 0.0},
		{1.0, 1.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif
#if 0
	float4 transferFunc[101];
	int colorTFSize = 4;
	float4 colorTransferFunc[] = {
		{ 0, 0.231373, 0.298039, 0.752941 },
		{ 0.270068, 0.865003, 0.865003, 0.865003 }, 
		{ 0.552149, 0.7, 0.01, 0.14 }, 
		{ 1.0, 0.705882, 0.0156863, 0.14902 },
	};
	int opacityTFSize = 11;
	float2 opacityTransferFunc[] = {
		{0, 0},
		{0.04, 0.0},
		{0.06, 0.1},
		{0.08, 0.0},
		{0.2, 0.0},
		{0.24, 0.15},
		{0.28, 0.0},
		{0.36, 0.0},
		{0.4, 0.5},
		{0.48, 0.0},
		{1.0, 1.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif

#if 1
	float4 transferFunc[101];
	int colorTFSize = 3;
	float4 colorTransferFunc[] = {
		{ 0, 0.0, 0.0, 1.0 },
		{ 0.5, 0.0, 1.0, 0.0 }, 
		{ 1.0, 1.0, 0.0, 0.0 },
	};
	int opacityTFSize = 16;
	float2 opacityTransferFunc[] = {
		{0, 0},
		{0.1, 0.0},
		{0.11, 0.1},
		{0.14, 0.1},
		{0.15, 0.0},
		{0.21, 0.0},
		{0.22, 0.2},
		{0.25, 0.2},
		{0.26, 0.0},
		{0.37, 0.0},
		{0.38, 0.7},
		{0.41, 0.7},
		{0.42, 0.0},
		{0.73, 0.0},
		{0.74, 1.0},
		{1.0, 1.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif

#if 0
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };
#endif
	// RGBA
#if 0
	float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        // {  1.0, 0.0, 0.0, 0.33, },
        {  0.0, 1.0, 0.0, 0.66, },
        {  0.0, 0.0, 1.0, 1.0, },
    };
#endif
    // printf("transferFunc: %f\n", transferFunc[1].x);
    // printf("transferFunc: %f\n", transferFunc[1].y);
    // printf("transferFunc: %f\n", transferFunc[1].z);
    // printf("transferFunc: %f\n", transferFunc[1].w);

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale, float sampleDistance, int size_x, int size_y, int size_z)
{
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                      // brightness, transferOffset, transferScale);
                                      brightness, transferOffset, transferScale, selectListKernel, visibleBlocksListKernel, visibleBlocksSizeKernel,
									  X_BLOCK_NUM_LEVEL_1, Y_BLOCK_NUM_LEVEL_1, Z_BLOCK_NUM_LEVEL_1,
									  X_BLOCK_NUM_LEVEL_2, Y_BLOCK_NUM_LEVEL_2, Z_BLOCK_NUM_LEVEL_2,
									  X_BLOCK_NUM_LEVEL_3, Y_BLOCK_NUM_LEVEL_3, Z_BLOCK_NUM_LEVEL_3,
									  X_BLOCK_NUM_LEVEL_4, Y_BLOCK_NUM_LEVEL_4, Z_BLOCK_NUM_LEVEL_4,
									  selectBlockSizeKernel,
									  sampleDistance, 
									  size_x, size_y, size_z);
                                      // brightness, transferOffset, transferScale, texObjList[1]);
                                      // brightness, transferOffset, transferScale, texObjList[0], texObjList[1]);
                                      // brightness, transferOffset, transferScale, texObjList);
	cudaDeviceSynchronize();

}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
