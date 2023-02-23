/*
 * Jianxin (Jason) Sun, sunjianxin66@gmail.com
 * Visualization Lab
 * School of Computing
 * University of Nebraska-Lincoln
 *
 * Volume rendering using rmdnCache
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif


// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include <iostream>
#include <time.h>
#include <math.h>
#include <chrono>
#include "utility.hpp"
#include <unistd.h>
#include <pthread.h>
#include <map>

// libtorch
// #include <torch/torch.h>
// #include "/home/js/Downloads/libtorch/include/torch/script.h"
// #include "/home/js/Downloads/libtorch/include/torch/csrc/api/include/torch/torch.h"

typedef unsigned int uint;
typedef unsigned char uchar;

	
// 2x2x2 microblocks, level 1 volume size
#define VOLUME_SIZE_1        151
// 4x4x4 microblocks, level 2 volume size
#define VOLUME_SIZE_2        301
// 8x8x8 microblocks, level 3 volume size
#define VOLUME_SIZE_3        601
// 16x16x16 microblocks, level 4 volume size
#define VOLUME_SIZE_4        1201
// Microblock size
#define BLOCK_SIZE    77

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

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

#define BLOCK_SIZE_EACH_DIMENSION 77

bool doingRendering = false;
bool doingPrefetching = false;
bool prefetchOn = false;

int early_stop = 0;

struct threadData {
	int argc;
	char **argv;
};

const char *sSDKsample = "CUDA 3D Volume Render using RmdnCache";

// const char *volumeFilePath = "../data/data/blocks/";
// const char *volumeFilePath = "../data/data/blocks_shrink/";
const char *volumeFilePath = "../data/data/blocks_shrink_extend/";
const char *volumeFilename = "/home/js/ws/notebook/flameDsDs/0.blk";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
cudaExtent microBlockSize = make_cudaExtent(BLOCK_SIZE_EACH_DIMENSION, BLOCK_SIZE_EACH_DIMENSION, BLOCK_SIZE_EACH_DIMENSION);
typedef float VolumeType;

uint width = 512, height = 512;
// uint width = 1024, height = 1024;
// uint width = 1536, height = 1536;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
// float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float3 viewTranslation = make_float3(0.0, 0.0, 0.0f);
float invViewMatrix[12];

// float density = 0.05f;
// float density = 0.01f;
float density = 0.005f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

float *dis = NULL;
float *theta = NULL;
float *phi = NULL;

float *dis_predict = NULL;
float *theta_predict = NULL;
float *phi_predict = NULL;
std::string camera_trajectory_predict;

std::vector<int> cache; // block indexes list
std::vector<int> cache_mem; // block indexes list of the current cache memory
std::vector<int> visible_blocks;
std::vector<int> hit_blocks;
std::vector<int> miss_blocks;
std::vector<int> visible_blocks_predict;

int camera_index = 0;
void *h_visible_blocks_data_1 = malloc(173834496);
void *h_visible_blocks_data_2 = malloc(173834496);
void *cache_data_1 = malloc(351180800); // All blocks data in cache
void *cache_data_2 = malloc(351180800); // All blocks data in cache

size_t cache_data_size = microBlockSize.width*microBlockSize.height*microBlockSize.depth*sizeof(VolumeType)*CACHE_SIZE;
void *cache_data = malloc(cache_data_size); // All blocks data in cache

float render_time = 0;
float prefetch_cacheTime = 0;
float prefetch_find_visible_blocks_time = 0;
float inference_time = 0;

bool exit_by_user = false;

float sampleDistance = 0;
std::map<int, std::vector<std::vector<float>>> range;

std::string method_used = "lru";
std::map<std::string, std::vector<int>> all_map;

int size_x;
int size_y;
int size_z;

std::string imagePath = "screenShots/";
// int test_size = 400; // for benchmarks
int test_size = 144; // only for demo
// int test_size = 60; // only for demo

torch::jit::script::Module net_lstm;
torch::jit::script::Module net_mdn;
torch::Tensor seq = torch::tensor({{{0.0, 0.0, 0.0},
									{0.0, 0.0, 0.0},
									{0.0, 0.0, 0.0}}});

std::map<int, std::vector<std::vector<int>>> cornerBlockMap;
int *cornerBlockMapArray;
int *cornerVisibility;
int *blockCheck;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize, int* selectList, int* visibleBlocksList, int visibleBlocksSize, int selectBlockSize);
extern "C" void initCuda2(void *h_volume, cudaExtent volumeSize, int* selectList, int* visibleBlocksList, int visibleBlocksSize, int selectBlockSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, float sampleDistance, int size_x, int size_y, int size_z);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();

void computeFPS() {
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		// printf("%3.9f\n", (sdkGetAverageTimerValue(&timer) / 1000.f));
        // sprintf(fps, "Volume Render: %3.1f fps", ifps);
        sprintf(fps, "Rendering Time: %3.3f s", (sdkGetAverageTimerValue(&timer) / 1000.f));

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

// render image using CUDA
void render() {
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, sampleDistance, size_x, size_y, size_z);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

bool save_screenshot(std::string filename, int w, int h)
{	
  
  std::cout << "Writing " << filename << std::endl;
  //This prevents the images getting padded 
  // when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
 
  int nSize = w*h*3;
  // First let's create our buffer, 3 channels per Pixel
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));
 
  if (!dataBuffer) return false;
 
   // Let's fetch them from the backbuffer	
   // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
   glReadPixels((GLint)0, (GLint)0,
		(GLint)w, (GLint)h,
		 GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);

   //Now the file creation
   FILE *filePtr = fopen(filename.c_str(), "wb");
   if (!filePtr) return false;
 
 
   unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
   unsigned char header[6] = { w%256,w/256,
			       h%256,h/256,
			       24,0};
   // We write the headers
   fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
   fwrite(header,	sizeof(unsigned char),	6,	filePtr);
   // And finally our image data
   fwrite(dataBuffer,	sizeof(GLubyte),	nSize,	filePtr);
   fclose(filePtr);
 
   free(dataBuffer);
 
  return true;
}


// display results using OpenGL (called by GLUT)
void display() {
	while (doingPrefetching) {} // Wait for prefetching finished
	// std::cout << "In display call, camera index: " << camera_index << std::endl;
	doingRendering = true;

cudaEvent_t start_event, stop_event;
cudaEventCreate(&start_event);
cudaEventCreate(&stop_event);
cudaEventRecord(start_event);

	std::chrono::high_resolution_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	// std::cout << "do display" << std::endl;
    // sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    
	glLoadIdentity();
	// gluLookAt (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	// std::cout << dis[camera_index] << std::endl;
	// std::cout << theta[camera_index] << std::endl;
	// std::cout << phi[camera_index] << std::endl;

	// glRotatef(30.0, 0.0, 1.0, 0.0); // rotate theta degree around y axis
	// glRotatef(45.0, 0.0, 0.0, 1.0); // rotate phi degree around z axis
	// glTranslatef(0.0, 0.0, -1.0);


    glPushMatrix(); 

	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    
	glRotatef(-45, 0.0, 1.0, 0.0); // delete for paper demo
    glRotatef(-45, 1.0, 0.0, 0.0); // delete for paper demo
	glTranslatef(0.0, 0.0, 4.5); // delete for paper demo

	std::cout << "phi: " << phi[camera_index - 1] << std::endl;
	std::cout << "theta: " << theta[camera_index - 1] << std::endl;
	std::cout << "dis: " << dis[camera_index - 1] << std::endl;
	float x, y, z;
	tpd2xyz(theta[camera_index - 1], phi[camera_index - 1], dis[camera_index - 1], x, y, z);
	if (x < 0) {
		// glRotatef(180, x, y, z); // rotate theta degree around z axis
	}

	// glRotatef(phi[camera_index - 1], 0.0, 0.0, 1.0); // rotate theta degree around z axis
	// glRotatef(theta[camera_index - 1], 0.0, 1.0, 0.0); // rotate phi degree around y axis
	// glTranslatef(0.0, 0.0, dis[camera_index - 1] + 0.7);

    // glLoadIdentity();
    // glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    // glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
	std::cout << "-viewTranslation.z: " << -viewTranslation.z << std::endl;
    // glTranslatef(1.0, -viewTranslation.y, -viewTranslation.z);
    // glTranslatef(0.5, -viewTranslation.y, -viewTranslation.z);
    // glTranslatef(1.5/76.0 + 1.0/76.0, -viewTranslation.y, -viewTranslation.z);
    // glTranslatef(1.5/77.0, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];


    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    glutSwapBuffers();
    glutReportErrors();

    // sdkStopTimer(&timer);
	// printf("````````````````````%3.9f\n", (sdkGetAverageTimerValue(&timer) / 1000.f));
    computeFPS();

	t2 = std::chrono::high_resolution_clock::now();

cudaEventRecord(stop_event);
cudaEventSynchronize(stop_event);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start_event, stop_event);
// std::cout << "Cuda event time: " << milliseconds/1000 << std::endl;

	render_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/(float)1000000; // seconds
	// std::cout << "Rendering time: " << render_time << " seconds" << std::endl;

	// wait for finish of prefetching
	// while (doingPrefetching){}

	doingRendering = false;

	// save screenShot
    std::string imageFileName = imagePath + std::to_string(camera_index - 1) + ".tga";
	save_screenshot(imageFileName, width, height);

}

void idle() {
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key)
    {
        case 27:
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
				exit_by_user = true;
                return;
            #endif
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;

        case '+':
            // density += 0.01f;
            density += 0.005f;
            break;

        case '-':
            // density -= 0.01f;
            density -= 0.005f;
            break;

        case ']':
            brightness += 0.1f;
            break;

        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;

        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;

        case ',':
            transferScale -= 0.01f;
            break;

        default:
            break;
    }

    printf("density = %.3f, brightness = %.3f, transferOffset = %.3f, transferScale = %.3f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y) {
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h) {
    width = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	// gluPerspective(30.0, 1.0, 1.0, 40.0);

}

void cleanup() {
	std::cout << "cleanup function called" << std::endl;
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());

	free(cornerBlockMapArray);
	free(blockCheck);
}

void initGL(int *argc, char **argv) {
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions are missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer() {
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#if 0
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
#endif

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL) {
    int result = 0;

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, argv);
    }
    else
    {
        result = findCudaDevice(argc, argv);
    }

    return result;
}

int getCornerIndex(int corner_x, int corner_y, int corner_z) {
	return corner_x + corner_y*17 + corner_z*17*17;
}

std::vector<std::vector<int>> getCorners(int blockIndex, int block_num, int unit) {
	int z = blockIndex/(block_num*block_num);
	int remain = blockIndex%(block_num*block_num);
	int y = remain/block_num;
	int x = remain%block_num;
	
	int x_offset = x*unit;
	int y_offset = y*unit;
	int z_offset = z*unit;

	std::vector<std::vector<int>> corners;
	std::vector<int> corner;

	corner.push_back(0 + x_offset);
	corner.push_back(0 + y_offset);
	corner.push_back(0 + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(unit + x_offset);
	corner.push_back(0 + y_offset);
	corner.push_back(0 + z_offset);
	corners.push_back(corner);
	corner.clear();


	corner.push_back(0 + x_offset);
	corner.push_back(unit + y_offset);
	corner.push_back(0 + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(unit + x_offset);
	corner.push_back(unit + y_offset);
	corner.push_back(0 + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(0 + x_offset);
	corner.push_back(0 + y_offset);
	corner.push_back(unit + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(unit + x_offset);
	corner.push_back(0 + y_offset);
	corner.push_back(unit + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(0 + x_offset);
	corner.push_back(unit + y_offset);
	corner.push_back(unit + z_offset);
	corners.push_back(corner);
	corner.clear();

	corner.push_back(unit + x_offset);
	corner.push_back(unit + y_offset);
	corner.push_back(unit + z_offset);
	corners.push_back(corner);
	corner.clear();

	return corners;
}

bool isInside(int blockIndex, int block_num, int unit, int corner_x, int corner_y, int corner_z) {
	int z = blockIndex/(block_num*block_num);
	int remain = blockIndex%(block_num*block_num);
	int y = remain/block_num;
	int x = remain%block_num;
	
	int x_lower = x*unit;
	int y_lower = y*unit;
	int z_lower = z*unit;
	
	int x_upper = x_lower + unit;
	int y_upper = y_lower + unit;
	int z_upper = z_lower + unit;

	if ((corner_x >= x_lower && corner_x <= x_upper)&&(corner_y >= y_lower && corner_y <= y_upper)&&(corner_z >= z_lower && corner_z <= z_upper)) {
		return true;
	} else {
		return false;
	}
}

void getCoordinates(int corner, int &x, int &y, int &z) {
	z = corner/(17*17); // 1/(17*17) integer portion
	int remain = corner%(17*17); // 1%(17*17) remainding portion
	y = remain/(17); 
	x = remain%(17);
}


std::vector<int> getConnectBlocks(int corner, int level) {
	int size = pow(2, (level + 1)); // if level = 0, then size = 2  and offset = 0            and totalBlockNum = 8    and unit = 8
									// if level = 1, then size = 4  and offset = 8            and totalBlockNum = 64   and unit = 4
									// if level = 2, then size = 8  and offset = 64 + 8       and totalBlockNum = 512  and unit = 2
									// if level = 3, then size = 18 and offset = 512 + 64 + 8 and totalBlockNum = 4096 and unit = 1
	int offset = 0;
	if (level == 0 ) { offset = 0; }
	if (level == 1 ) { offset = 8; }
	if (level == 2 ) { offset = 64 + 8; }
	if (level == 3 ) { offset = 512 + 64 + 8; }
	int totalBlockNum = size*size*size;
	int unit = 16/size;
	std::vector<int> connectBlocks;

	int x, y, z;
	getCoordinates(corner, x, y, z);
	for (int i = 0; i < totalBlockNum; i++) {
		if (isInside(i, size, unit, x, y, z)) {
			connectBlocks.push_back(i + offset);
		}
	}
	return connectBlocks;
}


std::map<int, std::vector<std::vector<int>>> createCornerBlockMap(int volume_size_4, int block_size) {
	int block_num = (volume_size_4 - 1)/(block_size - 2) + 1; // 16 + 1
	int cover_num = block_num*block_num*block_num;
	int levels = 4;

	std::map<int, std::vector<std::vector<int>>> currentCornerBlockMap;
	for (int i = 0; i < cover_num; i++) {
		std::cout << i << std::endl;
		std::vector<std::vector<int>> connectBlocksAll;
		for (int j = 0; j < levels; j++) {
			std::vector<int> connectBlocks = getConnectBlocks(i, j);
			connectBlocksAll.push_back(connectBlocks);
		}
        currentCornerBlockMap[i] = connectBlocksAll;
	}
	return currentCornerBlockMap;
}

std::map<int, std::vector<std::vector<float>>> getRange(std::string ellipses_path, int test_num, int points_num, int factor) {
    // load ellipses
    int total_ellipses = test_num * points_num;
    float *d = NULL;
    d = new float[total_ellipses];
    float *theta = NULL;
    theta = new float[total_ellipses];
    float *phi = NULL; 
    phi = new float[total_ellipses];
     
    std::ifstream infile;
    infile.open(ellipses_path);
    assert(infile);
    std::string line;
    std::getline(infile, line);
    int ctr = 0;
    while (std::getline(infile, line))
    {
        // delete newline character
        line.pop_back();
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        int cur_index;
        float d_cur;
        float theta_cur;
        float phi_cur;

        ss >> cur_index;
        ss >> d_cur;
        ss >> theta_cur;
        ss >> phi_cur;

        d[ctr] = d_cur;
        theta[ctr] = theta_cur;
        phi[ctr] = phi_cur;

        ctr++;
    }
    infile.close();
    
    std::map<int, std::vector<std::vector<float>>> ellipses_table;

    for (int i = 0; i < test_num; i++) {
        std::vector<std::vector<float>> cur_point;
        for (int j = 0; j < points_num; j++) {
            std::vector<float> cur_sample;
            int index = i*points_num + j;    
            cur_sample.push_back(d[index]);
            cur_sample.push_back(theta[index]);
            cur_sample.push_back(phi[index]);
            
            cur_point.push_back(cur_sample);
        }
        ellipses_table[i] = cur_point;
    }
    std::map<int, std::vector<std::vector<float>>> ellipses_table_factor;

    for (int i = 0; i < test_num; i++) {
        std::vector<std::vector<float>> cur_point;
        // std::cout << i << std::endl;
        for (int j = 0; j < points_num/factor; j++) {
            // std::cout << j << std::endl;
            int index = j*factor;    
            cur_point.push_back(ellipses_table[i].at(index));
        }
        ellipses_table_factor[i] = cur_point;
    }
    /*   
    for (int i = 0; i < test_num; i++) {
        for (int j = 0; j < points_num/factor; j++) {
            std::cout << i << "---" << ellipses_table_factor[i].at(j).at(0) << "," << ellipses_table_factor[i].at(j).at(1) << ", "  << ellipses_table_factor[i].at(j).at(2) << std::endl;
        }
    }
    */
    
 
    return ellipses_table_factor;
    
}

void updateSeq(float x, float y, float z) {
	seq[0][0] = seq[0][1];
	seq[0][1] = seq[0][2];
	seq[0][2][0] = x;
	seq[0][2][1] = y;
	seq[0][2][2] = z;
}

void queryLstm(float &x_next, float &y_next, float &z_next) {
	std::vector<torch::jit::IValue> input;
	input.push_back(seq);
	auto out_lstm = net_lstm.forward(input).toTensor();
	x_next = out_lstm[0][0].item<float>();
	y_next = out_lstm[0][1].item<float>();
	z_next = out_lstm[0][2].item<float>();
}

void queryMdn(float theta_lstm, float phi_lstm,
			  float &mu_theta, float &mu_phi,
			  float &sigma_theta, float &sigma_phi,
			  float &correlation) {
	torch::Tensor angle = torch::tensor({0.0, 0.0});
	angle[0] = theta_lstm/180.0;
	angle[1] = phi_lstm/360.0;
	std::vector<torch::jit::IValue> input;
	input.push_back(angle);
	auto out_mdn = net_mdn.forward(input);
	float max_pi_value = -1.0;
	int max_pi_index;
	for (int i = 0; i < 5; i++) {
		float pi_value = out_mdn.toTuple()->elements()[0].toTensor()[i].item<float>();
		if (max_pi_value < pi_value) {
			max_pi_value = pi_value;
			max_pi_index = i;
		}
	}
	mu_theta = out_mdn.toTuple()->elements()[1].toTensor()[max_pi_index].item<float>();
	mu_phi = out_mdn.toTuple()->elements()[2].toTensor()[max_pi_index].item<float>();
	sigma_theta = out_mdn.toTuple()->elements()[3].toTensor()[max_pi_index].item<float>();
	sigma_phi = out_mdn.toTuple()->elements()[4].toTensor()[max_pi_index].item<float>();
	correlation = out_mdn.toTuple()->elements()[5].toTensor()[max_pi_index].item<float>();
}


void getEllipsePoints(float mu_theta, float mu_phi, float sigma_theta, float sigma_phi, float correlation,
				 int points_num,
				 std::vector<float> &thetas, std::vector<float> &phis) {
	float a = sigma_theta*sigma_theta;
    float c = sigma_phi*sigma_phi;
    float b = correlation*sigma_theta*sigma_phi;
	float lamda_1 = (a + c)/2 + sqrt(((a - c)/2)*((a - c)/2) + b*b);
    float lamda_2 = (a + c)/2 - sqrt(((a - c)/2)*((a - c)/2) + b*b);
    float angle = atan2(b, lamda_1 - a);
	float angle_unit = M_PI*2/points_num;
	for (int i = 0; i < points_num; i++) {
        float cur_angle = i*angle_unit;

        float x = sqrt(lamda_1)*cos(angle)*cos(cur_angle) -
                  sqrt(lamda_2)*sin(angle)*sin(cur_angle);
        float y = sqrt(lamda_1)*sin(angle)*cos(cur_angle) +
                  sqrt(lamda_2)*cos(angle)*sin(cur_angle);

        x = x + mu_theta;
        y = y + mu_phi;
			
		thetas.push_back(x);
		phis.push_back(y);
	}
}

void timerCall(int) {
	
	if (camera_index == test_size) {
		#if defined (__APPLE__) || defined(MACOSX)
            exit(EXIT_SUCCESS);
        #else
            glutDestroyWindow(glutGetWindow());
			exit_by_user = true;
            return;
       #endif
	}

	glutPostRedisplay();

	std::cout << "clear filesystem cache" << std::endl;
	sync();
	std::ofstream ofs("/proc/sys/vm/drop_caches");
	ofs << "1" << std::endl;



	float find_visible_blocks_time, cache_time, info_time, cudainit_time, cache_total_time;
	std::chrono::high_resolution_clock::time_point t1, t2, t_start, t_end;
	std::cout << "---------------------------------------" << camera_index << "------------------------------------" << std::endl;

	// camera_index = 39;

	float x, y, z;
	tpd2xyz(theta[camera_index], phi[camera_index], dis[camera_index], x, y, z);
	std::cout << "theta, phi, d: " << theta[camera_index] << " " << phi[camera_index] << " " << dis[camera_index] << std::endl;
	std::cout << "x, y, z: " << x << " " << y << " " << z << std::endl;
	updateSeq(x, y, z);
		

	t1 = std::chrono::high_resolution_clock::now();
	// visible_blocks.clear();
	// t_start = std::chrono::high_resolution_clock::now();
	// getCurrentVisibleBlocks(theta[camera_index], phi[camera_index], dis[camera_index], VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4, BLOCK_SIZE, 30, visible_blocks);
	// t_end = std::chrono::high_resolution_clock::now();
	// find_visible_blocks_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;
	// std::cout << "=======timecall find vb time Old: " << find_visible_blocks_time << std::endl;
	
	visible_blocks.clear();
	t_start = std::chrono::high_resolution_clock::now();
	// getCurrentVisibleBlocks(theta[camera_index], phi[camera_index], dis[camera_index], VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4, BLOCK_SIZE, 30, visible_blocks);
	getCurrentVisibleBlocksFast(theta[camera_index], phi[camera_index], dis[camera_index], VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4, BLOCK_SIZE, 30, visible_blocks, cornerBlockMapArray, blockCheck);
	t_end = std::chrono::high_resolution_clock::now();
	find_visible_blocks_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;
	// std::cout << "=======timecall find vb time New: " << find_visible_blocks_time << std::endl;

	// std::cout << "vb number: " << visible_blocks.size() << std::endl;
	// testing starts //
	visible_blocks.clear();
	// visible_blocks.push_back(0);
	// visible_blocks.push_back(1);
	// visible_blocks.push_back(2);
	// visible_blocks.push_back(3);
	// visible_blocks.push_back(4);
	// visible_blocks.push_back(5);
	for (int i = 0; i < 8; i++) {
	 	visible_blocks.push_back(i);
	}
	// testing ends // 
	

	/*
	std::ofstream outf_vb;
	outf_vb.open("vbs.txt", std::ios_base::app);
	for (int i = 0; i < visible_blocks.size(); i++) {
		if (i == visible_blocks.size() - 1) {
			outf_vb << visible_blocks[i];
		} else {
			outf_vb << visible_blocks[i] << ",";
		}
	}
	outf_vb << "\n";
	outf_vb.close();
	*/


	int hit = 0;
	int miss = 0;
	t_start = std::chrono::high_resolution_clock::now();
	hit_blocks.clear();
	miss_blocks.clear();
	for (int j = 0; j < visible_blocks.size(); j++) {
		// std::cout << "vb: " << j << std::endl;
		int hit_miss = updateCacheAndCacheData(cache, cache_mem, cache_data, visible_blocks.at(j), microBlockSize, "lru", volumeFilePath, camera_index);
		hit_miss?hit++:miss++;
		if (hit_miss) {
			hit_blocks.push_back(visible_blocks.at(j));
		} else {
			miss_blocks.push_back(visible_blocks.at(j));
		}
	}
	// std::cout << "miss: " << miss << std::endl;
	// std::cout << "hit: " << hit << std::endl;
/*
	std::cout << "v blocks:" << std::endl;	
	for (int j = 0; j < visible_blocks.size(); j++) { std::cout << visible_blocks.at(j) << " ";}
	std::cout << std::endl;
	std::cout << "cache:" << std::endl;	
	for (int j = 0; j < cache.size(); j++) { std::cout << cache.at(j) << " ";}
	std::cout << std::endl;
	std::cout << "cache_mem:" << std::endl;	
	for (int j = 0; j < cache_mem.size(); j++) { std::cout << cache_mem.at(j) << " ";}
	std::cout << std::endl;
*/

	t_end = std::chrono::high_resolution_clock::now();
	cache_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;


	/* save visible blocks and missing blocks */
	/*
	std::ofstream outf_vb;
	outf_vb.open("vbs.txt", std::ios_base::app);
	for (int i = 0; i < visible_blocks.size(); i++) {
		if (i == visible_blocks.size() - 1) {
			outf_vb << visible_blocks.at(i);
		} else {
			outf_vb << visible_blocks.at(i) << ",";
		}
	}
	outf_vb << "\n";
	outf_vb.close();
	
	outf_vb.open("hbs.txt", std::ios_base::app);
	for (int i = 0; i < hit_blocks.size(); i++) {
		if (i == hit_blocks.size() - 1) {
			outf_vb << hit_blocks.at(i);
		} else {
			outf_vb << hit_blocks.at(i) << ",";
		}
	}
	outf_vb << "\n";
	outf_vb.close();

	outf_vb.open("mbs.txt", std::ios_base::app);
	for (int i = 0; i < miss_blocks.size(); i++) {
		if (i == miss_blocks.size() - 1) {
			outf_vb << miss_blocks.at(i);
		} else {
			outf_vb << miss_blocks.at(i) << ",";
		}
	}
	outf_vb << "\n";
	outf_vb.close();
	*/



	t_start = std::chrono::high_resolution_clock::now();
	// All blocks label, visible blocks labels 1, others 0, will be sent to device.	
	int totalBlockNum = X_BLOCK_NUM_LEVEL_1*Y_BLOCK_NUM_LEVEL_1*Z_BLOCK_NUM_LEVEL_1 +
						X_BLOCK_NUM_LEVEL_2*Y_BLOCK_NUM_LEVEL_2*Z_BLOCK_NUM_LEVEL_2 +
						X_BLOCK_NUM_LEVEL_3*Y_BLOCK_NUM_LEVEL_3*Z_BLOCK_NUM_LEVEL_3 +
						X_BLOCK_NUM_LEVEL_4*Y_BLOCK_NUM_LEVEL_4*Z_BLOCK_NUM_LEVEL_4;
	int selectList[totalBlockNum];
	for (int j = 0; j < totalBlockNum; j++) {
		selectList[j] = 0;
	}
	for (int j = 0; j < visible_blocks.size(); j++) {
		selectList[visible_blocks.at(j)] = 1;
	}
	int visible_blocks_num = visible_blocks.size();
	int *selectList_d;
	cudaMalloc((int **)&selectList_d, totalBlockNum*sizeof(int));
	cudaMemcpy(selectList_d, selectList, totalBlockNum*sizeof(int), cudaMemcpyHostToDevice);

	int cache_mem_h[cache_mem.size()];
	for (int j = 0; j < cache_mem.size(); j++) {
		cache_mem_h[j] = cache_mem.at(j);
	}
	int *cache_mem_d;
	cudaMalloc((int **)&cache_mem_d, cache_mem.size()*sizeof(int));
	cudaMemcpy(cache_mem_d, cache_mem_h, cache_mem.size()*sizeof(int), cudaMemcpyHostToDevice);

	
	int visible_blocks_h[visible_blocks.size()];
	for (int j = 0; j < visible_blocks.size(); j++) {
		visible_blocks_h[j] = visible_blocks.at(j);
	}
	int *visible_blocks_d;
	cudaMalloc((int **)&visible_blocks_d, visible_blocks.size()*sizeof(int));
	cudaMemcpy(visible_blocks_d, visible_blocks_h, visible_blocks.size()*sizeof(int), cudaMemcpyHostToDevice);
	t_end = std::chrono::high_resolution_clock::now();
	info_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;

	
	cudaExtent volumeSizeAll = make_cudaExtent(BLOCK_SIZE_EACH_DIMENSION, BLOCK_SIZE_EACH_DIMENSION, BLOCK_SIZE_EACH_DIMENSION*cache_mem.size());
	size_x = volumeSizeAll.width;
	size_y = volumeSizeAll.height;
	size_z = volumeSizeAll.depth;
	t_start = std::chrono::high_resolution_clock::now();
	initCuda2(cache_data, volumeSizeAll, selectList_d, cache_mem_d, cache_mem.size(), visible_blocks_num);
	t_end = std::chrono::high_resolution_clock::now();
	cudainit_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;


	t2 = std::chrono::high_resolution_clock::now();
	cache_total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/(float)1000000;
	// std::cout << "Check vb time: " << cache_total_time << " seconds" << std::endl;

	std::ofstream myfile;
	myfile.open("benchmark.txt", std::ios_base::out | std::ios_base::app);
	if (camera_index == 0) {
		myfile << "cache_findVbTime" << " "
			   << "cache_cacheTime" << " "
	    	   << "miss" << " "
			   << "cacheSize" << " "
		       << "cache_cudaInitTime" << " "
		   	   << "cache_totalTime" << " "
		       << "prefetch_cacheTime" << " "
    		   << "render_time" << " "
			   << "early_stop" << " "
 			   << "vb_size" << " "
			   << "prefetch_findVbTime" << " "
			   << "prefetch_inferenceTime" << "\n";
	}
  	if (myfile.is_open())
  	{
    	myfile << find_visible_blocks_time << " ";
    	myfile << cache_time << " ";
		myfile << miss << " ";
		myfile << cache.size() << " ";
    	myfile << cudainit_time << " ";
    	myfile << cache_total_time << " ";
    	myfile << prefetch_cacheTime << " ";
		myfile << render_time << " ";
		myfile << early_stop << " ";
		myfile << visible_blocks.size() << " ";
    	myfile << prefetch_find_visible_blocks_time << " ";
    	myfile << inference_time << "\n";
    	myfile.close();
  	} else {
		std::cout << "Unable to open file" << std::endl;;
    	exit(EXIT_SUCCESS);
	}

	camera_index++;

	// glutTimerFunc(100, timerCall, 0);
	glutTimerFunc(100, timerCall, 0);
}

void *renderFunc(void *arg) {

	// This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    // glutIdleFunc(idle);
	glutTimerFunc(0, timerCall, 0);

    initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif
    glutMainLoop();

   pthread_exit(NULL);
}


// int main(int argc, char **argv) {
void *renderThreadFunc(void *threadArg) {
	struct threadData *thread_data;
	thread_data = (struct threadData *)threadArg;
	
	int argc = thread_data->argc;
	char **argv;
    argv = thread_data->argv;

    char *ref_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
        fpsLimit = frameCheckNumber;
    }

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    chooseCudaDevice(argc, (const char **)argv, true);

    // parse arguments
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "volume", &filename))
    {
		std::cout << "filename: " << filename << std::endl;
        volumeFilename = filename;
    }

    int n;

    if (checkCmdLineFlag(argc, (const char **) argv, "size"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "size");
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "xsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "xsize");
        volumeSize.width = n;
    }
    
	if (checkCmdLineFlag(argc, (const char **) argv, "ysize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "ysize");
        volumeSize.height = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "zsize"))
    {
        n= getCmdLineArgumentInt(argc, (const char **) argv, "zsize");
        volumeSize.depth = n;
    }



    sdkCreateTimer(&timer);

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");
	std::cout << "pid: " << getpid() << std::endl;
    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    // glutIdleFunc(idle);
	glutTimerFunc(0, timerCall, 0);

    initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif
    glutMainLoop();
}

void getRealTimePrediction(float &dis_center, float &theta_center, float &phi_center, std::vector<float> &thetas, std::vector<float> &phis) {
	/* make prediction on next camera location from LSTM*/
	float x_next, y_next, z_next;
	queryLstm(x_next, y_next, z_next);
	// std::cout << "x_next, y_next, z_next: " << x_next << " " << y_next << " " << z_next << std::endl;
	float theta_p, phi_p, dis_p; 
	xyz2tpd(x_next, y_next, z_next, theta_p, phi_p, dis_p);
	/* make prediction on next camera location and range from MDN*/
	float mu_theta, mu_phi, sigma_theta, sigma_phi, correlation;
	queryMdn(theta_p, phi_p, mu_theta, mu_phi, sigma_theta, sigma_phi, correlation);
	// std::cout << "mu_theta, mu_phi, sigma_theta, sigma_phi, correlation: " << mu_theta << " "
	// 																	   <<  mu_phi << " "
	// 																	   << sigma_theta << " " 
	// 																	   << sigma_phi << " "
	// 																	   << correlation << std::endl;
	mu_theta = mu_theta*180.0;
	mu_phi = mu_phi*360.0;
	sigma_theta = sigma_theta*180.0;
	sigma_phi = sigma_phi*360.0;

	dis_center = dis_p;
	theta_center = mu_theta;
	phi_center = mu_phi;

	// int points_num = 8;
	int points_num = 4;
	getEllipsePoints(mu_theta, mu_phi, sigma_theta, sigma_phi, correlation, points_num, thetas, phis);
	// for (int i = 0; i < 4; i++) {
	// 	std::cout << "thetas: " << thetas[i] << std::endl;
	// 	std::cout << "phis: " << phis[i] << std::endl;
	// }
}

std::map<int, std::vector<std::vector<int>>> getConerBlockMap(std::string fileName) {
	std::map<int, std::vector<std::vector<int>>> cbm;
	std::ifstream infile(fileName);
	std::string line;
	std::vector<std::vector<int>> blocksAllLevels;
	std::vector<int> blocks;
	while (std::getline(infile, line))
	{
    	std::istringstream iss(line);
    	int corner, level, block;
		iss >> corner;
		iss >> level;
		// std::cout << corner << " " << level << "--" << std::endl;
		while (iss >> block) {
			blocks.push_back(block);
		}
		blocksAllLevels.push_back(blocks);
		blocks.clear();
		// std::cout << blocksAllLevels << std::endl;
		// std::cout << "count: " << count << std::endl;
    	// if (!(iss >> a >> b)) { break; } // error
		// std::cout << a << "    " << b << std::endl;
    	// process pair (a,b)
		// std::cout << "level: " << level << std::endl;
		if (level == 3) {
			// std::cout << "corner: " << corner << std::endl;
			cbm[corner] = blocksAllLevels;
			blocksAllLevels.clear();
		}
	}
	return cbm;
}

void getConerBlockMap2Array(std::string fileName, int* array) {
	std::map<int, std::vector<std::vector<int>>> cbm;
	std::ifstream infile(fileName);
	std::string line;
	std::vector<std::vector<int>> blocksAllLevels;
	std::vector<int> blocks;
	while (std::getline(infile, line))
	{
    	std::istringstream iss(line);
    	int corner, level, block;
		iss >> corner;
		iss >> level;
		// std::cout << corner << " " << level << "--" << std::endl;
		while (iss >> block) {
			blocks.push_back(block);
		}
		int begin = corner*9*4 + level*9;
		array[begin] = blocks.size();
		for (int i = 0; i < blocks.size(); i++) {
			array[begin + i + 1] = blocks.at(i);
		}
		blocks.clear();
	}
}

void *prefetchThreadFunc(void *threadArg) {
	while (!exit_by_user) {
		// Block here and wait for rendering starts
		while (!doingRendering) {}
		if (prefetchOn) {
#if 1 // prefetch on/off 
			doingPrefetching = true;
			early_stop = 0;
			// Find visible blocks
			std::chrono::high_resolution_clock::time_point t_start, t_end;
			std::chrono::high_resolution_clock::time_point t_s, t_e;
			visible_blocks_predict.clear();
			t_start = std::chrono::high_resolution_clock::now();
			// std::cout << "in prefetch process: camera index: " << camera_index << std::endl;
			if (method_used == "appa") {
				// getCurrentVisibleBlocksLUT(theta_predict[camera_index], phi_predict[camera_index], dis_predict[camera_index],
				// 						visible_blocks_predict);
				float delta_d = 0.1;
			   	float delta_theta = 6.0;
	    		float delta_phi = 6.0;
				char view_pos[50]; 
			    float norm_dis, norm_theta, norm_phi;

				// std::cout  << "camera:::: " << dis[camera_index - 1] << std::endl;
				// std::cout  << "camera:::: " << theta[camera_index - 1] << std::endl;
				// std::cout  << "camera:::: " << phi[camera_index - 1] << std::endl;


			    double intpart, fractpart;
			    fractpart = modf(double(dis[camera_index - 1]/delta_d) , &intpart);
			    norm_dis = float(intpart)*delta_d;
				fractpart = modf(double(theta[camera_index - 1]/delta_theta) , &intpart);
			    norm_theta = float(intpart)*delta_theta;
			    fractpart = modf(double(phi[camera_index - 1]/delta_phi) , &intpart);
			    norm_phi = float(intpart)*delta_phi;
			    sprintf(view_pos, "%4.2f%5.1f%5.1f", norm_dis, norm_theta, norm_phi);
				// printf("view_pos::::::: %s\n", view_pos);
				std::map<std::string, std::vector<int> >::iterator sample;
			    sample = all_map.find(view_pos);
				if (sample == all_map.end()) { std::cout << "Not find in table" << std::endl; exit(EXIT_SUCCESS); }
				// std::cout << "size::::::: " << sample->second.size() << std::endl;
				visible_blocks_predict = sample->second;
			}
			if (method_used == "markov" || method_used == "lstm") {	
				/*
				getCurrentVisibleBlocks(theta_predict[camera_index], phi_predict[camera_index], dis_predict[camera_index],
										VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4,
										BLOCK_SIZE,
										30,
										visible_blocks_predict);
				*/
				getCurrentVisibleBlocksFast(theta_predict[camera_index], phi_predict[camera_index], dis_predict[camera_index],
										VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4,
										BLOCK_SIZE,
										30,
										visible_blocks_predict,
										cornerBlockMapArray,
										blockCheck);
			}
			if (method_used == "rmdn") { // for RmdnCache
				if (camera_index >= 3 && camera_index <= 399) { // Skip the first 2 and the last camera locations
					getCurrentVisibleBlocksRange(theta_predict[camera_index], phi_predict[camera_index], dis_predict[camera_index],
												 camera_index,
												 VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4,
												 BLOCK_SIZE,
												 30,
												 visible_blocks_predict,
												 range,
												 cornerBlockMapArray,
												 blockCheck);
					/*
					std::ofstream outf;
					outf.open("prefetchedVbs.txt", std::ios_base::app);
					for (int i = 0; i < visible_blocks_predict.size(); i++) {
						outf << visible_blocks_predict[i] << " ";
					}
					outf << "\n";
					outf.close();
					*/
				}
			}
			if (method_used == "rmdn-realtime") { // for RmdnCache with real-time inference
				if (camera_index >= 3 && camera_index <= 399) { // Skip the first 2 and the last camera locations
					float dis_center, theta_center, phi_center;
				   	std::vector<float> thetas, phis;	
					t_s = std::chrono::high_resolution_clock::now();
					getRealTimePrediction(dis_center, theta_center, phi_center, thetas, phis);
					t_e = std::chrono::high_resolution_clock::now();
					getCurrentVisibleBlocksRangeRealtime(theta_center, phi_center, dis_center,
												 		 VOLUME_SIZE_1, VOLUME_SIZE_2, VOLUME_SIZE_3, VOLUME_SIZE_4,
												 		 BLOCK_SIZE,
												 		 30,
												 		 visible_blocks_predict,
												 		 thetas, phis,
												 		 cornerBlockMapArray,
												 		 blockCheck);
					/*
					std::ofstream outf;
					outf.open("prefetchedVbsRealtime.txt", std::ios_base::app);
					for (int i = 0; i < visible_blocks_predict.size(); i++) {
						outf << visible_blocks_predict[i] << " ";
					}
					outf << "\n";
					outf.close();
					*/
				}
			}

			// std::cout << "vb size::::::: " << visible_blocks_predict.size() << std::endl;
			t_end = std::chrono::high_resolution_clock::now();
			prefetch_find_visible_blocks_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;
			inference_time = std::chrono::duration_cast<std::chrono::microseconds>(t_e - t_s).count()/(float)1000000;
		
			/*
			std::ofstream outf;
			outf.open("findVbtime.txt", std::ios_base::app);
			outf << prefetch_find_visible_blocks_time << "\n";
			outf.close();

			outf.open("inferencetime.txt", std::ios_base::app);
			outf << inference_time << "\n";
			outf.close();
			*/
	
			t_start = std::chrono::high_resolution_clock::now();
			int hit = 0;
			int miss = 0;
			for (int i = 0; i < visible_blocks_predict.size(); i++) {
				if (doingRendering) {
					int hit_miss = updateCacheAndCacheData(cache, cache_mem, cache_data, visible_blocks_predict.at(i), microBlockSize, "lru", volumeFilePath, camera_index);
					hit_miss?hit++:miss++;
				} else {
					// std::cout << "Prefetch Early Stops!" << std::endl;
					early_stop = 1;
					break;
				}
			}
			t_end = std::chrono::high_resolution_clock::now();
			prefetch_cacheTime = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000 + prefetch_find_visible_blocks_time;
			
			/*
			outf.open("prefetchtime.txt", std::ios_base::app);
			outf << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000 << "\n";
			outf.close();
			*/

			// std::cout << "Done prefetching with " << hit << " hit; " << miss << " miss." << std::endl;
			doingRendering = false;
			doingPrefetching = false;
		}
#endif
	}
	std::cout << "prefecth thread close" << std::endl;
	pthread_exit(NULL);
}

int main(int argc, char **argv) {
#if 1
	#if 0 // used for generating the corner block map, only need to run once and save the map to hard drive for later loading
	/* generate corner block map */
	cornerBlockMap = createCornerBlockMap(VOLUME_SIZE_4, BLOCK_SIZE);
	std::ofstream outff;
	outff.open("cornerBlockMap.txt");
	for (int i = 0; i < cornerBlockMap.size(); i++) {
		for (int j = 0; j < cornerBlockMap[i].size(); j++) {
				outff << i << " " << j << " ";
			for (int k = 0; k < cornerBlockMap[i].at(j).size(); k++) {
				outff << cornerBlockMap[i].at(j).at(k) << " ";
			}
			outff << "\n";
		}
	}
	outff.close();
	#endif
	/* for testing correctness of cornerblockmap
	cornerBlockMap = getConerBlockMap("cornerBlockMap.txt");
	std::ofstream outff;
	outff.open("map.txt");
	for (int i = 0; i < cornerBlockMap.size(); i++) {
		for (int j = 0; j < cornerBlockMap[i].size(); j++) {
			for (int k = 0; k < cornerBlockMap[i].at(j).size(); k++) {
				if ((j == cornerBlockMap[i].size() - 1) && (k == cornerBlockMap[i].at(j).size() - 1)) {
					outff << cornerBlockMap[i].at(j).at(k);
				} else {
					outff << cornerBlockMap[i].at(j).at(k) << ",";
				}
			}
		}
		outff << "\n";
	}
	outff.close();
	*/
	/* read in the corner map */
	cornerBlockMap = getConerBlockMap("cornerBlockMap.txt");
	cornerBlockMapArray = (int*)malloc(sizeof(int)*17*17*17*4*9);
	memset(cornerBlockMapArray, 0, sizeof(int)*17*17*17*4*9);
	getConerBlockMap2Array("cornerBlockMap.txt", cornerBlockMapArray);
	blockCheck = (int*)malloc(sizeof(int)*(8 + 64 + 512 + 4096));
	memset(blockCheck, 0, sizeof(int)*(8 + 64 + 512 + 4096));


	sampleDistance = atof(argv[1]);
	std::string method(argv[2]);
	method_used = method;
    std::cout << "Method: " << method << std::endl;
	/* Read pretrained models */
	net_lstm = torch::jit::load("notebook/traced_lstm.pt");
	net_mdn = torch::jit::load("notebook/traced_mdn.pt");

	// Read real camera trajectory
	std::ifstream infile;
	// std::string camera_trajectory = "../performance/data/test/test_1.dat";
	std::string camera_trajectory = "../performance/notebook/start.dat";
	// std::string camera_trajectory = "../performance/notebook/mytest2.dat";
  	infile.open(camera_trajectory, std::ios::binary);
	assert(infile);
	std::cerr << "Open " << camera_trajectory << std::endl;
	dis = new float[test_size];
	theta = new float[test_size];
	phi = new float[test_size];
    // data is stored in Spherical Coordinate System
	infile.read(reinterpret_cast<char *>(dis), sizeof(float)*test_size);
	infile.read(reinterpret_cast<char *>(theta), sizeof(float)*test_size);
	infile.read(reinterpret_cast<char *>(phi), sizeof(float)*test_size);
	infile.close();
	if (method_used == "lru") {
		std::cout << "Using LRU, no prefetch" << std::endl;
	} else {
		prefetchOn = true;
		if (method_used == "appa") {
			// Load sample maps
			infile.open("../tools/lookUpTable_0.1-6.0-6.0.dat");
			assert(infile);
			int map_size;
			infile.read(reinterpret_cast<char *>(&map_size), sizeof(int));
			printf("map_size = %d\n",map_size );
			std::vector<int> items;
			
			char temp[4]; // debug

			for (int i = 0; i < map_size; ++i)
			{
				char index[50];
				infile.read(reinterpret_cast<char *>(index), sizeof(char)*50);
				// printf("index = %s\n", index); // debug
				if (strncasecmp (temp, index, 4) != 0) { // debug
					strncpy(temp, index, 4); // debug
					printf("index = %s\n", index); // debug
				} // debug
				// printf("index = %s\n", index); // debug
				  
				int item_size;
				infile.read(reinterpret_cast<char *>(&item_size), sizeof(int));
				//printf("item_size = %d\n", item_size);
				// printf("items = ");
				//
				
				if (strncasecmp ("1.00 12.0 36.0", index, 14) == 0) {
					std::cout << "item_size: " << item_size << std::endl;
				}

				for (int j = 0; j < item_size; ++j)
				{   
					int elem;
					infile.read(reinterpret_cast<char *>(&elem), sizeof(int));
					// printf("%d ", elem);
					items.push_back(elem);  
				}
				// printf("\n");
				all_map.insert(std::pair<std::string, std::vector<int> >(index, items));
				items.clear();
			}
			infile.close();
		} else if (method_used == "rmdn-realtime") {
			// do nothing
		} else { // methods make trajectory prediction
			// Read predicted camera trajectory
			// Markov Method
			if (method_used == "markov") {
				std::cout << "Using ForeCache (Markov), do prefetch" << std::endl;
				camera_trajectory_predict = "../performance/data/markov_prediction/markov_predict1.dat"; // ForeCache
			}
			// LSTM Method
			if (method_used == "lstm") {
				std::cout << "Using LSTM, do prefetch" << std::endl;
				camera_trajectory_predict = "../performance/data/V2_js_predict/predict_test1.dat"; // LSTM
			}
			// Rmdn Method
			if (method_used == "rmdn") {
				std::cout << "Using rmdnCache (LSTM + MDN), do prefetch" << std::endl;
				camera_trajectory_predict = "../performance/data/mdn_predict/predict_test1.dat"; // Rmdn
				std::string ellipses = "../performance/data/mdn_predict/predict_range_test1_dtp.txt"; // LSTM
				range = getRange(ellipses, test_size, 8, 2);
			}

			infile.open(camera_trajectory_predict, std::ios::binary);
			assert(infile);
			std::cerr << "Open " << camera_trajectory << std::endl;
			dis_predict = new float[test_size];
			theta_predict = new float[test_size];
			phi_predict = new float[test_size];
			// data is stored in Spherical Coordinate System
			infile.read(reinterpret_cast<char *>(dis_predict), sizeof(float)*test_size);
			infile.read(reinterpret_cast<char *>(theta_predict), sizeof(float)*test_size);
			infile.read(reinterpret_cast<char *>(phi_predict), sizeof(float)*test_size);
			infile.close();
		}
	}
	struct threadData thread_data;
	thread_data.argc = argc;
	thread_data.argv = argv;
	pthread_t renderThread, prefetchThread;
	int rc;
  
 	rc = pthread_create(&renderThread, NULL, renderThreadFunc, (void *)&thread_data);
   	if (rc) {
	    std::cout << "Error:unable to create thread," << rc << std::endl;
    	exit(-1);
	}
	rc = pthread_create(&prefetchThread, NULL, prefetchThreadFunc, NULL);
   	if (rc) {
	    std::cout << "Error:unable to create thread," << rc << std::endl;
    	exit(-1);
	}
   	pthread_exit(NULL);
#endif
	return 0;
}
