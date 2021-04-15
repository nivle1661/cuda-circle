#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <sys/time.h>
#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelGetCirBinsCount -- (CUDA device code)
//
// Each thread counts how many bins there are in a corresponding circle.
__global__ void kernelGetCirBinsCount(uint* count, uint binPixLength) {
  int circleIdx = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (circleIdx >= cuConstRendererParams.numCircles)
    return;
    
  int circleIdx3 = circleIdx * 3;
  float3 cen = *(float3*) (&cuConstRendererParams.position[circleIdx3]);
  float rad = cuConstRendererParams.radius[circleIdx];
  
  //Get bounding box by pixel index
  short imgWidth	= cuConstRendererParams.imageWidth;
	short imgHeight = cuConstRendererParams.imageHeight;
 
  short minX = fminf(fmaxf((cen.x - rad) * imgWidth , 0), imgWidth-1);
	short maxX = fminf(fmaxf((cen.x + rad) * imgWidth , 0), imgWidth-1);
	short minY = fminf(fmaxf((cen.y - rad) * imgHeight, 0), imgHeight-1);
	short maxY = fminf(fmaxf((cen.y + rad) * imgHeight, 0), imgHeight-1);
 
  short xbinStart =  minX / binPixLength;
  short xbinEnd   = (maxX / binPixLength) + 1;
  short ybinStart =  minY / binPixLength;
  short ybinEnd   = (maxY / binPixLength) + 1;
  
  count[circleIdx] = static_cast<uint>((xbinEnd - xbinStart) * (ybinEnd - ybinStart));
}

// kernelGetBin_CirInd -- (CUDA device code)
//
// Each thread corresponds to a circle, but it modifies two arrays containing all bins,
// one for containing circle index and the other with its bin index on the image grid.
// Seems like repetitive logic, but we needed to know the length beforehand!
__global__ void kernelGetCirBinsPair(uint* devInputCirIdxStart, 
                                     uint* devOutputBinsCir_Bin,
                                     uint* devOutputBinsCir_Cir,
                                     uint binNumLength,
                                     uint binPixLength) {
  int circleIdx = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (circleIdx >= cuConstRendererParams.numCircles)
    return;
    
  int circleIdx3 = circleIdx * 3;
  float3 cen = *(float3*) (&cuConstRendererParams.position[circleIdx3]);
  float rad = cuConstRendererParams.radius[circleIdx];
  
  //Get bounding box by pixel index
  short imgWidth	= cuConstRendererParams.imageWidth;
	short imgHeight = cuConstRendererParams.imageHeight;
 
  short minX = fminf(fmaxf((cen.x-rad) * imgWidth, 0), imgWidth-1);
	short maxX = fminf(fmaxf((cen.x+rad) * imgWidth, 0), imgWidth-1);
	short minY = fminf(fmaxf((cen.y-rad) * imgHeight, 0), imgHeight-1);
	short maxY = fminf(fmaxf((cen.y+rad) * imgHeight, 0), imgHeight-1);
 
  short xbinStart =  minX / binPixLength;
  short xbinEnd   = (maxX / binPixLength) + 1;
  short ybinStart =  minY / binPixLength;
  short ybinEnd   = (maxY / binPixLength) + 1;
  
  uint ind = devInputCirIdxStart[circleIdx];
  
  //Row-major order!
  for (uint y = ybinStart; y < ybinEnd; y++) {
    uint binOffset = y * binNumLength;
    for (uint x = xbinStart; x < xbinEnd; x++) {
      devOutputBinsCir_Bin[ind] = binOffset + x;
      devOutputBinsCir_Cir[ind] = circleIdx;
      ind++;
    }
  } 
}

// kernelGetBinStartIndex -- (CUDA device code)
//
// For each bin, we want to know its starting index in out array. 
// We can set up shared memory across threads and check if the previous index
// is different from the current index.
__global__ void kernelGetBinStartIndex(uint* devOutputBinStartIndex,
                                       uint* devInputCirBins_Bin,
                                       uint binsCirLength) {
  __shared__ int cache[257]; //blockDim.x + 1
  
  int binsCirIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (binsCirIdx >= binsCirLength) 
    return;
    
  if (threadIdx.x == 0) {
    // Do most common case first for if-else
    if (binsCirIdx != 0) {
      cache[0] = devInputCirBins_Bin[binsCirIdx-1];
    } else {
      cache[0] = 0;
    }
  }
  cache[1+threadIdx.x] = devInputCirBins_Bin[binsCirIdx];
  
  // ------------------ //
  __syncthreads();      //
  // ------------------ //
  
  int index = cache[1+threadIdx.x];
  bool newBin = (index != cache[threadIdx.x]);
  if (newBin) {
    // printf("New bin at: %d, %u\n", index, binsCirIdx);
    devOutputBinStartIndex[index] = binsCirIdx;
  }
  
  // ------------------ //
  __syncthreads();      //
  // ------------------ //
  
  if (binsCirIdx == binsCirLength - 1) {
    newBin = true;
    binsCirIdx = (int) binsCirLength;
  }
  
  if (newBin) {
    int j = index;
    while (j > 0 && devOutputBinStartIndex[j] == 0) {
      devOutputBinStartIndex[j] = (uint) binsCirIdx;
      j--;
    }
  }
}


// kernelGetBinSizes -- (CUDA device code)
//
// Find the size of each bin (how many circles are inside), which is done with
// pairwise subtraction on the starting indices.
__global__ void kernelGetBinSizes(uint* devOutputBinSizes,
                                  uint* devInputBinStartIndex,
                                  uint binsTotal,
                                  uint binsCirLength) {
  __shared__ int cache[257]; //blockDim.x + 1
  
  int binsCirIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (binsCirIdx >= binsTotal) 
    return;
    
  if (threadIdx.x == blockDim.x - 1) {
    // Do most common case first for if-else
    if (binsCirIdx != binsTotal - 1) {
      cache[threadIdx.x+1] = devInputBinStartIndex[binsCirIdx+1];
    } 
  }
  
  if (binsCirIdx == binsTotal - 1) {
    cache[1+threadIdx.x] = binsCirLength;
  }
  cache[threadIdx.x] = devInputBinStartIndex[binsCirIdx];
  
  __syncthreads();      
  
  devOutputBinSizes[binsCirIdx] = cache[1+threadIdx.x] - cache[threadIdx.x];
}

// kernelRenderCirclesTRUE -- (CUDA device code)
//
// My implementation of rendering circles properly. We need each thread to
// operate on a grid of pixels since if each thread rendered a circle separately,
// there is no guarantee of order being preserved. 
// 
__global__ void kernelRenderCirclesTRUE(uint* devCirBins_Cir, 
                                        uint* devBinStartIndex,
                                        uint binNumLength, 
                                        uint binPixLength,
                                        uint* devBinNumCir, 
                                        uint maxBinNumCir, bool conditional,
                                        bool share) {
  //extern keyword since dynamically allocated!
  extern __shared__ float cache[];  
  float *shareCen = cache;
  float *shareRad = cache + maxBinNumCir * 3;
  float *shareCol = shareRad + maxBinNumCir;
  
  //Find bin from pixel coordinate
  short imageWidth = cuConstRendererParams.imageWidth;
  short imageHeight = cuConstRendererParams.imageHeight;
  short pixX = blockDim.x * blockIdx.x + threadIdx.x;
  short pixY = blockDim.y * blockIdx.y + threadIdx.y;
  
  short binX = pixX / binPixLength;
  short binY = pixY / binPixLength;
  short binInd = binY * binNumLength + binX;
  
  int binStart = devBinStartIndex[binInd];
  int binSize = devBinNumCir[binInd];
  short tCount = blockDim.x * blockDim.y;
  short threadId = threadIdx.y * blockDim.x + threadIdx.x;
  
  //Move radius and center to shared data
  if (share) 
    for (int i = threadId; i < binSize; i += tCount) {
      uint cirIndex = devCirBins_Cir[binStart + i];
      shareRad[i] = cuConstRendererParams.radius[cirIndex];
      *(float3*)(&shareCen[i * 3]) = *(float3*)(&cuConstRendererParams.position[cirIndex * 3]);
      *(float3*)(&shareCol[i * 3]) = *(float3*)(&cuConstRendererParams.color[cirIndex * 3]);
    }
  
  
  //We do this now after moving stuff to shared data. Otheriwse results are weird!
  if (pixX >= imageWidth || pixY >= imageHeight) {
    return;
  }
  
  __syncthreads();
  
  //Move lots of logic from shadePixel into here
  float4 *imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixY * imageWidth + pixX)]);
	float4 newColor = *imagePtr;
	float invWidth = 1.f / imageWidth;
  float invHeight = 1.f / imageHeight;
  float2 pcen = make_float2(invWidth * (static_cast<float>(pixX) + 0.5f), 
                            invHeight * (static_cast<float>(pixY) + 0.5f));
 
  for (int i = 0; i < binSize; i++) {
    uint cirIndex = devCirBins_Cir[binStart + i];
    
    float3 cen = (share) ? *(float3*) (&shareCen[i*3]) : 
                           *(float3*) (&cuConstRendererParams.position[cirIndex * 3]);
    float rad = (share) ? shareRad[i] : cuConstRendererParams.radius[cirIndex];
    float maxDist = rad * rad;
    
    float diffX	= pcen.x - cen.x;
		float diffY = pcen.y - cen.y;
		float pixelDist	= diffX * diffX + diffY * diffY;

    // Circle does not contribute to the image
    if (pixelDist > maxDist) {
      continue;
    }

    float3 rgb;
    float alpha;

    if (conditional) {
      const float kCircleMaxAlpha = .5f;
      const float falloffScale = 4.f;

      float normPixelDist = sqrt(pixelDist) / rad;
      rgb = lookupColor(normPixelDist);

      float maxAlpha = .6f + .4f * (1.f-cen.z);
      maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
      alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
      // Simple: each circle has an assigned color
      rgb = (share) ? *(float3*) (&shareCol[i * 3]) : 
                      *(float3*) (&cuConstRendererParams.color[cirIndex * 3]);
      alpha = .5f;
    }
    
    float oneMinusAlpha = 1.f - alpha;
		//draw into to the pixel shared buffer
		newColor.x = alpha*rgb.x + oneMinusAlpha * newColor.x;
		newColor.y = alpha*rgb.y + oneMinusAlpha * newColor.y;
		newColor.z = alpha*rgb.z + oneMinusAlpha * newColor.z;
		newColor.w = alpha + newColor.w;
  }
  *imagePtr = newColor;
}

// kerneldoesIntersectCircle -- (CUDA device code)
//
__global__ void kernelDoesIntersectCircle (int **dev_result, short **box, 
                                           int numCir, int numPix) {
                                           printf(":D");
  int indexX = blockDim.x * blockIdx.x + threadIdx.x;  //Circle idx
  int indexY = blockDim.y * blockIdx.y + blockIdx.y;   //Pixel idx
  
  if (indexX >= numCir || indexY >= numPix) 
    return;
  short imageWidth = cuConstRendererParams.imageWidth;
  int pixX = indexY % imageWidth;
  int pixY = indexY / imageWidth;
  
  short *bbox = box[indexX];
  if (pixX >= bbox[0] && pixX <= bbox[1] && pixY >= bbox[2] && pixY <= bbox[3]){
    float rad = cuConstRendererParams.radius[indexX];
    float3 p = *(float3*) (&cuConstRendererParams.position[indexX]);
    
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
  
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pcen = make_float2(invWidth * (static_cast<float>(pixX) + 0.5f), 
                              invHeight * (static_cast<float>(pixY) + 0.5f));
    float maxDist = rad * rad;
    
    float diffX	= pcen.x - p.x;
		float diffY = pcen.y - p.y;
		float pixelDist	= diffX * diffX + diffY * diffY;

    // Circle does not contribute to the image
    if (pixelDist <= maxDist) {
      dev_result[indexY][indexX] = 1;
      return;
    }
  }
  dev_result[indexY][indexX] = 0;
}

// kernelRenderCirclesBoxes-- (CUDA device code)
//
// Gets the bounding boxes for all circles
__global__ void kernelRenderCirclesBoxes (int numCir, short** box) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float3 cen = *(float3*) (&cuConstRendererParams.position[3*index]);
  float rad = cuConstRendererParams.radius[index];
  
  //Get bounding box by pixel index
  short imgWidth	= cuConstRendererParams.imageWidth;
	short imgHeight = cuConstRendererParams.imageHeight;
 
  box[index][0] = fminf(fmaxf((cen.x-rad) * imgWidth, 0), imgWidth-1);
	box[index][1] = fminf(fmaxf((cen.x+rad) * imgWidth, 0), imgWidth-1);
	box[index][2] = fminf(fmaxf((cen.y-rad) * imgHeight, 0), imgHeight-1);
	box[index][3] = fminf(fmaxf((cen.y+rad) * imgHeight, 0), imgHeight-1);
  printf("yo");
}


// kernelRenderCirclesMAYBE-- (CUDA device code)
//
// Really naive implementation that does a circle across multiple threads
// Meant to be sequentially called across all circles.
__global__ void kernelRenderCirclesMAYBE (int** pointInCircle, int numCir, 
                                          short** box, bool conditional) {
  printf("blah");
  dim3 blockDim(256, 1);
  dim3 gridDim((numCir + blockDim.x - 1) / blockDim.x);
  
  short pixX = blockDim.x * blockIdx.x + threadIdx.x;
  short pixY = blockDim.y * blockIdx.y + threadIdx.y;
  short imageWidth = cuConstRendererParams.imageWidth;
  short imageHeight = cuConstRendererParams.imageHeight;
  
  if (pixX >= imageWidth || pixY >= imageHeight) 
    return;
  int pixInd = pixY * imageWidth + pixX;
  int* isInCircle = pointInCircle[pixInd];
  
  float4 *imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixY * imageWidth + pixX)]);
	float4 newColor = *imagePtr;
  for (int i = 0; i < numCir; i++) {
    if (!isInCircle[i]) 
      continue;
    float rad = cuConstRendererParams.radius[i];
    float3 p = *(float3*) (&cuConstRendererParams.position[i*3]);
    float3 color = *(float3*) (&cuConstRendererParams.color[i*3]);
    
    float3 rgb;
    float alpha;

    if (conditional) {
      const float kCircleMaxAlpha = .5f;
      const float falloffScale = 4.f;

      float diffX = p.x - pixX;
      float diffY = p.y - pixY;
      float pixelDist = diffX * diffX + diffY * diffY;
    
      float normPixelDist = sqrt(pixelDist) / rad;
      rgb = lookupColor(normPixelDist);

      float maxAlpha = .6f + .4f * (1.f-p.z);
      maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
      alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
      // Simple: each circle has an assigned color
      rgb = color;
      alpha = .5f;
    }
    
    float oneMinusAlpha = 1.f - alpha;
		//draw into to the pixel shared buffer
		newColor.x = alpha*rgb.x + oneMinusAlpha * newColor.x;
		newColor.y = alpha*rgb.y + oneMinusAlpha * newColor.y;
		newColor.z = alpha*rgb.z + oneMinusAlpha * newColor.z;
		newColor.w = alpha + newColor.w;
  }
  *imagePtr = newColor;
}

// kernelRenderCirclesFALSE -- (CUDA device code)
//
// Really naive implementation that does a circle across multiple threads
// Meant to be sequentially called across all circles.
__global__ void kernelRenderCirclesFALSE (short minX, short minY, 
                                          float3 p, float rad, float3 color,
                                          bool conditional, int cirIndex) {
  short pixX = blockDim.x * blockIdx.x + threadIdx.x + minX;
  short pixY = blockDim.y * blockIdx.y + threadIdx.y + minY;
  short imageWidth = cuConstRendererParams.imageWidth;
  short imageHeight = cuConstRendererParams.imageHeight;
  
  float4 *imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixY * imageWidth + pixX)]);
	float4 newColor = *imagePtr;
  
  float invWidth = 1.f / imageWidth;
  float invHeight = 1.f / imageHeight;
  float2 pcen = make_float2(invWidth * (static_cast<float>(pixX) + 0.5f), 
                            invHeight * (static_cast<float>(pixY) + 0.5f));
    float maxDist = rad * rad;
    
    float diffX	= pcen.x - p.x;
		float diffY = pcen.y - p.y;
		float pixelDist	= diffX * diffX + diffY * diffY;

    // Circle does not contribute to the image
    if (pixelDist > maxDist) {
      return;
    }

    float3 rgb;
    float alpha;

    if (conditional) {
      const float kCircleMaxAlpha = .5f;
      const float falloffScale = 4.f;

      float normPixelDist = sqrt(pixelDist) / rad;
      rgb = lookupColor(normPixelDist);

      float maxAlpha = .6f + .4f * (1.f-p.z);
      maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
      alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
      // Simple: each circle has an assigned color
      rgb = color;
      alpha = .5f;
    }
    
    float oneMinusAlpha = 1.f - alpha;
		//draw into to the pixel shared buffer
		newColor.x = alpha*rgb.x + oneMinusAlpha * newColor.x;
		newColor.y = alpha*rgb.y + oneMinusAlpha * newColor.y;
		newColor.z = alpha*rgb.z + oneMinusAlpha * newColor.z;
		newColor.w = alpha + newColor.w;
  
  *imagePtr = newColor;
}
                                                                

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // For all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        
        cudaFree(devPointInCircle);
        cudaFree(devCirBinsCount);   
        cudaFree(devCirBinsIndex);   
        cudaFree(devBinStartIndex);  
        cudaFree(devBinNumCir);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce GTX 1080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 1080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    
    /** Setup code for new approach! */
    //We want more bins if there are more circles, but we also want them to be aligned 
    //with 16x16 grids when running renderCircles.
    //
    if (numCircles < 100) {
		  binNumLength = 4;
    } else if (numCircles >= 100 && numCircles < 1000) {
		  binNumLength = 8;
	  } else if (numCircles >= 1000 && numCircles < 10000) {
		  binNumLength = 16;
	  } else {
		  binNumLength = 64;
	  }
     
    uint notBinPixLength = (image->width - 1)/binNumLength + 1;  //standard
    binPixLength = max(1, (notBinPixLength / 16)) * 16;          //push to floor multiple of 16
    binNumLength = (image->width - 1)/binPixLength + 1;          //correct binNumLength 
     
    //allocate memory for cuda
    cudaMalloc(&devPointInCircle, sizeof(uint) * image->width * image->height);
    cudaMalloc(&devCirBinsCount, sizeof(uint) * numCircles);
    cudaMalloc(&devCirBinsIndex, sizeof(uint) * numCircles);
    cudaMalloc(&devBinStartIndex, sizeof(uint) * binNumLength * binNumLength);
    cudaMalloc(&devBinNumCir, sizeof(uint) * binNumLength * binNumLength);

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    // 256 threads per block is a healthy number
    bool conditional = (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
    
    
    /** Step 1: Get number of bins per circle
    */
    kernelGetCirBinsCount<<<gridDim, blockDim>>>(devCirBinsCount, binPixLength);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
    
    /** step 1.5: If we have reason to believe that the graph has really low density, 
                  then we switch approach. This is either from having a very small
                  amount of circles, or a sparse pattern. 
                  We can tell by checking how many circles are in each bin.
    */
    if (numCircles < 5) {
      for (int i = 0; i < numCircles; i++) {
        float3 p = *(float3*)(&position[i*3]);
        float  rad = radius[i];
        float3 col = *(float3*)(&color[i*3]);
      
        int imgWidth = image->width;
        int imgHeight = image->height;
        int minX = fminf(fmaxf((p.x-rad) * imgWidth, 0), imgWidth-1);
	      int maxX = fminf(fmaxf((p.x+rad) * imgWidth, 0), imgWidth-1);
	      int minY = fminf(fmaxf((p.y-rad) * imgHeight, 0), imgHeight-1);
	      int maxY = fminf(fmaxf((p.y+rad) * imgHeight, 0), imgHeight-1);
        printf("%d, %d | %d, %d\n", minX, maxX, minY, maxY);
         
        dim3 pixelBlockDim(16,16);
        dim3 pixelGridDim((maxX - minX) / pixelBlockDim.x + 1,
			    			          (maxY - minY) / pixelBlockDim.y + 1);
        kernelRenderCirclesFALSE<<<pixelGridDim, pixelBlockDim>>>(minX, minY, 
                                                                  p, rad,
                                                                  col, conditional, i);
        cudaDeviceSynchronize();                                                           
      }
      return;
    } /**else {
      int imgWidth = image->width;
      int imgHeight = image->height;
      dim3 pixelBlockDim(16,16);
      dim3 pixelGridDim((numCircles - 1) / pixelBlockDim.x + 1,
			   			          (imgWidth*imgHeight - 1) / pixelBlockDim.y + 1);
                                                                             
      short** box;
      cudaMalloc(&box, 4 * sizeof(short) * numCircles);
      kernelRenderCirclesBoxes<<<gridDim, blockDim>>>(numCircles, box);
      //once per bin-circle
      kernelDoesIntersectCircle<<<pixelGridDim, pixelBlockDim>>>(devPointInCircle, box,
                                                                 numCircles, imgWidth*imgHeight);
      //once per circle
      kernelRenderCirclesMAYBE<<<gridDim, blockDim>>> (devPointInCircle,
                                                       numCircles, 
                                                       box, conditional);
      cudaFree(box);
      return;
    }*/
    
    /** Step 2: Get starting index of the bins per circle  
                Done by using thrust::exclusive_scan (sorry!)
    */
    thrust::exclusive_scan(thrust::device_ptr<uint>(devCirBinsCount), 
                           thrust::device_ptr<uint>(devCirBinsCount + numCircles),
                           thrust::device_ptr<uint>(devCirBinsIndex));
                           
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);

    // Get how many bin-circle pairs there are
    uint lastCirBinsCount, lastCirBinsIndex;
    cudaMemcpy(&lastCirBinsCount, devCirBinsCount + numCircles - 1, 
               sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastCirBinsIndex, devCirBinsIndex + numCircles - 1, 
               sizeof(uint), cudaMemcpyDeviceToHost);
    uint cirBinsLength = lastCirBinsCount + lastCirBinsIndex;
    
    /** Step 3: Bind each bin with its circle and relative index within its circle
      *      4: Sort by bin index (using thrust::stable_sort to preserve circle 
      *         order)
      */
    uint *devCirBins_Bin, *devCirBins_Cir;
    cudaMalloc(&devCirBins_Bin, sizeof(uint) * cirBinsLength);
    cudaMalloc(&devCirBins_Cir, sizeof(uint) * cirBinsLength);
    kernelGetCirBinsPair<<<gridDim, blockDim>>>(devCirBinsIndex, 
                                                devCirBins_Bin, devCirBins_Cir,
                                                binNumLength, binPixLength);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
    
    thrust::stable_sort_by_key(thrust::device_ptr<uint>(devCirBins_Bin), 
                               thrust::device_ptr<uint>(devCirBins_Bin + cirBinsLength),
                               thrust::device_ptr<uint>(devCirBins_Cir));
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
    
    /** Now that we have all the bins in order with circle order preserved, we
      * can do each bin in parallel!
      *  
      * Step 5: Find the starting index of each bin and how many circles are in there
      */
    printf("Step 5\n");
    // Still use 256 threads per block
    uint numBins = binNumLength * binNumLength;
    dim3 cirBinsGridDim((cirBinsLength + blockDim.x - 1) / blockDim.x);
    kernelGetBinStartIndex<<<cirBinsGridDim, blockDim>>>(devBinStartIndex,
                                                         devCirBins_Bin,
                                                         cirBinsLength);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
                                                         
    cudaMemset(devBinStartIndex, 0, sizeof(uint));
    dim3 binsGridDim((numBins + blockDim.x - 1) / blockDim.x);
    kernelGetBinSizes<<<binsGridDim, blockDim>>>(devBinNumCir, 
                                                 devBinStartIndex, 
                                                 numBins,
                                                 cirBinsLength);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
    
    thrust::device_ptr<uint> result = thrust::max_element(thrust::device_ptr<uint>(devBinNumCir), 
                                                          thrust::device_ptr<uint>(devBinNumCir + numBins));
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Time: %u\n", delta_us);
    
    uint maxBinNumCir = result[0];
  
    /** Step 6: Finally render the circles, with each block of pixels being drawn
      *         on a separate thread.
      */
    //printf("Step 6, %d\n", maxBinNumCir);
    dim3 pixelBlockDim(16,16);
    dim3 pixelGridDim((image->width - 1)  / pixelBlockDim.x + 1,
						          (image->height - 1) / pixelBlockDim.y + 1);
    
    if (maxBinNumCir < 1000) {
      uint sharedMemSize = maxBinNumCir * (7*sizeof(float));
                                                   //3 for center coordinates
                                                   //1 for radius
      //printf("%d\n", sharedMemSize);             //3 for color
      kernelRenderCirclesTRUE<<<pixelGridDim, pixelBlockDim, 
                                              sharedMemSize>>>(devCirBins_Cir, 
                                                               devBinStartIndex,
                                                               binNumLength,
                                                               binPixLength,
                                                               devBinNumCir,
                                                               maxBinNumCir,
                                                               conditional,
                                                               true);
    } else {
      //Too much memory to share across threads!!!
      kernelRenderCirclesTRUE<<<pixelGridDim, pixelBlockDim>>>(devCirBins_Cir, 
                                                               devBinStartIndex,
                                                               binNumLength,
                                                               binPixLength,
                                                               devBinNumCir,
                                                               maxBinNumCir,
                                                               conditional,
                                                               false);
    }
    
    // Initial solution given (BAD!)
    // kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    cudaFree(devCirBins_Bin);
    cudaFree(devCirBins_Cir);
}
