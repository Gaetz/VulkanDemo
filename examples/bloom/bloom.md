# Implementing Bloom Effects in Vulkan

Bloom is a popular post-processing effect that simulates light bleeding beyond the boundaries of bright objects in a
scene.

In this tutorial, we'll walk through implementing a bloom effect in Vulkan using a multi-pass approach with Gaussian
blur. This method is widely used in games and real-time rendering applications.

## What is Bloom?

Bloom simulates the way extremely bright light appears to "bleed" beyond its natural boundaries in photography and human
vision. This happens because of various optical phenomena including lens imperfections, diffraction, and scattering in
the eye.

A properly implemented bloom effect can add significant visual quality. It:

- Makes light sources appear more intense
- Enhances the perception of brightness
- Creates a more realistic and cinematic look
- Helps distinguish between different lighting intensities

## Overview of the Bloom Implementation

Our bloom implementation will use the following approach:

1. **Brightness extraction**: Render only the bright parts of the scene to an offscreen framebuffer
2. **Blur**: Apply a two-pass Gaussian blur (vertical and horizontal) to this brightness information
3. **Composition**: Blend the blurred bright regions with the original scene

The two-pass blur approach is much more efficient than a single-pass blur because it requires far fewer texture samples
while providing the same quality.

## Necessary Structures

First, let's look at the basic structures we need:

```cpp
#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

// Define constants for our framebuffer
#define FB_DIM 256  // Dimensions of our offscreen framebuffers (256x256)
#define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM  // Color format for our framebuffer

class VulkanExample : public VulkanExampleBase
{
public:
	bool bloom = true;

    // Cubemap texture for the skybox
    vks::TextureCubeMap cubemap;

    // 3D model resources
    struct {
        vkglTF::Model ufo;      // Main UFO model
        vkglTF::Model ufoGlow;  // Glowing parts of the UFO (this will be used for bloom extraction)
        vkglTF::Model skyBox;   // Skybox model (a simple cube)
    } models;

    // Uniform buffers for passing data to shaders
    struct {
        vks::Buffer scene;      // Scene transformation matrices
        vks::Buffer skyBox;     // Skybox transformation matrices
        vks::Buffer blurParams; // Parameters for the blur effect
    } uniformBuffers;

    // Uniform data structures that will be copied to the uniform buffers
    struct UBO {
        glm::mat4 projection;   // Perspective projection matrix
        glm::mat4 view;         // Camera view matrix
        glm::mat4 model;        // Model transformation matrix
    };

    // Blur parameters that control the bloom effect
    struct UBOBlurParams {
        float blurScale = 1.0f;     // Controls the spread of the bloom
        float blurStrength = 1.5f;  // Controls the intensity of the bloom
    };

    // Instances of the uniform data structures
    struct {
        UBO scene, skyBox;          // Transformation matrices for scene and skybox
        UBOBlurParams blurParams;   // Bloom effect parameters
    } ubos;

    // Graphics pipelines (compiled shader programs with state)
    struct {
        VkPipeline blurVert;   // Pipeline for vertical blur
        VkPipeline blurHorz;   // Pipeline for horizontal blur
        VkPipeline glowPass;   // Pipeline for extracting bright parts (glow)
        VkPipeline phongPass;  // Pipeline for normal scene rendering
        VkPipeline skyBox;     // Pipeline for skybox rendering
    } pipelines;

    // Pipeline layouts (define shader resource bindings)
    struct {
        VkPipelineLayout blur;  // Layout for blur pipelines
        VkPipelineLayout scene; // Layout for scene pipelines
    } pipelineLayouts;

    // Descriptor sets (connect resources to shaders)
    struct {
        VkDescriptorSet blurVert;  // Descriptor set for vertical blur
        VkDescriptorSet blurHorz;  // Descriptor set for horizontal blur
        VkDescriptorSet scene;     // Descriptor set for main scene rendering
        VkDescriptorSet skyBox;    // Descriptor set for skybox rendering
    } descriptorSets;

    // Descriptor set layouts (templates for descriptor sets)
    struct {
        VkDescriptorSetLayout blur;  // Layout for blur descriptor sets
        VkDescriptorSetLayout scene; // Layout for scene descriptor sets
    } descriptorSetLayouts;

   // Structure to represent a single attachment (image) for our framebuffer
   struct FrameBufferAttachment {
       VkImage image;         // The actual image
       VkDeviceMemory mem;    // GPU memory allocated for the image
       VkImageView view;      // View into the image (needed for binding)
   };
   
   // Structure to represent a complete framebuffer
   struct FrameBuffer {
       VkFramebuffer framebuffer;         // The Vulkan framebuffer object
       FrameBufferAttachment color, depth; // Color and depth attachments
       VkDescriptorImageInfo descriptor;   // Descriptor for sampling from the color attachment
   };
   
   // Structure for our offscreen rendering pass
   struct OffscreenPass {
       int32_t width, height;                 // Dimensions of the framebuffer
       VkRenderPass renderPass;               // Render pass for offscreen rendering
       VkSampler sampler;                     // Sampler for the framebuffer attachments
       std::array<FrameBuffer, 2> framebuffers; // Two framebuffers: [0] for brightness extraction, [1] for vertical blur
   } offscreenPass;
   ...
```

We use two framebuffers in our offscreen pass:

- The first one stores the extracted bright parts of our scene
- The second one is used for the first blur pass (vertical)

The results are then composited with the final scene during the horizontal blur pass.

## Constructor and destructor

```cpp
 VulkanExample() : VulkanExampleBase()
{
   title = "Bloom (offscreen rendering)";
   timerSpeed *= 0.5f;
   camera.type = Camera::CameraType::lookat;
   camera.setPosition(glm::vec3(0.0f, 0.0f, -10.25f));
   camera.setRotation(glm::vec3(7.5f, -343.0f, 0.0f));
   camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
}

~VulkanExample()
{
   // Clean up used Vulkan resources
   // Note : Inherited destructor cleans up resources stored in base class
   
   // Clean up the sampler used for offscreen rendering
   vkDestroySampler(device, offscreenPass.sampler, nullptr);
   
   // Clean up framebuffers and their attachments
   for (auto& framebuffer : offscreenPass.framebuffers)
   {
      // Clean up color attachment
      vkDestroyImageView(device, framebuffer.color.view, nullptr);
      vkDestroyImage(device, framebuffer.color.image, nullptr);
      vkFreeMemory(device, framebuffer.color.mem, nullptr);
      
      // Clean up depth attachment
      vkDestroyImageView(device, framebuffer.depth.view, nullptr);
      vkDestroyImage(device, framebuffer.depth.image, nullptr);
      vkFreeMemory(device, framebuffer.depth.mem, nullptr);
      
      // Clean up framebuffer object
      vkDestroyFramebuffer(device, framebuffer.framebuffer, nullptr);
   }
   
   // Clean up render pass
   vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);
   
   // Clean up pipelines
   vkDestroyPipeline(device, pipelines.blurHorz, nullptr);
   vkDestroyPipeline(device, pipelines.blurVert, nullptr);
   vkDestroyPipeline(device, pipelines.phongPass, nullptr);
   vkDestroyPipeline(device, pipelines.glowPass, nullptr);
   vkDestroyPipeline(device, pipelines.skyBox, nullptr);
   
   // Clean up pipeline layouts
   vkDestroyPipelineLayout(device, pipelineLayouts.blur, nullptr);
   vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);
   
   // Clean up descriptor set layouts
   vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.blur, nullptr);
   vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);
   
   // Clean up uniform buffers
   uniformBuffers.scene.destroy();
   uniformBuffers.skyBox.destroy();
   uniformBuffers.blurParams.destroy();
   
   // Clean up cubemap texture
   cubemap.destroy();
}
```

## Preparing the Offscreen Framebuffers

We need to set up offscreen framebuffers for the bloom effect:

```cpp
// Creates a single offscreen framebuffer with color and depth attachments
void prepareOffscreenFramebuffer(FrameBuffer *frameBuf, VkFormat colorFormat, VkFormat depthFormat)
{
    // ---- STEP 1: Create the color attachment (this will store color information) ----
    
    // Set up the image creation info structure
    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;                // 2D image
    image.format = colorFormat;                       // Format specified by parameter
    image.extent.width = FB_DIM;                      // Width from our define
    image.extent.height = FB_DIM;                     // Height from our define
    image.extent.depth = 1;                           // Not a 3D image
    image.mipLevels = 1;                              // No mipmapping needed
    image.arrayLayers = 1;                            // Not an array texture
    image.samples = VK_SAMPLE_COUNT_1_BIT;            // No multisampling
    image.tiling = VK_IMAGE_TILING_OPTIMAL;           // Let Vulkan optimize the layout
    
    // These usage flags are important:
    // - COLOR_ATTACHMENT_BIT: We'll render to this image
    // - SAMPLED_BIT: We'll sample from this image in the shader (for the blur passes)
    image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    // Set up memory allocation info
    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    // Set up the image view creation info
    VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;      // 2D image view
    colorImageView.format = colorFormat;                 // Same format as the image
    colorImageView.flags = 0;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;  // This is a color image
    colorImageView.subresourceRange.baseMipLevel = 0;   
    colorImageView.subresourceRange.levelCount = 1;      // Just one level, no mipmaps
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;      // Not an array texture

    // Create the image
    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->color.image));
    
    // Get memory requirements and allocate memory
    vkGetImageMemoryRequirements(device, frameBuf->color.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->color.mem));
    
    // Bind the memory to the image
    VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->color.image, frameBuf->color.mem, 0));

    // Set the image for the view and create the view
    colorImageView.image = frameBuf->color.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &frameBuf->color.view));

    // ---- STEP 2: Create the depth attachment (for depth testing during rendering) ----
    
    // Now set up the depth image with the same pattern
    image.format = depthFormat;    // Use the depth format
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;  // This will be used as depth/stencil

    // Set up the depth view creation info
    VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = depthFormat;
    depthStencilView.flags = 0;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;  // This is a depth image
    
    // Add stencil aspect if the format includes stencil component
    if (vks::tools::formatHasStencil(depthFormat)) {
        depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;

    // Create depth image, get requirements, allocate and bind memory
    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->depth.image));
    vkGetImageMemoryRequirements(device, frameBuf->depth.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->depth.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->depth.image, frameBuf->depth.mem, 0));

    // Create the depth image view
    depthStencilView.image = frameBuf->depth.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &frameBuf->depth.view));

    // ---- STEP 3: Create the framebuffer by combining the attachments ----
    
    // Gather attachment views
    VkImageView attachments[2];
    attachments[0] = frameBuf->color.view;
    attachments[1] = frameBuf->depth.view;

    // Set up framebuffer creation info
    VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass.renderPass;  // Use the offscreen render pass
    fbufCreateInfo.attachmentCount = 2;                    // 2 attachments (color and depth)
    fbufCreateInfo.pAttachments = attachments;             // The attachment array
    fbufCreateInfo.width = FB_DIM;                         // Width of the framebuffer
    fbufCreateInfo.height = FB_DIM;                        // Height of the framebuffer
    fbufCreateInfo.layers = 1;                             // Single layer framebuffer

    // Create the framebuffer
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuf->framebuffer));
    
    // ---- STEP 4: Set up the descriptor for sampling the color attachment in shaders ----
    
    // This info will be used to create a descriptor that allows us to sample from this framebuffer's color attachment
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // Layout for sampling
    frameBuf->descriptor.imageView = frameBuf->color.view;                        // The view to sample from
    frameBuf->descriptor.sampler = offscreenPass.sampler;                         // The sampler to use
}

// Create both framebuffers needed for the bloom effect
void prepareOffscreen()
{
    offscreenPass.width = FB_DIM;    // Set dimensions of our offscreen buffers
    offscreenPass.height = FB_DIM;

    // STEP 1: Find a suitable depth format that's supported by the GPU
    VkFormat fbDepthFormat;
    VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
    assert(validDepthFormat);  // Make sure we got a valid format
    
    // STEP 2: Create a render pass for offscreen rendering
    // This defines how the attachments are used
    std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
    
    // Color attachment description
    attchmentDescriptions[0].format = FB_COLOR_FORMAT;  // Format from our define
    attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;  // No multisampling
    attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // Clear at start of render pass
    attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // Store at end for later use
    attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // No stencil
    attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;  // No stencil
    attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Don't care about initial layout
    // This final layout is crucial: We need the image in shader read only layout to sample from it
    attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    // Depth attachment description
    attchmentDescriptions[1].format = fbDepthFormat;  // Format from earlier
    attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // Clear depth at start
    attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;  // Don't need to store depth
    attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // STEP 3: Set up attachment references for the subpass
    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };  // Attachment 0 is color
    VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };  // Attachment 1 is depth

    // STEP 4: Create the subpass description
    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;  // This is a graphics subpass
    subpassDescription.colorAttachmentCount = 1;  // One color attachment
    subpassDescription.pColorAttachments = &colorReference;  // The color attachment reference
    subpassDescription.pDepthStencilAttachment = &depthReference;  // The depth attachment reference

    // STEP 5: Set up subpass dependencies for layout transitions
    // These control the timing of layout transitions to ensure data is available
    std::array<VkSubpassDependency, 2> dependencies;

    // First dependency - external to this subpass
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;  // Come from outside the render pass
    dependencies[0].dstSubpass = 0;  // Going to our first subpass
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // Wait on fragment shader (previous frame)
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  // Before writing to color attachment
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;  // Previous shader reads
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;  // Going to write to color attachment
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;  // Only depend on regions that overlap

    // Second dependency - this subpass to external
    dependencies[1].srcSubpass = 0;  // Coming from our subpass
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;  // Going to outside the render pass
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  // After writing color
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;  // Before fragment shader reads (next pass)
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;  // After color writes
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;  // Before shader reads
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // STEP 6: Create the render pass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
    renderPassInfo.pAttachments = attchmentDescriptions.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass.renderPass));
    
    // STEP 7: Create a sampler for sampling from the attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;  // Linear filtering for upscaling
    sampler.minFilter = VK_FILTER_LINEAR;  // Linear filtering for downscaling
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;  // No mipmapping needed
    // Use clamp to edge to prevent edge artifacts when sampling
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;  // No anisotropic filtering
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreenPass.sampler));

    // STEP 8: Create both framebuffers
    // First framebuffer for bright parts extraction
    prepareOffscreenFramebuffer(&offscreenPass.framebuffers[0], FB_COLOR_FORMAT, fbDepthFormat);
    // Second framebuffer for vertical blur
    prepareOffscreenFramebuffer(&offscreenPass.framebuffers[1], FB_COLOR_FORMAT, fbDepthFormat);
}
```

## Shader Implementation

The bloom effect requires several shader pairs. Let's examine each of them:

### 1. Color Pass (Brightness Extraction)

The color pass identifies and extracts bright parts of our scene:

`colorpass.vert`

```glsl
// Vertex shader for brightness extraction
#version 450

// Input vertex attributes
layout (location = 0) in vec4 inPos;    // Vertex position
layout (location = 1) in vec2 inUV;     // Texture coordinates
layout (location = 2) in vec3 inColor;  // Vertex color

// Uniform buffer containing transformation matrices
layout (binding = 0) uniform UBO
{
    mat4 projection;  // Projection matrix
    mat4 view;        // View matrix
    mat4 model;       // Model matrix
} ubo;

// Outputs to fragment shader
layout (location = 0) out vec3 outColor;  // Pass color to fragment shader
layout (location = 1) out vec2 outUV;     // Pass texture coordinates to fragment shader

// Builtin output required by the pipeline
out gl_PerVertex
{
    vec4 gl_Position;  // Output clip-space position
};

void main()
{
    // Pass texture coordinates to fragment shader
    outUV = inUV;
    
    // Pass vertex color to fragment shader
    outColor = inColor;
    
    // Calculate clip-space position by multiplying with MVP matrices
    gl_Position = ubo.projection * ubo.view * ubo.model * inPos;
}
```

`colorpass.frag`

```glsl
// Fragment shader for brightness extraction
#version 450

// Sampler for color texture (optional usage)
layout (binding = 1) uniform sampler2D colorMap;

// Inputs from vertex shader
layout (location = 0) in vec3 inColor;  // Vertex color
layout (location = 1) in vec2 inUV;     // Texture coordinates

// Output color
layout (location = 0) out vec4 outFragColor;

void main()
{
    // For this simple example, we just output the vertex color directly
    // This is the color from our "glowing" model parts
    outFragColor.rgb = inColor;
    
    // Alternative approach - we could sample from a texture instead
    // This would allow for more complex brightness extraction
    // outFragColor = texture(colorMap, inUV);
    
    // In a more complex implementation, we might threshold brightness here:
    // vec3 color = texture(colorMap, inUV).rgb;
    // float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722)); // Luminance calculation
    // if (brightness > 1.0) {
    //     outFragColor = vec4(color, 1.0);
    // } else {
    //     outFragColor = vec4(0.0, 0.0, 0.0, 1.0);
    // }
}
```

### 2. Gaussian Blur

For the blur, we use a two-pass approach (vertical and horizontal) with the same shaders but different parameters:

`gaussblur.vert`

```glsl
// Vertex shader for fullscreen quad blur passes
#version 450

// Output texture coordinates to the fragment shader
layout (location = 0) out vec2 outUV;

// Builtin output required by the pipeline
out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    // Generate fullscreen triangle from vertex ID
    // This trick generates a fullscreen triangle with only 3 vertices
    // without needing a vertex buffer:
    //   gl_VertexIndex=0: outUV=(0,0), gl_Position=(-1,-1)
    //   gl_VertexIndex=1: outUV=(2,0), gl_Position=( 3,-1)
    //   gl_VertexIndex=2: outUV=(0,2), gl_Position=(-1, 3)
    // This creates a triangle that covers the entire screen with minimal vertices
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
```

`gaussblur.frag`

```glsl
// Fragment shader for Gaussian blur
#version 450

// Sampler for the input texture to blur
layout (binding = 1) uniform sampler2D samplerColor;

// Uniform buffer for blur parameters
layout (binding = 0) uniform UBO
{
    float blurScale;     // Controls the blur sample distance
    float blurStrength;  // Controls the blur intensity
} ubo;

// Specialization constant to determine blur direction
// 0 = vertical, 1 = horizontal
// This allows us to use the same shader for both directions
layout (constant_id = 0) const int blurdirection = 0;

// Input texture coordinates from vertex shader
layout (location = 0) in vec2 inUV;

// Output color
layout (location = 0) out vec4 outFragColor;

void main()
{
    // Define Gaussian weights for a 5-tap filter
    // These weights approximate a Gaussian distribution
    // The center sample gets the highest weight
    float weight[5];
    weight[0] = 0.227027;  // Center weight
    weight[1] = 0.1945946; // First offset (both directions)
    weight[2] = 0.1216216; // Second offset
    weight[3] = 0.054054;  // Third offset
    weight[4] = 0.016216;  // Fourth offset (weakest contribution)
    
    // Calculate texel size for proper sampling offset
    // This ensures blur scale works correctly regardless of texture resolution
    vec2 tex_offset = 1.0 / textureSize(samplerColor, 0) * ubo.blurScale;
    
    // Start with center sample weighted by center weight
    vec3 result = texture(samplerColor, inUV).rgb * weight[0];

    // Sample in both positive and negative directions from center
    for (int i = 1; i < 5; ++i)
    {
        if (blurdirection == 1)
        {
            // Horizontal blur: sample along the x-axis
            // Add weighted samples from the positive direction (right)
            result += texture(samplerColor, inUV + vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
            // Add weighted samples from the negative direction (left)
            result += texture(samplerColor, inUV - vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
        }
        else
        {
            // Vertical blur: sample along the y-axis
            // Add weighted samples from the positive direction (down)
            result += texture(samplerColor, inUV + vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
            // Add weighted samples from the negative direction (up)
            result += texture(samplerColor, inUV - vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
        }
    }
    
    // Output the blurred result
    outFragColor = vec4(result, 1.0);
}
```

### 3. Final Scene Rendering (Phong shading)

For the main scene, we use a basic Phong lighting model:

`phongpass.vert`

```glsl
// Vertex shader for main scene rendering with Phong lighting
#version 450

// Input vertex attributes
layout (location = 0) in vec4 inPos;     // Vertex position
layout (location = 1) in vec2 inUV;      // Texture coordinates
layout (location = 2) in vec3 inColor;   // Vertex color
layout (location = 3) in vec3 inNormal;  // Vertex normal

// Uniform buffer containing transformation matrices
layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

// Outputs to fragment shader
layout (location = 0) out vec3 outNormal;    // Surface normal in view space
layout (location = 1) out vec2 outUV;        // Texture coordinates
layout (location = 2) out vec3 outColor;     // Vertex color
layout (location = 3) out vec3 outViewVec;   // Vector from vertex to camera
layout (location = 4) out vec3 outLightVec;  // Vector from vertex to light

// Builtin output required by the pipeline
out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    // Pass normal to fragment shader
    outNormal = inNormal;
    
    // Pass color to fragment shader
    outColor = inColor;
    
    // Pass texture coordinates to fragment shader
    outUV = inUV;
    
    // Calculate clip-space position
    gl_Position = ubo.projection * ubo.view * ubo.model * inPos;

    // Define a fixed light position
    vec3 lightPos = vec3(-5.0, -5.0, 0.0);
    
    // Transform vertex position to view space
    vec4 pos = ubo.view * ubo.model * inPos;
    
    // Transform normal to view space
    // Using the view-model matrix ensures normals are correctly transformed
    outNormal = mat3(ubo.view * ubo.model) * inNormal;
    
    // Calculate light vector (from vertex to light) in view space
    outLightVec = lightPos - pos.xyz;
    
    // Calculate view vector (from vertex to camera) in view space
    // In view space, the camera is at the origin (0,0,0), so this is just the negative of position
    outViewVec = -pos.xyz;
}
```

`phongpass.frag`

```glsl
// Fragment shader for main scene rendering with Phong lighting
#version 450

// Sampler for optional texture (not used in this basic example)
layout (binding = 1) uniform sampler2D colorMap;

// Inputs from vertex shader
layout (location = 0) in vec3 inNormal;    // Surface normal
layout (location = 1) in vec2 inUV;        // Texture coordinates
layout (location = 2) in vec3 inColor;     // Vertex color
layout (location = 3) in vec3 inViewVec;   // View vector (vertex to camera)
layout (location = 4) in vec3 inLightVec;  // Light vector (vertex to light)

// Output color
layout (location = 0) out vec4 outFragColor;

void main()
{
    // Initialize ambient light component
    vec3 ambient = vec3(0.0f);

    // Special handling for glow parts (identified by high color values)
    // Parts of the model that should glow get a boost in ambient lighting
    if ((inColor.r >= 0.9) || (inColor.g >= 0.9) || (inColor.b >= 0.9))
    {
        // Add ambient lighting to glow parts
        ambient = inColor * 0.25;
    }

    // Normalize vectors for lighting calculations
    vec3 N = normalize(inNormal);       // Surface normal
    vec3 L = normalize(inLightVec);     // Light direction
    vec3 V = normalize(inViewVec);      // View direction
    vec3 R = reflect(-L, N);            // Reflection vector (for specular)
    
    // Calculate diffuse component (Lambert's law)
    // dot(N,L) gives cosine of angle between normal and light vector
    // max ensures no negative values (light from backside doesn't illuminate)
    vec3 diffuse = max(dot(N, L), 0.0) * inColor;
    
    // Calculate specular component (Phong reflection model)
    // dot(R,V) gives cosine of angle between reflection and view vector
    // pow(value, 8.0) creates a focused specular highlight
    vec3 specular = pow(max(dot(R, V), 0.0), 8.0) * vec3(0.75);
    
    // Combine ambient, diffuse, and specular components for final color
    outFragColor = vec4(ambient + diffuse + specular, 1.0);
}
```

## Prepare uniform buffers

We will prepare and initialize uniform buffer containing the shader uniforms

```cpp
void prepareUniformBuffers()
{
   // STEP 1: Create uniform buffer for scene rendering (model, view, projection matrices)
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,  // Buffer will be used as a uniform buffer
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,  // Host can access, changes are automatically visible to GPU
      &uniformBuffers.scene,  // Output buffer object
      sizeof(ubos.scene)));   // Size of the buffer (size of UBO struct)
   
   // STEP 2: Create uniform buffer for blur parameters
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.blurParams,
      sizeof(ubos.blurParams)));
   
   // STEP 3: Create uniform buffer for skybox rendering
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.skyBox,
      sizeof(ubos.skyBox)));
   
   // STEP 4: Map memory for all uniform buffers so we can update them from CPU
   // With HOST_COHERENT, changes to mapped memory are automatically flushed to GPU
   VK_CHECK_RESULT(uniformBuffers.scene.map());
   VK_CHECK_RESULT(uniformBuffers.blurParams.map());
   VK_CHECK_RESULT(uniformBuffers.skyBox.map());
   
   // STEP 5: Initialize all uniform buffers with default values
   updateUniformBuffersScene();   // Initialize scene matrices
   updateUniformBuffersBlur();    // Initialize blur parameters
}

```

## Setting Up the Pipeline

We need to set up several pipelines for our bloom effect:

```cpp
void preparePipelines()
{
    // STEP 1: Create pipeline layouts
    // First, create layout for blur pipelines
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts.blur,  // Descriptor set layout for blur shaders
        1);                          // Only one descriptor set
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.blur));

    // Next, create layout for scene rendering pipelines
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts.scene,  // Descriptor set layout for scene shaders
        1);                           // Only one descriptor set
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.scene));

    // STEP 2: Set up common pipeline creation state structures 
    // Input assembly - how to interpret vertex data
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  // Interpret as triangles
        0,                                    // No primitive restart
        VK_FALSE);                            // Don't enable primitive restart
        
    // Rasterization - how to convert vertices to fragments
    VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(
        VK_POLYGON_MODE_FILL,                // Fill polygons with fragments
        VK_CULL_MODE_NONE,                   // Don't cull any triangles by default
        VK_FRONT_FACE_COUNTER_CLOCKWISE,     // Counter-clockwise order is front-facing
        0);                                  // No depth bias
        
    // Color blending attachment - how to blend fragment colors
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(
        0xf,                                 // Color write mask (RGBA)
        VK_FALSE);                           // Disable blending initially
        
    // Color blending state - contains attachment state
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(
        1,                                   // One attachment
        &blendAttachmentState);              // The attachment state
        
    // Depth stencil - controls depth testing
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(
        VK_TRUE,                             // Enable depth test
        VK_TRUE,                             // Enable depth write
        VK_COMPARE_OP_LESS_OR_EQUAL);        // Depth test function
        
    // Viewport state - defines viewport transformation
    VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(
        1,                                   // One viewport
        1,                                   // One scissor rectangle
        0);                                  // No static viewport/scissor
        
    // Multisample state - controls multisampling for anti-aliasing
    VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(
        VK_SAMPLE_COUNT_1_BIT,               // No multisampling (1 sample per pixel)
        0);                                  // No sample shading
        
    // Dynamic state - states that can be changed at draw time
    std::vector<VkDynamicState> dynamicStateEnables = { 
        VK_DYNAMIC_STATE_VIEWPORT,           // Dynamic viewport
        VK_DYNAMIC_STATE_SCISSOR             // Dynamic scissor
    };
    VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(
        dynamicStateEnables);                // Array of enabled dynamic states
        
    // Shader stages - each pipeline needs vertex and fragment shader
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // STEP 3: Create common pipeline creation info structure
    VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(
        pipelineLayouts.blur,                // Initial pipeline layout (will change per pipeline)
        renderPass,                          // Initial render pass (will change for offscreen)
        0);                                  // Subpass index
    pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
    pipelineCI.pRasterizationState = &rasterizationStateCI;
    pipelineCI.pColorBlendState = &colorBlendStateCI;
    pipelineCI.pMultisampleState = &multisampleStateCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pDynamicState = &dynamicStateCI;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();

    // STEP 4: Create blur pipelines
    // Load blur shader modules
    shaderStages[0] = loadShader(getShadersPath() + "bloom/gaussblur.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/gaussblur.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Empty vertex input state for fullscreen quad (vertices generated in shader)
    VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts.blur;
    
    // STEP 5: Configure blend state for horizontal blur (final composition)
    // This is crucial for the bloom effect: we use additive blending for the final pass
    blendAttachmentState.colorWriteMask = 0xF;  // Write to all RGBA channels
    blendAttachmentState.blendEnable = VK_TRUE; // Enable blending
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD; // Add source and destination
    // Use ONE blend factor to achieve additive blending
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; 
    blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    // Same for alpha channel
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

    // STEP 6: Set up specialization constants for blur direction
    // This allows us to use the same shader for both vertical and horizontal blur
    // Specialization constants are compile-time constants for shaders
    uint32_t blurdirection = 0;  // 0 = vertical, 1 = horizontal
    VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(
        0,                        // Constant ID in shader
        0,                        // Offset in data
        sizeof(uint32_t));        // Size of data
    VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(
        1,                         // One constant
        &specializationMapEntry,   // Map entry
        sizeof(uint32_t),          // Data size
        &blurdirection);           // Data pointer
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    
    // STEP 7: Create vertical blur pipeline for offscreen pass
    pipelineCI.renderPass = offscreenPass.renderPass;  // Use offscreen render pass
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.blurVert));
    
    // STEP 8: Create horizontal blur pipeline for main pass (with additive blending)
    blurdirection = 1;  // Switch to horizontal blur
    pipelineCI.renderPass = renderPass;  // Use main render pass
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.blurHorz));

    // STEP 9: Create pipeline for main scene rendering (Phong shading)
    // Set up vertex input state for 3D model rendering
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({
        vkglTF::VertexComponent::Position,  // Vertex position
        vkglTF::VertexComponent::UV,        // Texture coordinates
        vkglTF::VertexComponent::Color,     // Vertex color
        vkglTF::VertexComponent::Normal     // Vertex normal
    });
    pipelineCI.layout = pipelineLayouts.scene;  // Use scene pipeline layout
    
    // Load Phong shader modules
    shaderStages[0] = loadShader(getShadersPath() + "bloom/phongpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/phongpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Disable blending for normal scene rendering
    blendAttachmentState.blendEnable = VK_FALSE;
    
    // Enable depth writing for proper occlusion
    depthStencilStateCI.depthWriteEnable = VK_TRUE;
    
    // Set back-face culling for efficiency
    rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
    
    // Use main render pass
    pipelineCI.renderPass = renderPass;
    
    // Create Phong pipeline
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.phongPass));

    // STEP 10: Create pipeline for glow parts extraction (color-only pass)
    // Load color pass shader modules
    shaderStages[0] = loadShader(getShadersPath() + "bloom/colorpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/colorpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Use offscreen render pass for initial brightness extraction
    pipelineCI.renderPass = offscreenPass.renderPass;
    
    // Create glow pass pipeline
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.glowPass));

    // STEP 11: Create pipeline for skybox
    // Load skybox shader modules
    shaderStages[0] = loadShader(getShadersPath() + "bloom/skybox.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/skybox.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Disable depth writes for skybox (it's always in the background)
    depthStencilStateCI.depthWriteEnable = VK_FALSE;
    
    // Use front face culling for skybox (we're inside the cube)
    rasterizationStateCI.cullMode = VK_CULL_MODE_FRONT_BIT;
    
    // Use main render pass
    pipelineCI.renderPass = renderPass;
    
    // Create skybox pipeline
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.skyBox));
}
```

Note the use of specialization constants to create both blur pipelines from the same shader. This is a more efficient
approach than having two separate shaders.

## Descriptor Setup

We need to set up descriptors to bind resources to our shaders:

```cpp
void setupDescriptors()
{
    // STEP 1: Create descriptor pool
    // This defines how many descriptors of each type we can allocate
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6)
    };
    // Create the pool with space for 5 descriptor sets
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 5);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // STEP 2: Create descriptor set layouts
    // First, create blur descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0: Fragment shader uniform buffer (blur parameters)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            VK_SHADER_STAGE_FRAGMENT_BIT,              // Shader stage visibility
            0),                                        // Binding point
        // Binding 1: Fragment shader image sampler (texture to blur)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Descriptor type
            VK_SHADER_STAGE_FRAGMENT_BIT,              // Shader stage visibility
            1)                                         // Binding point
    };
    // Create the blur descriptor set layout
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = 
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),                  // Bindings array
            static_cast<uint32_t>(setLayoutBindings.size())); // Number of bindings
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.blur));

    // STEP 3: Create scene rendering descriptor set layout
    setLayoutBindings = {
        // Binding 0: Vertex shader uniform buffer (transformation matrices)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            VK_SHADER_STAGE_VERTEX_BIT,                // Shader stage visibility
            0),                                        // Binding point
        // Binding 1: Fragment shader image sampler (texture)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Descriptor type
            VK_SHADER_STAGE_FRAGMENT_BIT,              // Shader stage visibility
            1),                                        // Binding point
        // Binding 2: Fragment shader uniform buffer (extra parameters if needed)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            VK_SHADER_STAGE_FRAGMENT_BIT,              // Shader stage visibility
            2),                                        // Binding point
    };
    // Create the scene descriptor set layout
    descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.scene));

    // STEP 4: Allocate and update descriptor sets
    // First, vertical blur descriptor set
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo = 
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool,                            // Descriptor pool
            &descriptorSetLayouts.blur,                // Descriptor set layout
            1);                                        // Number of sets to allocate
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurVert));
    
    // Update vertical blur descriptor set
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
        // Binding 0: Uniform buffer with blur parameters
        vks::initializers::writeDescriptorSet(
            descriptorSets.blurVert,                   // Destination descriptor set
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            0,                                         // Binding point
            &uniformBuffers.blurParams.descriptor),    // Buffer info
        // Binding 1: Image sampler for reading color from first framebuffer
        vks::initializers::writeDescriptorSet(
            descriptorSets.blurVert,                   // Destination descriptor set
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Descriptor type
            1,                                         // Binding point
            &offscreenPass.framebuffers[0].descriptor) // Image info
    };
    // Apply the writes to the descriptor set
    vkUpdateDescriptorSets(
        device,                                        // Logical device
        static_cast<uint32_t>(writeDescriptorSets.size()), // Write count
        writeDescriptorSets.data(),                    // Write array
        0, nullptr);                                   // No copies
    
    // STEP 5: Horizontal blur descriptor set (similar to vertical)
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurHorz));
    writeDescriptorSets = {
        // Binding 0: Uniform buffer with blur parameters (same as vertical)
        vks::initializers::writeDescriptorSet(
            descriptorSets.blurHorz,                   // Destination descriptor set
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            0,                                         // Binding point
            &uniformBuffers.blurParams.descriptor),    // Buffer info
        // Binding 1: Image sampler for reading color from second framebuffer (vertical blur result)
        vks::initializers::writeDescriptorSet(
            descriptorSets.blurHorz,                   // Destination descriptor set
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Descriptor type
            1,                                         // Binding point
            &offscreenPass.framebuffers[1].descriptor) // Image info
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    // STEP 6: Scene rendering descriptor set
    descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(
        descriptorPool,                                // Descriptor pool
        &descriptorSetLayouts.scene,                   // Descriptor set layout
        1);                                            // Number of sets to allocate
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.scene));
    
    // Update scene descriptor set
    writeDescriptorSets = {
        // Binding 0: Vertex shader uniform buffer (scene matrices)
        vks::initializers::writeDescriptorSet(
            descriptorSets.scene,                      // Destination descriptor set
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            0,                                         // Binding point
            &uniformBuffers.scene.descriptor)          // Buffer info
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    // STEP 7: Skybox descriptor set
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.skyBox));
    
    // Update skybox descriptor set
    writeDescriptorSets = {
        // Binding 0: Vertex shader uniform buffer (skybox matrices)
        vks::initializers::writeDescriptorSet(
            descriptorSets.skyBox,                     // Destination descriptor set
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         // Descriptor type
            0,                                         // Binding point
            &uniformBuffers.skyBox.descriptor),        // Buffer info
        // Binding 1: Fragment shader cubemap sampler
        vks::initializers::writeDescriptorSet(
            descriptorSets.skyBox,                     // Destination descriptor set
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // Descriptor type
            1,                                         // Binding point
            &cubemap.descriptor)                       // Image info
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}
```

## Command Buffer Construction

Now let's build our command buffer:

```cpp
void buildCommandBuffers()
{
// Basic initialization for command buffer recording
VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    // Clear values for color and depth attachments
    VkClearValue clearValues[2];
    VkViewport viewport;
    VkRect2D scissor;

    /*
        The bloom method used in this example is multi-pass and renders the vertical blur first and then the horizontal one.
        While it's possible to blur in one pass, this method is widely used as it requires far fewer samples to generate the blur.
        
        Here's the overall rendering sequence:
        1. First pass: Render bright parts to offscreen buffer 0
        2. Second pass: Apply vertical blur from buffer 0 to buffer 1
        3. Final pass: Render normal scene and blend in horizontally blurred result from buffer 1 using additive blending
    */
    
    // Create a command buffer for each swap chain image
    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        // Begin command buffer recording
        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        // Only execute bloom-related rendering if the bloom effect is enabled
        if (bloom) {
            // ------------------------------------------------------------------------------------------------
            // FIRST PASS: Extract bright parts of the scene
            // ------------------------------------------------------------------------------------------------
            
            // Set clear values for the framebuffer (black color, full depth)
            clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };  // Black clear color
            clearValues[1].depthStencil = { 1.0f, 0 };              // Far plane depth

            // Set up the render pass begin info
            VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
            renderPassBeginInfo.renderPass = offscreenPass.renderPass;
            renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[0].framebuffer;  // First framebuffer for brightness extraction
            renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
            renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
            renderPassBeginInfo.clearValueCount = 2;                // Clear both color and depth
            renderPassBeginInfo.pClearValues = clearValues;

            // Set viewport and scissor rectangle for offscreen rendering
            viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            // Begin the first render pass: Render only the glowing parts to the first offscreen framebuffer
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            // Bind the descriptor set containing transformation matrices
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
            
            // Bind the glow pass pipeline - this renders bright parts only
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.glowPass);
            
            // Draw the glowing model parts - this is key for the bloom effect
            models.ufoGlow.draw(drawCmdBuffers[i]);
            
            // End the render pass
            vkCmdEndRenderPass(drawCmdBuffers[i]);

            // ------------------------------------------------------------------------------------------------
            // SECOND PASS: Apply vertical blur to the bright parts
            // ------------------------------------------------------------------------------------------------
            
            // Now set up to render to the second framebuffer with vertical blur
            renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[1].framebuffer;  // Second framebuffer for vertical blur
            
            // Begin the second render pass
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            // Bind the vertical blur descriptor set (containing the first framebuffer as texture)
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blur, 0, 1, &descriptorSets.blurVert, 0, NULL);
            
            // Bind the vertical blur pipeline
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blurVert);
            
            // Draw a full-screen triangle (3 vertices, 1 instance) - no vertex buffers needed
            // The vertex shader generates the triangle positions
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
            
            // End the render pass
            vkCmdEndRenderPass(drawCmdBuffers[i]);
        }

        // ------------------------------------------------------------------------------------------------
        // THIRD PASS: Main scene rendering with final horizontal blur blend
        // ------------------------------------------------------------------------------------------------
        
        // Set clear values for the final framebuffer
        clearValues[0].color = defaultClearColor;      // Clear to the default background color
        clearValues[1].depthStencil = { 1.0f, 0 };     // Far plane depth

        // Set up the render pass begin info for the final pass
        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;                   // Main render pass
        renderPassBeginInfo.framebuffer = frameBuffers[i];             // Swap chain framebuffer
        renderPassBeginInfo.renderArea.extent.width = width;           // Full window size
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        // Begin the final render pass
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Set viewport and scissor for main window rendering
        viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        // ------------------------------------------------------------------------------------------------
        // STEP 1: Render skybox first (furthest back)
        // ------------------------------------------------------------------------------------------------
        
        // Bind skybox descriptor set
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.skyBox, 0, NULL);
        
        // Bind skybox pipeline
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.skyBox);
        
        // Draw skybox cube
        models.skyBox.draw(drawCmdBuffers[i]);

        // ------------------------------------------------------------------------------------------------
        // STEP 2: Render the main 3D scene (UFO model)
        // ------------------------------------------------------------------------------------------------
        
        // Bind scene descriptor set
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
        
        // Bind Phong lighting pipeline
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.phongPass);
        
        // Draw the main UFO model
        models.ufo.draw(drawCmdBuffers[i]);

        // ------------------------------------------------------------------------------------------------
        // STEP 3: Apply horizontal blur with additive blending (if bloom is enabled)
        // ------------------------------------------------------------------------------------------------
        if (bloom)
        {
            // Bind horizontal blur descriptor set (contains the vertically blurred texture)
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blur, 0, 1, &descriptorSets.blurHorz, 0, NULL);
            
            // Bind horizontal blur pipeline
            // This pipeline is set up with additive blending to achieve the bloom effect
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blurHorz);
            
            // Draw a full-screen triangle - this applies the horizontally blurred glow
            // and adds it to the scene with additive blending
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
        }

        // ------------------------------------------------------------------------------------------------
        // STEP 4: Draw UI elements on top of everything
        // ------------------------------------------------------------------------------------------------
        drawUI(drawCmdBuffers[i]);
        
        // End the render pass
        vkCmdEndRenderPass(drawCmdBuffers[i]);
        
        // End command buffer recording
        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}
```

## Loading assets and finishing preparations

Finally, we load our assets and finish the setup:

```cpp
void loadAssets()
{
    // Set up flags for how to load the glTF models
    const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | 
                                     vkglTF::FileLoadingFlags::PreMultiplyVertexColors | 
                                     vkglTF::FileLoadingFlags::FlipY;
    
    // Load the main UFO model (regular parts)
    models.ufo.loadFromFile(getAssetPath() + "models/retroufo.gltf", vulkanDevice, queue, glTFLoadingFlags);
    
    // Load the glowing parts of the UFO model
    // This is a separate model file containing only the parts that should glow
    models.ufoGlow.loadFromFile(getAssetPath() + "models/retroufo_glow.gltf", vulkanDevice, queue, glTFLoadingFlags);
    
    // Load a simple cube model for the skybox
    models.skyBox.loadFromFile(getAssetPath() + "models/cube.gltf", vulkanDevice, queue, glTFLoadingFlags);
    
    // Load the cubemap texture for the skybox
    cubemap.loadFromFile(getAssetPath() + "textures/cubemap_space.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
}

void prepare()
{
    // Call base class preparation (creates window, instance, device, swapchain, etc.)
    VulkanExampleBase::prepare();
    
    // Load 3D models and textures
    loadAssets();
    
    // Create uniform buffers for scene transformations and parameters
    prepareUniformBuffers();
    
    // Set up offscreen framebuffers for the bloom effect
    prepareOffscreen();
    
    // Create descriptor sets to bind resources to shaders
    setupDescriptors();
    
    // Create graphics pipelines
    preparePipelines();
    
    // Build command buffers for rendering
    buildCommandBuffers();
    
    // Mark everything as prepared so rendering can begin
    prepared = true;
}
```

## Updating the uniform buffers in loop

A missing piece is updating the uniform buffers for the scene:

```cpp
// Update uniform buffers for rendering the 3D scene
void updateUniformBuffersScene()
{
   // STEP 1: Update UFO model matrices
   // Set projection matrix from camera perspective
   ubos.scene.projection = camera.matrices.perspective;
   
   // Set view matrix from camera position and orientation
   ubos.scene.view = camera.matrices.view;
   
   // Create model matrix for UFO
   // First translate based on timer animation
   ubos.scene.model = glm::translate(glm::mat4(1.0f), 
       glm::vec3(
           sin(glm::radians(timer * 360.0f)) * 0.25f,  // X oscillation
           -1.0f,                                      // Fixed Y position
           cos(glm::radians(timer * 360.0f)) * 0.25f   // Z oscillation
       ));
   
   // Add some rotation around X axis based on position
   ubos.scene.model = glm::rotate(
       ubos.scene.model,                              // Current matrix
       -sinf(glm::radians(timer * 360.0f)) * 0.15f,   // Rotation angle
       glm::vec3(1.0f, 0.0f, 0.0f));                  // Rotation axis (X)
   
   // Add continuous rotation around Y axis
   ubos.scene.model = glm::rotate(
       ubos.scene.model,                              // Current matrix
       glm::radians(timer * 360.0f),                  // Rotation angle
       glm::vec3(0.0f, 1.0f, 0.0f));                  // Rotation axis (Y)
   
   // Copy the updated matrices to the GPU
   memcpy(uniformBuffers.scene.mapped, &ubos.scene, sizeof(ubos.scene));
   
   // STEP 2: Update skybox matrices
   // Use a different projection for the skybox (fixed FOV)
   ubos.skyBox.projection = glm::perspective(
       glm::radians(45.0f),                           // 45-degree FOV
       (float)width / (float)height,                  // Aspect ratio
       0.1f,                                          // Near plane
       256.0f);                                       // Far plane
   
   // For the skybox, we only use the rotation part of the view matrix
   // This ensures the skybox rotates with the camera but doesn't translate
   ubos.skyBox.view = glm::mat4(glm::mat3(camera.matrices.view));
   
   // Use identity model matrix for skybox (centered on camera)
   ubos.skyBox.model = glm::mat4(1.0f);
   
   // Copy the updated skybox matrices to the GPU
   memcpy(uniformBuffers.skyBox.mapped, &ubos.skyBox, sizeof(ubos.skyBox));
}
```

## Rendering

We call our trustful `render()` and `draw()` functions:

```cpp
// Submit command buffer to the graphics queue for execution
void draw()
{
    // Prepare the next frame (acquire next swap chain image)
    VulkanExampleBase::prepareFrame();
    
    // Submit the command buffer
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    
    // Present the frame to the screen
    VulkanExampleBase::submitFrame();
}

// Main render function called every frame
virtual void render()
{
    // Skip if the application is not fully prepared
    if (!prepared)
        return;
        
    // Render the scene
    draw();
    
    // Only update uniforms if the scene is not paused or the camera moved
    if (!paused || camera.updated)
    {
        // Update scene transformation matrices for the next frame
        updateUniformBuffersScene();
    }
}
```

## Controlling Bloom Parameters

We use a uniform buffer to control bloom parameters:

```cpp
// Bloom effect parameters
struct UBOBlurParams {
    float blurScale = 1.0f;      // Controls blur sample distance
    float blurStrength = 1.5f;   // Controls blur intensity
};

// Update blur parameters in the uniform buffer
void updateUniformBuffersBlur()
{
    // Simply copy the current blur parameters to the GPU
    // We do this whenever the UI controls change these values
    memcpy(uniformBuffers.blurParams.mapped, &ubos.blurParams, sizeof(ubos.blurParams));
}

// UI overlay update function
virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
{
    if (overlay->header("Settings")) {
        // Toggle for enabling/disabling bloom effect
        if (overlay->checkBox("Bloom", &bloom)) {
            // When toggling bloom, rebuild command buffers to either include or exclude bloom passes
            buildCommandBuffers();
        }
        
        // Slider for controlling the blur scale (spread)
        if (overlay->inputFloat("Scale", &ubos.blurParams.blurScale, 0.1f, 2)) {
            // When changing blur scale, update the uniform buffer
            updateUniformBuffersBlur();
        }
    }
}
```

## How It All Works Together

To summarize, here's how the bloom effect works in our implementation:

1. **First Pass**: We render only the glowing parts of our scene (in this case, a separate mesh model.ufoGlow) to an
   offscreen framebuffer.

2. **Second Pass (Vertical Blur)**: We apply a vertical Gaussian blur to the bright parts by sampling the texture with
   specific weights along the y-axis and write the result to another framebuffer.

3. **Final Pass**:
   a. We render the normal scene with regular lighting.
   b. We apply a horizontal blur to the already vertically blurred texture, sampling along the x-axis.
   c. We composite this result over the scene using additive blending to create the bloom effect.

By separating the blur into two passes (vertical and horizontal), we achieve a proper 2D Gaussian blur with far fewer
texture samples compared to a single-pass approach, making it much more efficient.

## Tips for Improving Your Bloom Effect

1. **Adjust the brightness threshold**: In a more sophisticated implementation, you might want to extract only pixels
   above a certain brightness threshold. This can be done by sampling the scene color and checking if its luminance is
   above a threshold.

   ```glsl
   // Example fragment shader code for brightness extraction
   vec3 color = texture(sceneTexture, inUV).rgb;
   float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722)); // Luminance calculation
   if (brightness > THRESHOLD) {
       outFragColor = vec4(color, 1.0);
   } else {
       outFragColor = vec4(0.0, 0.0, 0.0, 1.0);
   }
   ```

2. **Use multiple blur passes with different kernel sizes**: For more advanced bloom effects, consider using multiple
   blur passes with different kernel sizes to create a more natural-looking bloom effect.

   ```cpp
   // Example of how to set up multiple blur passes
   struct BloomPass {
       float scale;
       float strength;
       FrameBuffer framebuffer;
   };
   std::vector<BloomPass> bloomPasses = {
       {1.0f, 1.0f, framebuffer1},
       {2.0f, 0.5f, framebuffer2},
       {4.0f, 0.25f, framebuffer3}
   };
   ```

3. **HDR rendering**: Bloom works best in a high dynamic range rendering pipeline. Consider implementing HDR rendering
   for even better results.

   ```glsl
   // In your final composition shader, you could implement tone mapping
   vec3 hdrColor = /* ... */;
   // Apply tone mapping (e.g., Reinhard)
   vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
   // Apply gamma correction
   mapped = pow(mapped, vec3(1.0 / 2.2));
   outFragColor = vec4(mapped, 1.0);
   ```

4. **Performance optimization**: Consider using a lower resolution for the bloom effect. You can render to a smaller
   offscreen framebuffer (e.g., half the screen size) and then upsample when compositing. This can significantly improve
   performance with minimal quality loss.

   ```cpp
   // Example of using smaller framebuffers for bloom
   #define BLOOM_SCALE 0.5f
   offscreenPass.width = width * BLOOM_SCALE;
   offscreenPass.height = height * BLOOM_SCALE;
   ```

5. **Tone mapping**: If implementing HDR, be sure to apply proper tone mapping after adding bloom to ensure your final
   image looks correct on standard displays.

   ```glsl
   // Sample tone mapping implementation (ACES filmic)
   vec3 ACESFilm(vec3 x) {
       float a = 2.51f;
       float b = 0.03f;
       float c = 2.43f;
       float d = 0.59f;
       float e = 0.14f;
       return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
   }
   ```

## Conclusion

Implementing bloom in Vulkan requires a good understanding of offscreen rendering, multiple render passes, and shader
programming. The multi-pass Gaussian blur approach provides a high-quality bloom effect while maintaining good
performance.

Experiment with different parameters, combine bloom with other effects like HDR rendering and tone mapping, and see how
it can transform the look and feel of your 3D scenes.
