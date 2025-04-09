# SSAO (Screen Space Ambient Occlusion)

Screen Space Ambient Occlusion (SSAO) is a rendering technique that approximates how light is occluded by nearby geometry. It adds depth and realism to 3D scenes by darkening areas where objects are close together (like corners, crevices, and areas under objects).

This document breaks down a Vulkan SSAO implementation, including both the C++ code and shader components, explaining each step in detail for beginners. The full code for each component is included to provide a complete understanding of the implementation.

## Table of Contents
1. [Overview of SSAO](#overview-of-ssao)
2. [Implementation Pipeline](#implementation-pipeline)
3. [C++ Implementation](#c-implementation)
   - [Data Structures and Constants](#data-structures-and-constants)
   - [G-Buffer Creation](#g-buffer-creation)
   - [SSAO Kernel and Noise Texture](#ssao-kernel-and-noise-texture)
   - [Render Pipeline Setup](#render-pipeline-setup)
   - [Command Buffer Building](#command-buffer-building)
4. [Shader Implementation](#shader-implementation)
   - [G-Buffer Pass](#g-buffer-pass)
   - [SSAO Pass](#ssao-pass)
   - [Blur Pass](#blur-pass)
   - [Composition Pass](#composition-pass)
5. [Complete Rendering Process](#complete-rendering-process)

## Overview of SSAO

SSAO adds depth and realism to 3D scenes by darkening corners, crevices, and areas where objects are close together. It simulates how ambient light is blocked in certain areas:

- Areas that are more occluded (like crevices, corners) receive less ambient light
- Areas that are more exposed receive more ambient light

The key insight is that if we sample points in a hemisphere around each pixel and check if those points are occluded by other geometry, we can approximate how much ambient light that pixel should receive.

## Implementation Pipeline

This SSAO implementation consists of four main passes:

1. **G-Buffer Pass**: Renders scene information to textures
2. **SSAO Pass**: Calculates occlusion values using the G-Buffer data
3. **Blur Pass**: Smooths the SSAO results to remove noise
4. **Composition Pass**: Combines everything for the final image

## C++ Implementation

Let's examine key components of the C++ implementation:

### Data Structures and Constants

```cpp
// Constants that define SSAO quality and appearance
#define SSAO_KERNEL_SIZE 64    // Number of sample points in the hemisphere
                              // Higher = better quality but slower performance
#define SSAO_RADIUS 0.3f      // Radius of sample hemisphere
                              // Controls how far SSAO looks for occluders

// Platform-specific noise texture size (smaller on mobile for performance)
#if defined(__ANDROID__)
#define SSAO_NOISE_DIM 4      // 4x4 noise texture on mobile (16 pixels)
#else
#define SSAO_NOISE_DIM 8      // 8x8 noise texture on desktop (64 pixels)
#endif

class VulkanExample : public VulkanExampleBase
{
public:
    // Noise texture used to rotate the sampling kernel randomly
    // This helps prevent banding artifacts in the final SSAO
    vks::Texture2D ssaoNoise;
    
    // 3D scene model to render
    vkglTF::Model scene;

    // Uniform buffer for scene parameters - sent to shaders
    struct UBOSceneParams {
        glm::mat4 projection;  // Camera projection matrix
        glm::mat4 model;       // Model matrix (object to world space)
        glm::mat4 view;        // View matrix (world to camera space)
        float nearPlane = 0.1f;  // Camera near clip plane
        float farPlane = 64.0f;  // Camera far clip plane
    } uboSceneParams;

    // Uniform buffer for SSAO-specific parameters
    struct UBOSSAOParams {
        glm::mat4 projection;   // Camera projection matrix (needed in SSAO shader)
        int32_t ssao = true;    // Toggle SSAO effect on/off (UI control)
        int32_t ssaoOnly = false; // Show only the SSAO effect (UI control)
        int32_t ssaoBlur = true;  // Enable/disable blur on SSAO (UI control)
    } uboSSAOParams;

    // Graphics pipelines for each rendering pass
    struct {
        VkPipeline offscreen{ VK_NULL_HANDLE };  // G-Buffer pass pipeline
        VkPipeline composition{ VK_NULL_HANDLE }; // Final composition pipeline
        VkPipeline ssao{ VK_NULL_HANDLE };        // SSAO generation pipeline
        VkPipeline ssaoBlur{ VK_NULL_HANDLE };    // SSAO blur pipeline
    } pipelines;

    // Pipeline layouts (define what uniform variables each shader has access to)
    struct {
        VkPipelineLayout gBuffer{ VK_NULL_HANDLE };
        VkPipelineLayout ssao{ VK_NULL_HANDLE };
        VkPipelineLayout ssaoBlur{ VK_NULL_HANDLE };
        VkPipelineLayout composition{ VK_NULL_HANDLE };
    } pipelineLayouts;

    // Descriptor sets (bind resources like textures/buffers to shaders)
    struct {
        VkDescriptorSet gBuffer{ VK_NULL_HANDLE };
        VkDescriptorSet ssao{ VK_NULL_HANDLE };
        VkDescriptorSet ssaoBlur{ VK_NULL_HANDLE };
        VkDescriptorSet composition{ VK_NULL_HANDLE };
        const uint32_t count = 4; // Total number of descriptor sets
    } descriptorSets;

    // Descriptor set layouts (define what types of resources each shader expects)
    struct {
        VkDescriptorSetLayout gBuffer{ VK_NULL_HANDLE };
        VkDescriptorSetLayout ssao{ VK_NULL_HANDLE };
        VkDescriptorSetLayout ssaoBlur{ VK_NULL_HANDLE };
        VkDescriptorSetLayout composition{ VK_NULL_HANDLE };
    } descriptorSetLayouts;

    // Uniform buffers used across the rendering pipeline
    struct {
        vks::Buffer sceneParams;  // Contains matrices and camera settings
        vks::Buffer ssaoKernel;   // Contains sample hemisphere points
        vks::Buffer ssaoParams;   // Contains SSAO-specific settings
    } uniformBuffers;

    // Frame buffer attachment - a single texture in a framebuffer
    struct FrameBufferAttachment {
        VkImage image;         // Vulkan image handle
        VkDeviceMemory mem;    // GPU memory for this image
        VkImageView view;      // Image view for accessing in shaders
        VkFormat format;       // Pixel format (e.g., RGBA8)
        
        // Clean up resources
        void destroy(VkDevice device)
        {
            vkDestroyImage(device, image, nullptr);
            vkDestroyImageView(device, view, nullptr);
            vkFreeMemory(device, mem, nullptr);
        }
    };
    
    // Frame buffer - a collection of attachments for a render pass
    struct FrameBuffer {
        int32_t width, height;  // Dimensions of this framebuffer
        VkFramebuffer frameBuffer; // Vulkan framebuffer handle
        VkRenderPass renderPass;   // Render pass compatible with this framebuffer
        
        // Set framebuffer size
        void setSize(int32_t w, int32_t h)
        {
            this->width = w;
            this->height = h;
        }
        
        // Clean up resources
        void destroy(VkDevice device)
        {
            vkDestroyFramebuffer(device, frameBuffer, nullptr);
            vkDestroyRenderPass(device, renderPass, nullptr);
        }
    };

    // Collection of all framebuffers used in the SSAO pipeline
    struct {
        // G-Buffer framebuffer (stores position, normal, albedo, depth)
        struct Offscreen : public FrameBuffer {
            FrameBufferAttachment position; // Position + depth
            FrameBufferAttachment normal;   // Surface normals
            FrameBufferAttachment albedo;   // Color information
            FrameBufferAttachment depth;    // Depth buffer
        } offscreen;
        
        // SSAO and blur framebuffers (each has just one color attachment)
        struct SSAO : public FrameBuffer {
            FrameBufferAttachment color;    // The actual SSAO/blur value
        } ssao, ssaoBlur;
    } frameBuffers{};

	// One sampler for the frame buffer color attachments
	VkSampler colorSampler;
    ...
```

These data structures define the parameters that control the SSAO effect. The uniform buffers (UBO) will be passed to the shaders.

### G-Buffer Creation

The G-Buffer consists of multiple textures (also called "attachments"):

```cpp
// Create a single framebuffer attachment (texture)
void createAttachment(
    VkFormat format,              // Pixel format of the texture
    VkImageUsageFlagBits usage,   // How the texture will be used
    FrameBufferAttachment *attachment,
    uint32_t width,               // Texture width
    uint32_t height)              // Texture height
{
    // Determine the aspect mask based on usage
    // (color attachments vs depth attachments have different aspects)
    VkImageAspectFlags aspectMask = 0;
    attachment->format = format;

    // Set the correct aspect mask based on whether this is a color or depth attachment
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        // If format includes stencil, add stencil aspect too
        if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
            aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    // Make sure we have a valid aspect mask
    assert(aspectMask > 0);

    // Create the image (the actual texture storage)
    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;         // 2D texture
    image.format = format;                      // Pixel format
    image.extent.width = width;                 // Texture width
    image.extent.height = height;               // Texture height
    image.extent.depth = 1;                     // Not a 3D texture
    image.mipLevels = 1;                        // No mipmaps
    image.arrayLayers = 1;                      // Not a texture array
    image.samples = VK_SAMPLE_COUNT_1_BIT;      // No multisampling
    image.tiling = VK_IMAGE_TILING_OPTIMAL;     // Let GPU optimize layout
    // Allow this image to be both a render target AND sampled in shaders
    image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

    // Allocate memory for the image
    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    // Create the image
    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
    // Get memory requirements for this image type
    vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    // Get memory type index that supports these requirements
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Allocate the memory
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
    // Bind memory to the image
    VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

    // Create an image view to access the image
    VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
    imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageView.format = format;
    imageView.subresourceRange = {};
    imageView.subresourceRange.aspectMask = aspectMask;
    imageView.subresourceRange.baseMipLevel = 0;
    imageView.subresourceRange.levelCount = 1;
    imageView.subresourceRange.baseArrayLayer = 0;
    imageView.subresourceRange.layerCount = 1;
    imageView.image = attachment->image;
    VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
}

// Set up all framebuffers for the SSAO pipeline
void prepareOffscreenFramebuffers()
{
    // Set appropriate resolution for SSAO
    // On mobile, we use half resolution for SSAO to improve performance
    #if defined(__ANDROID__)
    const uint32_t ssaoWidth = width / 2;
    const uint32_t ssaoHeight = height / 2;
    #else
    const uint32_t ssaoWidth = width;
    const uint32_t ssaoHeight = height;
    #endif

    // Set framebuffer sizes
    frameBuffers.offscreen.setSize(width, height);        // G-Buffer at full resolution
    frameBuffers.ssao.setSize(ssaoWidth, ssaoHeight);     // SSAO may be at reduced resolution
    frameBuffers.ssaoBlur.setSize(width, height);         // Blur back at full resolution

    // Find an appropriate depth format that's supported by the device
    VkFormat attDepthFormat;
    VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
    assert(validDepthFormat);

    // Create G-Buffer attachments
    // Position buffer - stores 3D position and depth (RGBA32F format)
    createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                    &frameBuffers.offscreen.position, width, height);
    
    // Normal buffer - stores surface normals (RGBA8 format)
    createAttachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                    &frameBuffers.offscreen.normal, width, height);
    
    // Albedo buffer - stores color information (RGBA8 format)
    createAttachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                    &frameBuffers.offscreen.albedo, width, height);
    
    // Depth buffer - stores depth information
    createAttachment(attDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
                    &frameBuffers.offscreen.depth, width, height);

    // SSAO buffer - stores raw ambient occlusion (R8 format - only need one channel)
    createAttachment(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                    &frameBuffers.ssao.color, ssaoWidth, ssaoHeight);

    // SSAO blur buffer - stores blurred ambient occlusion (R8 format)
    createAttachment(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                    &frameBuffers.ssaoBlur.color, width, height);

    // Create render passes for each framebuffer
    
    // G-Buffer render pass setup - outputs to position, normal, and albedo attachments
    {
        // We need 4 attachments: position, normal, albedo, and depth
        std::array<VkAttachmentDescription, 4> attachmentDescs = {};

        // Set up common properties for all attachments
        for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++)
        {
            attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;         // No multisampling
            attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;    // Clear at start
            attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // Keep contents after render
            attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // No stencil
            attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            
            // Set final layout based on attachment type
            // Depth attachment gets special layout for depth testing
            // Others get shader read-only for sampling in SSAO pass
            attachmentDescs[i].finalLayout = (i == 3) ? 
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : 
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Set attachment formats
        attachmentDescs[0].format = frameBuffers.offscreen.position.format;
        attachmentDescs[1].format = frameBuffers.offscreen.normal.format;
        attachmentDescs[2].format = frameBuffers.offscreen.albedo.format;
        attachmentDescs[3].format = frameBuffers.offscreen.depth.format;

        // Define color attachment references
        std::vector<VkAttachmentReference> colorReferences;
        colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
        colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
        colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

        // Define depth attachment reference
        VkAttachmentReference depthReference = {};
        depthReference.attachment = 3;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // Define the single subpass that will render to all G-Buffer attachments
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pColorAttachments = colorReferences.data();
        subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
        subpass.pDepthStencilAttachment = &depthReference;

        // Set up subpass dependencies for correct transition timing
        std::array<VkSubpassDependency, 2> dependencies;

        // Wait for previous fragment shader to finish before starting new render
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Wait for render to finish before next fragment shader reads attachments
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Create the G-Buffer render pass
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.pAttachments = attachmentDescs.data();
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 2;
        renderPassInfo.pDependencies = dependencies.data();
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &frameBuffers.offscreen.renderPass));

        // Create framebuffer for G-Buffer render pass
        std::array<VkImageView, 4> attachments;
        attachments[0] = frameBuffers.offscreen.position.view;
        attachments[1] = frameBuffers.offscreen.normal.view;
        attachments[2] = frameBuffers.offscreen.albedo.view;
        attachments[3] = frameBuffers.offscreen.depth.view;

        VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
        fbufCreateInfo.renderPass = frameBuffers.offscreen.renderPass;
        fbufCreateInfo.pAttachments = attachments.data();
        fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbufCreateInfo.width = frameBuffers.offscreen.width;
        fbufCreateInfo.height = frameBuffers.offscreen.height;
        fbufCreateInfo.layers = 1;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuffers.offscreen.frameBuffer));
    }

    // SSAO render pass setup - outputs occlusion values
    {
        // Only need one attachment for SSAO pass
        VkAttachmentDescription attachmentDescription{};
        attachmentDescription.format = frameBuffers.ssao.color.format;
        attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Define color attachment reference
        VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        // Define the single subpass
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pColorAttachments = &colorReference;
        subpass.colorAttachmentCount = 1;

        // Set up dependencies similar to G-Buffer
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Create the SSAO render pass
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.pAttachments = &attachmentDescription;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 2;
        renderPassInfo.pDependencies = dependencies.data();
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &frameBuffers.ssao.renderPass));

        // Create framebuffer for SSAO render pass
        VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
        fbufCreateInfo.renderPass = frameBuffers.ssao.renderPass;
        fbufCreateInfo.pAttachments = &frameBuffers.ssao.color.view;
        fbufCreateInfo.attachmentCount = 1;
        fbufCreateInfo.width = frameBuffers.ssao.width;
        fbufCreateInfo.height = frameBuffers.ssao.height;
        fbufCreateInfo.layers = 1;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuffers.ssao.frameBuffer));
    }

    // SSAO Blur render pass setup - smooths the raw SSAO output
    // [Code is similar to SSAO render pass, but with blur framebuffer]
    {
        // Similar setup to SSAO pass, but for blur buffer
        // ... (code omitted as it's nearly identical to SSAO pass)
    }

    // Create sampler used for reading from the attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_NEAREST;  // Using nearest filter for accurate reads
    sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &colorSampler));
}
```

This code creates several textures:
- **Position**: Stores 3D position and depth of each pixel
- **Normal**: Stores surface normal vectors
- **Albedo**: Stores base color (diffuse)
- **SSAO**: Stores the raw ambient occlusion values
- **SSAO Blur**: Stores the smoothed ambient occlusion values

### SSAO Kernel and Noise Texture

```cpp
// Helper function for linear interpolation
float lerp(float a, float b, float f)
{
    return a + f * (b - a);
}

// Create uniform buffers for SSAO parameters and sample points
void prepareUniformBuffers()
{
    // Create scene parameters uniform buffer
    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &uniformBuffers.sceneParams,
        sizeof(uboSceneParams));

    // Create SSAO parameters uniform buffer
    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &uniformBuffers.ssaoParams,
        sizeof(uboSSAOParams));

    // Update uniform buffers with initial values
    updateUniformBufferMatrices();
    updateUniformBufferSSAOParams();

    // Create random engine with consistent seed if benchmarking
    std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned)time(nullptr));
    // Create uniform distribution between 0.0 and 1.0
    std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

    // Generate SSAO sample kernel (a set of random points in a hemisphere)
    std::vector<glm::vec4> ssaoKernel(SSAO_KERNEL_SIZE);
    for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i)
    {
        // Step 1: Create a random point inside a unit cube, with Z positive
        // Mapping random values from [0,1] to [-1,1] for X,Y
        // Leaving Z as [0,1] to keep points in front hemisphere
        glm::vec3 sample(rndDist(rndEngine) * 2.0 - 1.0, 
                         rndDist(rndEngine) * 2.0 - 1.0, 
                         rndDist(rndEngine));
        
        // Step 2: Normalize to get a point on the hemisphere
        sample = glm::normalize(sample);
        
        // Step 3: Scale the point by a random amount (keeps some points closer to center)
        sample *= rndDist(rndEngine);
        
        // Step 4: Apply acceleration function to bias points toward center
        // This places more samples near the fragment for better quality
        float scale = float(i) / float(SSAO_KERNEL_SIZE);
        
        // Lerp factor increases quadratically with i
        // This creates a non-linear distribution with more samples near origin
        scale = lerp(0.1f, 1.0f, scale * scale);
        
        // Store the sample point in our kernel (use w=0.0 for padding)
        ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
    }

    // Upload kernel to GPU
    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,  // Will be used as a uniform buffer
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &uniformBuffers.ssaoKernel,          // Buffer to store it in
        ssaoKernel.size() * sizeof(glm::vec4),  // Size in bytes
        ssaoKernel.data());                  // Initial data

    // Create random noise texture for rotating the sample kernel
    // The noise texture helps break up banding patterns and repeating artifacts
    std::vector<glm::vec4> noiseValues(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
    for (uint32_t i = 0; i < static_cast<uint32_t>(noiseValues.size()); i++) {
        // Create random vectors in tangent space (X,Y components only)
        // Z and W are set to 0, as we only need a 2D rotation
        noiseValues[i] = glm::vec4(
            rndDist(rndEngine) * 2.0f - 1.0f,  // Random X in [-1,1]
            rndDist(rndEngine) * 2.0f - 1.0f,  // Random Y in [-1,1]
            0.0f, 0.0f);                      // Z and W set to 0
    }
    
    // Upload noise as texture
    // The texture will be tiled across the screen
    ssaoNoise.fromBuffer(
        noiseValues.data(),                     // Noise data
        noiseValues.size() * sizeof(glm::vec4), // Size in bytes
        VK_FORMAT_R32G32B32A32_SFLOAT,          // 32-bit float format for precision
        SSAO_NOISE_DIM, SSAO_NOISE_DIM,         // Dimensions of noise texture 
        vulkanDevice, queue,                    // Vulkan handles
        VK_FILTER_NEAREST);                     // Use nearest filtering for exact values
}

 void updateUniformBufferMatrices()
 {
     uboSceneParams.projection = camera.matrices.perspective;
     uboSceneParams.view = camera.matrices.view;
     uboSceneParams.model = glm::mat4(1.0f);

     VK_CHECK_RESULT(uniformBuffers.sceneParams.map());
     uniformBuffers.sceneParams.copyTo(&uboSceneParams, sizeof(uboSceneParams));
     uniformBuffers.sceneParams.unmap();
 }

 void updateUniformBufferSSAOParams()
 {
     uboSSAOParams.projection = camera.matrices.perspective;

     VK_CHECK_RESULT(uniformBuffers.ssaoParams.map());
     uniformBuffers.ssaoParams.copyTo(&uboSSAOParams, sizeof(uboSSAOParams));
     uniformBuffers.ssaoParams.unmap();
 }
```

This code creates two important elements:

1. **Sample Kernel**: A set of random points distributed in a hemisphere. These points will be used as sample positions around each pixel. Note how the distribution is biased toward the center of the hemisphere, which gives better results for SSAO.

2. **Noise Texture**: A small texture of random vectors used to rotate the sample kernel for each pixel. This helps reduce banding artifacts in the final SSAO result.

### Render Pipeline Setup

Before we look at the complete rendering process, let's examine how the pipelines for each pass are created:

```cpp
void preparePipelines()
{
    // Create pipeline layouts (define what uniform variables each shader has access to)
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo();

    // G-Buffer pipeline layout (includes both our UBO layout and textures from the GLTF model)
    const std::vector<VkDescriptorSetLayout> setLayouts = { descriptorSetLayouts.gBuffer, vkglTF::descriptorSetLayoutImage };
    pipelineLayoutCreateInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutCreateInfo.setLayoutCount = 2;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.gBuffer));

    // SSAO pipeline layout
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.ssao;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.ssao));

    // SSAO blur pipeline layout
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.ssaoBlur;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.ssaoBlur));

    // Composition pipeline layout
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.composition;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.composition));

    // Setup common pipeline state variables
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // Base pipeline creation info
    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo( pipelineLayouts.composition, renderPass, 0);
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // Empty vertex input state for fullscreen passes (SSAO, blur, composition)
    VkPipelineVertexInputStateCreateInfo emptyVertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCreateInfo.pVertexInputState = &emptyVertexInputState;
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;

    // Final composition pipeline
    shaderStages[0] = loadShader(getShadersPath() + "ssao/fullscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "ssao/composition.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.composition));

    // SSAO generation pipeline
    pipelineCreateInfo.renderPass = frameBuffers.ssao.renderPass;
    pipelineCreateInfo.layout = pipelineLayouts.ssao;
    
    // SSAO Kernel size and radius are constant for this pipeline, so we set them using specialization constants
    struct SpecializationData {
        uint32_t kernelSize = SSAO_KERNEL_SIZE;
        float radius = SSAO_RADIUS;
    } specializationData;
    std::array<VkSpecializationMapEntry, 2> specializationMapEntries = {
        vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, kernelSize), sizeof(SpecializationData::kernelSize)),
        vks::initializers::specializationMapEntry(1, offsetof(SpecializationData, radius), sizeof(SpecializationData::radius))
    };
    VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(2, specializationMapEntries.data(), sizeof(specializationData), &specializationData);
    shaderStages[1] = loadShader(getShadersPath() + "ssao/ssao.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.ssao));

    // SSAO blur pipeline
    pipelineCreateInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
    pipelineCreateInfo.layout = pipelineLayouts.ssaoBlur;
    shaderStages[1] = loadShader(getShadersPath() + "ssao/blur.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.ssaoBlur));

    // Fill G-Buffer pipeline
    // Vertex input state from glTF model loader
    pipelineCreateInfo.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal });
    pipelineCreateInfo.renderPass = frameBuffers.offscreen.renderPass;
    pipelineCreateInfo.layout = pipelineLayouts.gBuffer;
    
    // Blend attachment states required for all color attachments
    // This is important, as color write mask will otherwise be 0x0 and you
    // won't see anything rendered to the attachment
    std::array<VkPipelineColorBlendAttachmentState, 3> blendAttachmentStates = {
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE)
    };
    colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
    colorBlendState.pAttachments = blendAttachmentStates.data();
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    shaderStages[0] = loadShader(getShadersPath() + "ssao/gbuffer.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "ssao/gbuffer.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.offscreen));
}
```

This function creates all the rendering pipelines:

1. **G-Buffer Pipeline**: Used to render the scene into position, normal, and color textures
2. **SSAO Pipeline**: Processes the G-Buffer to calculate ambient occlusion
3. **Blur Pipeline**: Applies blur to the SSAO result
4. **Composition Pipeline**: Combines everything for the final image

Notice the use of specialization constants for the SSAO pipeline - these allow compile-time optimization of the shader based on our SSAO kernel size and radius.

### Rest of the preparation

```cpp


void setupDescriptors()
{
   // Pool
   std::vector<VkDescriptorPoolSize> poolSizes = {
   vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10),
   vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12)
   };
   VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes,  descriptorSets.count);
   VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
   
   std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
   VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo;
   VkDescriptorSetAllocateInfo descriptorAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, nullptr, 1);
   std::vector<VkWriteDescriptorSet> writeDescriptorSets;
   std::vector<VkDescriptorImageInfo> imageDescriptors;
   
   // Layouts and Sets
   
   // G-Buffer creation (offscreen scene rendering)
   setLayoutBindings = {
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),	// VS + FS Parameter UBO
   };
   setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
   VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.gBuffer));
   
   descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.gBuffer;
   VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.gBuffer));
   writeDescriptorSets = {
   vks::initializers::writeDescriptorSet(descriptorSets.gBuffer, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.sceneParams.descriptor),
   };
   vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
   
   // SSAO Generation
   setLayoutBindings = {
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Position+Depth
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),						// FS Normals
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),						// FS SSAO Noise
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),								// FS SSAO Kernel UBO
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),								// FS Params UBO
   };
   setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
   VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.ssao));
   
   descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.ssao;
   VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.ssao));
   imageDescriptors = {
   vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
   vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
   };
   writeDescriptorSets = {
   vks::initializers::writeDescriptorSet(descriptorSets.ssao, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]),					// FS Position+Depth
   vks::initializers::writeDescriptorSet(descriptorSets.ssao, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageDescriptors[1]),					// FS Normals
   vks::initializers::writeDescriptorSet(descriptorSets.ssao, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &ssaoNoise.descriptor),		// FS SSAO Noise
   vks::initializers::writeDescriptorSet(descriptorSets.ssao, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, &uniformBuffers.ssaoKernel.descriptor),		// FS SSAO Kernel UBO
   vks::initializers::writeDescriptorSet(descriptorSets.ssao, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, &uniformBuffers.ssaoParams.descriptor),		// FS SSAO Params UBO
   };
   vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
   
   // SSAO Blur
   setLayoutBindings = {
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Sampler SSAO
   };
   setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
   VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.ssaoBlur));
   descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.ssaoBlur;
   VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.ssaoBlur));
   imageDescriptors = {
   vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssao.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
   };
   writeDescriptorSets = {
   vks::initializers::writeDescriptorSet(descriptorSets.ssaoBlur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]),
   };
   vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
   
   // Composition
   setLayoutBindings = {
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),						// FS Position+Depth
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),						// FS Normals
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),						// FS Albedo
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),						// FS SSAO
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),						// FS SSAO blurred
   vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 5),								// FS Lights UBO
   };
   
   setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
   VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorSetLayouts.composition));
   descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.composition;
   VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorAllocInfo, &descriptorSets.composition));
   
   imageDescriptors = {
      vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.position.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssao.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssaoBlur.color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
   };
   writeDescriptorSets = {
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageDescriptors[0]),			// FS Sampler Position+Depth
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageDescriptors[1]),			// FS Sampler Normals
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &imageDescriptors[2]),			// FS Sampler Albedo
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &imageDescriptors[3]),			// FS Sampler SSAO
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, &imageDescriptors[4]),			// FS Sampler SSAO blurred
      vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5, &uniformBuffers.ssaoParams.descriptor),	// FS SSAO Params UBO
   };
   vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

   void loadAssets()
   {
        vkglTF::descriptorBindingFlags  = vkglTF::DescriptorBindingFlags::ImageBaseColor;
      const uint32_t gltfLoadingFlags = vkglTF::FileLoadingFlags::FlipY | vkglTF::FileLoadingFlags::PreTransformVertices;
      scene.loadFromFile(getAssetPath() + "models/sponza/sponza.gltf", vulkanDevice, queue, gltfLoadingFlags);
   }

   float lerp(float a, float b, float f)
   {
       return a + f * (b - a);
   }

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareOffscreenFramebuffers();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}
```

### Command Buffer Build

The main rendering loop in `buildCommandBuffers()` executes all passes in sequence:

```cpp
void buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        /*
            Offscreen SSAO generation
        */
        {
            // Clear values for all attachments written in the fragment shader
            std::vector<VkClearValue> clearValues(4);
            clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[3].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
            renderPassBeginInfo.renderPass = frameBuffers.offscreen.renderPass;
            renderPassBeginInfo.framebuffer = frameBuffers.offscreen.frameBuffer;
            renderPassBeginInfo.renderArea.extent.width = frameBuffers.offscreen.width;
            renderPassBeginInfo.renderArea.extent.height = frameBuffers.offscreen.height;
            renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassBeginInfo.pClearValues = clearValues.data();

            /*
                First pass: Fill G-Buffer components (positions+depth, normals, albedo) using MRT
            */

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)frameBuffers.offscreen.width, (float)frameBuffers.offscreen.height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(frameBuffers.offscreen.width, frameBuffers.offscreen.height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.gBuffer, 0, 1, &descriptorSets.gBuffer, 0, nullptr);
            scene.draw(drawCmdBuffers[i], vkglTF::RenderFlags::BindImages, pipelineLayouts.gBuffer);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            /*
                Second pass: SSAO generation
            */

            clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[1].depthStencil = { 1.0f, 0 };

            renderPassBeginInfo.framebuffer = frameBuffers.ssao.frameBuffer;
            renderPassBeginInfo.renderPass = frameBuffers.ssao.renderPass;
            renderPassBeginInfo.renderArea.extent.width = frameBuffers.ssao.width;
            renderPassBeginInfo.renderArea.extent.height = frameBuffers.ssao.height;
            renderPassBeginInfo.clearValueCount = 2;
            renderPassBeginInfo.pClearValues = clearValues.data();

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            viewport = vks::initializers::viewport((float)frameBuffers.ssao.width, (float)frameBuffers.ssao.height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
            scissor = vks::initializers::rect2D(frameBuffers.ssao.width, frameBuffers.ssao.height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ssao, 0, 1, &descriptorSets.ssao, 0, nullptr);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.ssao);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            /*
                Third pass: SSAO blur
            */

            renderPassBeginInfo.framebuffer = frameBuffers.ssaoBlur.frameBuffer;
            renderPassBeginInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
            renderPassBeginInfo.renderArea.extent.width = frameBuffers.ssaoBlur.width;
            renderPassBeginInfo.renderArea.extent.height = frameBuffers.ssaoBlur.height;

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            viewport = vks::initializers::viewport((float)frameBuffers.ssaoBlur.width, (float)frameBuffers.ssaoBlur.height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
            scissor = vks::initializers::rect2D(frameBuffers.ssaoBlur.width, frameBuffers.ssaoBlur.height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ssaoBlur, 0, 1, &descriptorSets.ssaoBlur, 0, nullptr);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.ssaoBlur);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

            vkCmdEndRenderPass(drawCmdBuffers[i]);
        }

        /*
            Note: Explicit synchronization is not required between the render passes,
            as this is done implicitly via subpass dependencies
        */

        /*
            Final render pass: Scene rendering with applied SSAO
        */
        {
            std::vector<VkClearValue> clearValues(2);
            clearValues[0].color = defaultClearColor;
            clearValues[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
            renderPassBeginInfo.renderPass = renderPass;
            renderPassBeginInfo.framebuffer = VulkanExampleBase::frameBuffers[i];
            renderPassBeginInfo.renderArea.extent.width = width;
            renderPassBeginInfo.renderArea.extent.height = height;
            renderPassBeginInfo.clearValueCount = 2;
            renderPassBeginInfo.pClearValues = clearValues.data();

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.composition, 0, 1, &descriptorSets.composition, 0, NULL);

            // Final composition pass
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.composition);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);
        }

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}
```

This code sets up the sequence of render passes that will execute on the GPU:

1. **G-Buffer Pass**:
   - Begin the G-Buffer render pass
   - Set viewport and scissor rectangle
   - Bind the G-Buffer pipeline
   - Bind descriptor sets (uniform buffers, textures)
   - Draw the scene to fill the G-Buffer
   - End the render pass

2. **SSAO Pass**:
   - Begin the SSAO render pass
   - Set viewport and scissor
   - Bind the SSAO pipeline and descriptors
   - Draw a fullscreen triangle (3 vertices) to process every pixel
   - End the render pass

3. **Blur Pass**:
   - Begin the blur render pass
   - Set viewport and scissor
   - Bind the blur pipeline and descriptors
   - Draw a fullscreen triangle
   - End the render pass

4. **Composition Pass**:
   - Begin the final render pass
   - Set viewport and scissor
   - Bind the composition pipeline and descriptors
   - Draw a fullscreen triangle to produce the final image
   - Draw UI (for toggling SSAO options)
   - End the render pass

Note: The fullscreen triangle (used in SSAO, blur, and composition passes) is generated in the vertex shader without needing to specify vertex data explicitly. This is a common technique for full-screen effects.

## Shader Implementation

Let's examine each shader in detail to understand how SSAO is calculated.

### G-Buffer Pass

The G-Buffer pass renders scene information into multiple textures. This implementation uses vertex and fragment shaders:

#### Vertex Shader (gbuffer.vert) - Complete Code

```glsl
#version 450

// Input vertex attributes
layout (location = 0) in vec4 inPos;      // Vertex position
layout (location = 1) in vec2 inUV;       // Texture coordinates
layout (location = 2) in vec3 inColor;    // Vertex color
layout (location = 3) in vec3 inNormal;   // Vertex normal

// Uniform buffer containing transformation matrices
layout (binding = 0) uniform UBO 
{
    mat4 projection;  // Projection matrix
    mat4 model;       // Model matrix
    mat4 view;        // View matrix
} ubo;

// Output to fragment shader
layout (location = 0) out vec3 outNormal;  // Normal vector
layout (location = 1) out vec2 outUV;      // Texture coordinates
layout (location = 2) out vec3 outColor;   // Vertex color
layout (location = 3) out vec3 outPos;     // Position in view space

void main() 
{
    // Transform vertex position to clip space for rasterization
    gl_Position = ubo.projection * ubo.view * ubo.model * inPos;
    
    // Pass texture coordinates unchanged
    outUV = inUV;

    // Vertex position in view space (camera space)
    // This is critical for SSAO - we need positions in view space
    outPos = vec3(ubo.view * ubo.model * inPos);

    // Normal vector in view space
    // Using transpose(inverse()) ensures normals are correct even with non-uniform scaling
    mat3 normalMatrix = transpose(inverse(mat3(ubo.view * ubo.model)));
    outNormal = normalMatrix * inNormal;

    // Pass vertex color unchanged
    outColor = inColor;
}
```

#### Fragment Shader (gbuffer.frag) - Complete Code

```glsl
#version 450

// Input from vertex shader
layout (location = 0) in vec3 inNormal;  // Normal vector
layout (location = 1) in vec2 inUV;      // Texture coordinates
layout (location = 2) in vec3 inColor;   // Vertex color
layout (location = 3) in vec3 inPos;     // Position in view space

// Multiple render targets - writing to 3 different textures at once
layout (location = 0) out vec4 outPosition;  // Position + linear depth
layout (location = 1) out vec4 outNormal;    // Normal
layout (location = 2) out vec4 outAlbedo;    // Color

// Uniform buffer containing matrices and camera settings
layout (set = 0, binding = 0) uniform UBO 
{
    mat4 projection;
    mat4 model;
    mat4 view;
    float nearPlane;   // Camera near clip plane
    float farPlane;    // Camera far clip plane
} ubo;

// Texture sampler for the model's color map
layout (set = 1, binding = 0) uniform sampler2D samplerColormap;

// Convert hardware depth to linear depth
// Hardware depth is non-linear (higher precision near camera)
// Linear depth is more useful for calculations
float linearDepth(float depth)
{
    // Convert depth from [0,1] to [-1,1] range (normalized device coordinates)
    float z = depth * 2.0f - 1.0f; 
    
    // Apply perspective division formula to convert to linear depth
    return (2.0f * ubo.nearPlane * ubo.farPlane) / 
           (ubo.farPlane + ubo.nearPlane - z * (ubo.farPlane - ubo.nearPlane));    
}

void main() 
{
    // Store position and linearized depth
    // XYZ = position in view space, W = linear depth
    outPosition = vec4(inPos, linearDepth(gl_FragCoord.z));
    
    // Store normal (remapped from [-1,1] to [0,1] range for storage)
    // We'll convert it back to [-1,1] when reading
    outNormal = vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
    
    // Store albedo (color) = texture color * vertex color
    outAlbedo = texture(samplerColormap, inUV) * vec4(inColor, 1.0);
}
```

This G-Buffer pass writes to three different textures simultaneously:
1. **Position Buffer**: Stores 3D position in view space (XYZ) and linear depth (W)
2. **Normal Buffer**: Stores surface normals (remapped to [0,1] range for storage)
3. **Albedo Buffer**: Stores color information (texture color * vertex color)

### SSAO Pass

The SSAO pass uses the G-Buffer to calculate ambient occlusion. Here's the complete shader code:

#### Vertex Shader (fullscreen.vert) - Complete Code

```glsl
#version 450

// Output to fragment shader
layout (location = 0) out vec2 outUV;

// Built-in output variable for vertex position
out gl_PerVertex
{
    vec4 gl_Position;
};

void main() 
{
    // Generate a triangle that covers the entire screen using the vertex ID
    // This trick allows rendering a fullscreen effect without vertex buffers
    //
    // gl_VertexIndex values: 0, 1, 2 (for the three vertices of the triangle)
    // Results in UV coordinates: (0,0), (2,0), (0,2)
    // Which creates a large triangle covering the screen
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    
    // Map from [0,2] to [-1,1] range for clip space
    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
```

This vertex shader is used for all fullscreen passes (SSAO, blur, and composition). It generates a large triangle that covers the entire screen without requiring vertex input data.

#### Fragment Shader (ssao.frag) - Complete Code

```glsl
#version 450

// Input textures from G-Buffer
layout (binding = 0) uniform sampler2D samplerPositionDepth;  // Position + depth
layout (binding = 1) uniform sampler2D samplerNormal;         // Normals
layout (binding = 2) uniform sampler2D ssaoNoise;             // Random rotation vectors

// Specialization constants - set at pipeline creation time
// These are compile-time constants for better performance
layout (constant_id = 0) const int SSAO_KERNEL_SIZE = 64;     // Number of sample points
layout (constant_id = 1) const float SSAO_RADIUS = 0.5;       // Sampling radius

// Sample kernel (hemisphere of sample points)
layout (binding = 3) uniform UBOSSAOKernel
{
    vec4 samples[SSAO_KERNEL_SIZE];  // Array of sample points in tangent space
} uboSSAOKernel;

// Projection matrix needed to project samples to screen space
layout (binding = 4) uniform UBO 
{
    mat4 projection;  // Camera projection matrix
} ubo;

// Input from vertex shader
layout (location = 0) in vec2 inUV;  // Texture coordinates

// Output occlusion value (1.0 = no occlusion, 0.0 = full occlusion)
layout (location = 0) out float outFragColor;

void main() 
{
    // 1. Get data from G-Buffer
    vec3 fragPos = texture(samplerPositionDepth, inUV).rgb;  // Position in view space
    // Convert normal from [0,1] range back to [-1,1] range
    vec3 normal = normalize(texture(samplerNormal, inUV).rgb * 2.0 - 1.0);

    // 2. Get a random vector for rotating the sample kernel
    // Get dimensions of position and noise textures
    ivec2 texDim = textureSize(samplerPositionDepth, 0);
    ivec2 noiseDim = textureSize(ssaoNoise, 0);
    
    // Calculate UV coordinates to tile the small noise texture across the screen
    const vec2 noiseUV = vec2(float(texDim.x)/float(noiseDim.x), float(texDim.y)/(noiseDim.y)) * inUV;
    // Get random vector and convert from [0,1] to [-1,1]
    vec3 randomVec = texture(ssaoNoise, noiseUV).xyz * 2.0 - 1.0;
    
    // 3. Create TBN matrix (Tangent, Bitangent, Normal)
    // This matrix will rotate our sample kernel to be oriented along the surface normal
    
    // Compute tangent using Gram-Schmidt process:
    // Make randomVec perpendicular to normal
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    // Cross product to get third perpendicular vector
    vec3 bitangent = cross(tangent, normal);
    // Build rotation matrix with three perpendicular axes
    mat3 TBN = mat3(tangent, bitangent, normal);

    // 4. Calculate occlusion value
    float occlusion = 0.0f;
    // Small bias to prevent self-occlusion due to precision issues
    const float bias = 0.025f;
    
    // Loop through all samples in the kernel
    for(int i = 0; i < SSAO_KERNEL_SIZE; i++)
    {
        // a. Get sample position in view space
        // Rotate sample to surface orientation using TBN matrix
        vec3 samplePos = TBN * uboSSAOKernel.samples[i].xyz;
        // Position sample in view space at current fragment
        samplePos = fragPos + samplePos * SSAO_RADIUS;
        
        // b. Project sample position to screen space
        vec4 offset = vec4(samplePos, 1.0f);
        // Transform to clip space
        offset = ubo.projection * offset;
        // Perspective division (convert to NDC space)
        offset.xyz /= offset.w;
        // Convert from [-1,1] to [0,1] range (texture coordinates)
        offset.xyz = offset.xyz * 0.5f + 0.5f;
        
        // c. Get depth at sample position
        // Sample depth from position texture
        float sampleDepth = -texture(samplerPositionDepth, offset.xy).w;
        
        // d. Check if sample is occluded
        // Range check makes occlusion fade with distance
        // (samples too far from current fragment don't contribute)
        float rangeCheck = smoothstep(0.0f, 1.0f, SSAO_RADIUS / abs(fragPos.z - sampleDepth));
        
        // Add occlusion contribution:
        // If the sample depth is beyond the sample position (sample is occluded),
        // add to occlusion factor, scaled by range check
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0f : 0.0f) * rangeCheck;
    }
    
    // 5. Normalize and invert (1.0 = no occlusion, 0.0 = full occlusion)
    occlusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));
    
    // Output occlusion value
    outFragColor = occlusion;
}
```

This shader implements the core SSAO algorithm:

1. **Get Position and Normal**: Read the position and normal for the current pixel
2. **Create Rotation Matrix**: Use random vectors to create a rotation matrix (TBN) for the sample kernel
3. **Sample Points Around Pixel**: For each sample point in the hemisphere:
   - Transform the sample to be oriented with the surface normal
   - Project the sample to screen space
   - Check if the sample point is occluded (behind a surface)
4. **Calculate Occlusion**: Count the occluded samples and normalize to get the final occlusion value

### Blur Pass

The blur pass smooths the raw SSAO result to reduce noise. Here's the complete shader code:

#### Fragment Shader (blur.frag) - Complete Code

```glsl
#version 450

// Input SSAO texture to be blurred
layout (binding = 0) uniform sampler2D samplerSSAO;

// Input from vertex shader
layout (location = 0) in vec2 inUV;

// Output blurred value
layout (location = 0) out float outFragColor;

void main() 
{
    // Simple box blur implementation
    const int blurRange = 2;  // Blur radius (2 pixels in each direction)
    int n = 0;  // Sample counter
    
    // Calculate size of one pixel in texture coordinates
    vec2 texelSize = 1.0 / vec2(textureSize(samplerSSAO, 0));
    
    float result = 0.0;
    // Box filter: sample a square area around the pixel
    for (int x = -blurRange; x <= blurRange; x++) 
    {
        for (int y = -blurRange; y <= blurRange; y++) 
        {
            // Calculate offset for this sample
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            // Add the sample to our sum
            result += texture(samplerSSAO, inUV + offset).r;
            n++;  // Count the sample
        }
    }
    
    // Average all samples to get final blur result
    outFragColor = result / (float(n));
}
```

This shader implements a simple box blur:
1. For each pixel within the blur radius (5x5 area)
2. Sample the SSAO texture and sum the values
3. Divide by number of samples to get average

### Composition Pass

The composition pass combines all the information to produce the final image. Here's the complete shader code:

#### Fragment Shader (composition.frag) - Complete Code

```glsl
#version 450

// Input textures from G-Buffer and SSAO passes
layout (binding = 0) uniform sampler2D samplerposition;  // Position + depth
layout (binding = 1) uniform sampler2D samplerNormal;    // Normals
layout (binding = 2) uniform sampler2D samplerAlbedo;    // Color
layout (binding = 3) uniform sampler2D samplerSSAO;      // Raw SSAO
layout (binding = 4) uniform sampler2D samplerSSAOBlur;  // Blurred SSAO

// Parameters for SSAO control
layout (binding = 5) uniform UBO 
{
    mat4 _dummy;     // Not used, but keeps alignment
    int ssao;        // Toggle SSAO on/off (boolean)
    int ssaoOnly;    // Show only SSAO (boolean)
    int ssaoBlur;    // Use blurred or raw SSAO (boolean)
} uboParams;

// Input from vertex shader
layout (location = 0) in vec2 inUV;

// Output final color
layout (location = 0) out vec4 outFragColor;

void main() 
{
    // 1. Read data from G-Buffer
    vec3 fragPos = texture(samplerposition, inUV).rgb;  // Position in view space
    // Convert normal from [0,1] back to [-1,1] range
    vec3 normal = normalize(texture(samplerNormal, inUV).rgb * 2.0 - 1.0);
    vec4 albedo = texture(samplerAlbedo, inUV);  // Color
     
    // 2. Get SSAO value (either blurred or raw based on setting)
    float ssao = (uboParams.ssaoBlur == 1) ? 
                  texture(samplerSSAOBlur, inUV).r : 
                  texture(samplerSSAO, inUV).r;

    // 3. Basic lighting calculation (simplified)
    // Light at origin (camera position in view space)
    vec3 lightPos = vec3(0.0);
    // Light direction
    vec3 L = normalize(lightPos - fragPos);
    // Diffuse factor with ambient minimum of 0.5
    float NdotL = max(0.5, dot(normal, L));

    // 4. Output options:
    if (uboParams.ssaoOnly == 1)
    {
        // Option 1: Show only SSAO factor (grayscale visualization)
        outFragColor.rgb = ssao.rrr;
    }
    else
    {
        // Calculate base color with simple lighting
        vec3 baseColor = albedo.rgb * NdotL;

        if (uboParams.ssao == 1)
        {
            // Option 2: Apply SSAO to the scene
            outFragColor.rgb = ssao.rrr;

            if (uboParams.ssaoOnly != 1)
                // Multiply SSAO with base color
                outFragColor.rgb *= baseColor;
        }
        else
        {
            // Option 3: Show scene without SSAO
            outFragColor.rgb = baseColor;
        }
    }
}
```

This shader combines all the previous render passes to create the final image:
1. Reads position, normal, and color from G-Buffer
2. Gets the SSAO value (using either blurred or raw)
3. Calculates simple lighting
4. Combines everything based on the UI settings:
   - Can show only SSAO (grayscale)
   - Can apply SSAO to the scene
   - Can show the scene without SSAO

## Complete Rendering Process

Now that we've examined each component in detail, let's summarize the complete SSAO rendering process:

### 1. Data Setup

Before rendering starts, we prepare:

- **Sample Kernel**: A set of random 3D points distributed in a hemisphere shape
- **Noise Texture**: A small texture of random 2D vectors to rotate the kernel

```cpp
// Create sample kernel - set of points in a hemisphere
std::vector<glm::vec4> ssaoKernel(SSAO_KERNEL_SIZE);
for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i)
{
    // Create random vector in hemisphere
    glm::vec3 sample(rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine) * 2.0 - 1.0, rndDist(rndEngine));
    sample = glm::normalize(sample);
    sample *= rndDist(rndEngine);
    
    // Scale samples closer to center
    float scale = float(i) / float(SSAO_KERNEL_SIZE);
    scale = lerp(0.1f, 1.0f, scale * scale);
    ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
}

// Create noise texture for rotating kernel
std::vector<glm::vec4> noiseValues(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
for (uint32_t i = 0; i < static_cast<uint32_t>(noiseValues.size()); i++) {
    noiseValues[i] = glm::vec4(rndDist(rndEngine) * 2.0f - 1.0f, rndDist(rndEngine) * 2.0f - 1.0f, 0.0f, 0.0f);
}
```

### 2. G-Buffer Pass

In this pass, we render the scene geometry to multiple textures:

```cpp
// G-Buffer render pass
vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);
vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.gBuffer, 0, 1, &descriptorSets.gBuffer, 0, nullptr);
scene.draw(drawCmdBuffers[i], vkglTF::RenderFlags::BindImages, pipelineLayouts.gBuffer);
vkCmdEndRenderPass(drawCmdBuffers[i]);
```

The shaders for this pass create:
- **Position texture**: Stores 3D positions and depth
- **Normal texture**: Stores surface normals
- **Albedo texture**: Stores color information

### 3. SSAO Pass

Here we calculate ambient occlusion using the G-Buffer:

```cpp
// SSAO render pass
vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ssao, 0, 1, &descriptorSets.ssao, 0, nullptr);
vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.ssao);
vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
vkCmdEndRenderPass(drawCmdBuffers[i]);
```

The SSAO shader:
1. Gets position and normal for current pixel
2. Creates a TBN matrix to rotate sample kernel
3. For each sample in the kernel:
   - Positions the sample around the current pixel
   - Projects it to screen space
   - Checks if it's occluded by geometry
4. Calculates final occlusion value

### 4. Blur Pass

This pass smooths the raw SSAO result to reduce noise:

```cpp
// SSAO blur render pass
vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.ssaoBlur, 0, 1, &descriptorSets.ssaoBlur, 0, nullptr);
vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.ssaoBlur);
vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
vkCmdEndRenderPass(drawCmdBuffers[i]);
```

The blur shader:
- Takes 5x5 samples around each pixel
- Averages them to get a smoother result

### 5. Composition Pass

The final pass combines everything for the final image:

```cpp
// Final composition render pass
vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.composition, 0, 1, &descriptorSets.composition, 0, NULL);
vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.composition);
vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
vkCmdEndRenderPass(drawCmdBuffers[i]);
```

The composition shader:
- Reads position, normal, and color from G-Buffer
- Gets SSAO value (blurred or raw)
- Calculates simple lighting
- Applies SSAO to darken occluded areas
- Outputs final image

### Visual Results

The final result shows subtle darkening in corners, crevices, and areas where objects are close together. This enhances the perception of depth and makes the scene look more grounded and realistic.

## Key Performance and Quality Factors

Several factors affect the quality and performance of SSAO:

### 1. Kernel Size (SSAO_KERNEL_SIZE)
- **Higher values** (64-128): Better quality, smoother results, but slower
- **Lower values** (16-32): Faster, but may look grainy or have banding

### 2. Sampling Radius (SSAO_RADIUS)
- **Larger radius**: Detects larger-scale occlusion, but may miss fine details
- **Smaller radius**: Better for detailed geometry, but may miss larger occlusions

### 3. Noise Texture Size (SSAO_NOISE_DIM)
- **Larger**: Less repeating patterns, but more memory usage
- **Smaller**: More efficient, but may show repeating patterns

### 4. Blur Strength
- **More blur**: Smoother results, less noise, but may lose detail
- **Less blur**: Preserves detail, but may show noise

### 5. G-Buffer Resolution
- **Higher resolution**: More accurate, but requires more memory and processing
- **Lower resolution**: Faster, but may miss fine details (especially on mobile)

## Conclusion

SSAO is a powerful technique that adds significant realism to 3D scenes without the computational cost of true ambient occlusion. By smartly approximating how light is occluded in screen space, it creates subtle shadows that enhance the perception of depth and form.

The implementation we've examined splits the process into multiple render passes:
1. G-Buffer generation to capture scene data
2. SSAO calculation using the G-Buffer
3. Blur to smooth the results
4. Final composition to apply SSAO to the scene

This modular approach allows for efficient processing and easy customization of each step. By adjusting parameters like kernel size, radius, and blur strength, you can balance quality and performance for your specific application.
