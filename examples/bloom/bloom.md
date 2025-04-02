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

// Offscreen frame buffer properties
#define FB_DIM 256
#define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM

class VulkanExample : public VulkanExampleBase
{
public:
	bool bloom = true;

	vks::TextureCubeMap cubemap;

	struct {
		vkglTF::Model ufo;
		vkglTF::Model ufoGlow;
		vkglTF::Model skyBox;
	} models;

	struct {
		vks::Buffer scene;
		vks::Buffer skyBox;
		vks::Buffer blurParams;
	} uniformBuffers;

	struct UBO {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
	};

	struct UBOBlurParams {
		float blurScale = 1.0f;
		float blurStrength = 1.5f;
	};

	struct {
		UBO scene, skyBox;
		UBOBlurParams blurParams;
	} ubos;

	struct {
		VkPipeline blurVert;
		VkPipeline blurHorz;
		VkPipeline glowPass;
		VkPipeline phongPass;
		VkPipeline skyBox;
	} pipelines;

	struct {
		VkPipelineLayout blur;
		VkPipelineLayout scene;
	} pipelineLayouts;

	struct {
		VkDescriptorSet blurVert;
		VkDescriptorSet blurHorz;
		VkDescriptorSet scene;
		VkDescriptorSet skyBox;
	} descriptorSets;

	struct {
		VkDescriptorSetLayout blur;
		VkDescriptorSetLayout scene;
	} descriptorSetLayouts;

   // Frame buffer for offscreen rendering
   struct FrameBufferAttachment {
       VkImage image;
       VkDeviceMemory mem;
       VkImageView view;
   };
   
   struct FrameBuffer {
       VkFramebuffer framebuffer;
       FrameBufferAttachment color, depth;
       VkDescriptorImageInfo descriptor;
   };
   
   struct OffscreenPass {
       int32_t width, height;
       VkRenderPass renderPass;
       VkSampler sampler;
       std::array<FrameBuffer, 2> framebuffers;
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

vkDestroySampler(device, offscreenPass.sampler, nullptr);

// Frame buffer
for (auto& framebuffer : offscreenPass.framebuffers)
{
// Attachments
vkDestroyImageView(device, framebuffer.color.view, nullptr);
vkDestroyImage(device, framebuffer.color.image, nullptr);
vkFreeMemory(device, framebuffer.color.mem, nullptr);
vkDestroyImageView(device, framebuffer.depth.view, nullptr);
vkDestroyImage(device, framebuffer.depth.image, nullptr);
vkFreeMemory(device, framebuffer.depth.mem, nullptr);

vkDestroyFramebuffer(device, framebuffer.framebuffer, nullptr);
}
vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);

vkDestroyPipeline(device, pipelines.blurHorz, nullptr);
vkDestroyPipeline(device, pipelines.blurVert, nullptr);
vkDestroyPipeline(device, pipelines.phongPass, nullptr);
vkDestroyPipeline(device, pipelines.glowPass, nullptr);
vkDestroyPipeline(device, pipelines.skyBox, nullptr);

vkDestroyPipelineLayout(device, pipelineLayouts.blur, nullptr);
vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);

vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.blur, nullptr);
vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);

// Uniform buffers
uniformBuffers.scene.destroy();
uniformBuffers.skyBox.destroy();
uniformBuffers.blurParams.destroy();

cubemap.destroy();
}
```

## Preparing the Offscreen Framebuffers

We need to set up offscreen framebuffers for the bloom effect:

```cpp
void prepareOffscreenFramebuffer(FrameBuffer *frameBuf, VkFormat colorFormat, VkFormat depthFormat)
{
    // Color attachment
    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = colorFormat;
    image.extent.width = FB_DIM;
    image.extent.height = FB_DIM;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    // We will sample directly from the color attachment
    image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    // Create the image, allocate memory, and create the image view
    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorImageView.format = colorFormat;
    colorImageView.flags = 0;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorImageView.subresourceRange.baseMipLevel = 0;
    colorImageView.subresourceRange.levelCount = 1;
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->color.image));
    vkGetImageMemoryRequirements(device, frameBuf->color.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->color.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->color.image, frameBuf->color.mem, 0));

    colorImageView.image = frameBuf->color.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &frameBuf->color.view));

    // Depth stencil attachment
    image.format = depthFormat;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = depthFormat;
    depthStencilView.flags = 0;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vks::tools::formatHasStencil(depthFormat)) {
        depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->depth.image));
    vkGetImageMemoryRequirements(device, frameBuf->depth.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->depth.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->depth.image, frameBuf->depth.mem, 0));

    depthStencilView.image = frameBuf->depth.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &frameBuf->depth.view));

    VkImageView attachments[2];
    attachments[0] = frameBuf->color.view;
    attachments[1] = frameBuf->depth.view;

    VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass.renderPass;
    fbufCreateInfo.attachmentCount = 2;
    fbufCreateInfo.pAttachments = attachments;
    fbufCreateInfo.width = FB_DIM;
    fbufCreateInfo.height = FB_DIM;
    fbufCreateInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuf->framebuffer));
    
    // Set up the descriptor for sampling in the shaders
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frameBuf->descriptor.imageView = frameBuf->color.view;
    frameBuf->descriptor.sampler = offscreenPass.sampler;
}

// Prepare the offscreen framebuffers used for the vertical- and horizontal blur
void prepareOffscreen()
{
    offscreenPass.width = FB_DIM;
    offscreenPass.height = FB_DIM;

    // Find a suitable depth format
    VkFormat fbDepthFormat;
    VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
    assert(validDepthFormat);
    
    // Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering
    std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
    // Color attachment
    attchmentDescriptions[0].format = FB_COLOR_FORMAT;
    attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Depth attachment
    attchmentDescriptions[1].format = fbDepthFormat;
    attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pDepthStencilAttachment = &depthReference;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Create the actual renderpass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
    renderPassInfo.pAttachments = attchmentDescriptions.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass.renderPass));
    
    // Create sampler to sample from the color attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreenPass.sampler));

    // Create both framebuffers
    prepareOffscreenFramebuffer(&offscreenPass.framebuffers[0], FB_COLOR_FORMAT, fbDepthFormat);
    prepareOffscreenFramebuffer(&offscreenPass.framebuffers[1], FB_COLOR_FORMAT, fbDepthFormat);
}
```

## Shader Implementation

The bloom effect requires several shader pairs. Let's examine each of them:

### 1. Color Pass (Brightness Extraction)

The color pass identifies and extracts bright parts of our scene:

`colorpass.vert`

```glsl
#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    outUV = inUV;
    outColor = inColor;
    gl_Position = ubo.projection * ubo.view * ubo.model * inPos;
}
```

`colorpass.frag`

```glsl
#version 450

layout (binding = 1) uniform sampler2D colorMap;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
    outFragColor.rgb = inColor;
    // Alternative approach using textures
    // outFragColor = texture(colorMap, inUV);// * vec4(inColor, 1.0);
}
```

### 2. Gaussian Blur

For the blur, we use a two-pass approach (vertical and horizontal) with the same shaders but different parameters:

`gaussblur.vert`

```glsl
#version 450

layout (location = 0) out vec2 outUV;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
```

`gaussblur.frag`

```glsl
#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (binding = 0) uniform UBO
{
    float blurScale;
    float blurStrength;
} ubo;

layout (constant_id = 0) const int blurdirection = 0;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
    float weight[5];
    weight[0] = 0.227027;
    weight[1] = 0.1945946;
    weight[2] = 0.1216216;
    weight[3] = 0.054054;
    weight[4] = 0.016216;

    vec2 tex_offset = 1.0 / textureSize(samplerColor, 0) * ubo.blurScale;
    vec3 result = texture(samplerColor, inUV).rgb * weight[0];

    for (int i = 1; i < 5; ++i)
    {
        if (blurdirection == 1)
        {
            // Horizontal blur
            result += texture(samplerColor, inUV + vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
            result += texture(samplerColor, inUV - vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
        }
        else
        {
            // Vertical blur
            result += texture(samplerColor, inUV + vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
            result += texture(samplerColor, inUV - vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
        }
    }
    outFragColor = vec4(result, 1.0);
}
```

### 3. Final Scene Rendering (Phong shading)

For the main scene, we use a basic Phong lighting model:

`phongpass.vert`

```glsl
#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inNormal;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    outNormal = inNormal;
    outColor = inColor;
    outUV = inUV;
    gl_Position = ubo.projection * ubo.view * ubo.model * inPos;

    vec3 lightPos = vec3(-5.0, -5.0, 0.0);
    vec4 pos = ubo.view * ubo.model * inPos;
    outNormal = mat3(ubo.view * ubo.model) * inNormal;
    outLightVec = lightPos - pos.xyz;
    outViewVec = -pos.xyz;
}
```

`phongpass.frag`

```glsl
#version 450

layout (binding = 1) uniform sampler2D colorMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 ambient = vec3(0.0f);

    // Adjust light calculations for glow color 
    if ((inColor.r >= 0.9) || (inColor.g >= 0.9) || (inColor.b >= 0.9))
    {
        ambient = inColor * 0.25;
    }

    vec3 N = normalize(inNormal);
    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);
    vec3 R = reflect(-L, N);
    vec3 diffuse = max(dot(N, L), 0.0) * inColor;
    vec3 specular = pow(max(dot(R, V), 0.0), 8.0) * vec3(0.75);
    outFragColor = vec4(ambient + diffuse + specular, 1.0);
}
```

## Prepare uniform buffers

We will prepare and initialize uniform buffer containing the shader uniforms

```cpp
void prepareUniformBuffers()
{
   // Phong and color pass vertex shader uniform buffer
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.scene,
      sizeof(ubos.scene)));
   
   // Blur parameters uniform buffers
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.blurParams,
      sizeof(ubos.blurParams)));
   
   // Skybox
   VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.skyBox,
      sizeof(ubos.skyBox)));
   
   // Map persistent
   VK_CHECK_RESULT(uniformBuffers.scene.map());
   VK_CHECK_RESULT(uniformBuffers.blurParams.map());
   VK_CHECK_RESULT(uniformBuffers.skyBox.map());
   
   // Initialize uniform buffers
   updateUniformBuffersScene();
   updateUniformBuffersBlur();
}

```

## Setting Up the Pipeline

We need to set up several pipelines for our bloom effect:

```cpp
void preparePipelines()
{
    // Create pipeline layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.blur, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.blur));

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.scene, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.scene));

    // Common pipeline settings
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts.blur, renderPass, 0);
    pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
    pipelineCI.pRasterizationState = &rasterizationStateCI;
    pipelineCI.pColorBlendState = &colorBlendStateCI;
    pipelineCI.pMultisampleState = &multisampleStateCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pDynamicState = &dynamicStateCI;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();

    // Blur pipelines
    shaderStages[0] = loadShader(getShadersPath() + "bloom/gaussblur.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/gaussblur.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    // Empty vertex input state (fullscreen quad generated in vertex shader)
    VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts.blur;
    
    // Additive blending for the horizontal blur (final composition)
    blendAttachmentState.colorWriteMask = 0xF;
    blendAttachmentState.blendEnable = VK_TRUE;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

    // Use specialization constants to select between horizontal and vertical blur
    uint32_t blurdirection = 0;
    VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(uint32_t), &blurdirection);
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    
    // Vertical blur pipeline 
    pipelineCI.renderPass = offscreenPass.renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.blurVert));
    
    // Horizontal blur pipeline
    blurdirection = 1;
    pipelineCI.renderPass = renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.blurHorz));

    // Create other pipelines (phongPass, glowPass, skyBox)
    // Phong pass (3D model)
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal});
    pipelineCI.layout = pipelineLayouts.scene;
    shaderStages[0] = loadShader(getShadersPath() + "bloom/phongpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/phongpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    blendAttachmentState.blendEnable = VK_FALSE;
    depthStencilStateCI.depthWriteEnable = VK_TRUE;
    rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
    pipelineCI.renderPass = renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.phongPass));

    // Color only pass (offscreen blur base)
    shaderStages[0] = loadShader(getShadersPath() + "bloom/colorpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/colorpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineCI.renderPass = offscreenPass.renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.glowPass));

    // Skybox (cubemap)
    shaderStages[0] = loadShader(getShadersPath() + "bloom/skybox.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "bloom/skybox.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    depthStencilStateCI.depthWriteEnable = VK_FALSE;
    rasterizationStateCI.cullMode = VK_CULL_MODE_FRONT_BIT;
    pipelineCI.renderPass = renderPass;
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
    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 5);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Create descriptor set layouts
    // Blur descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
    };
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = 
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.blur));

    // Scene rendering descriptor set layout
    setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),			// Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),	// Binding 1 : Fragment shader image sampler
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),			// Binding 2 : Fragment shader image sampler
    };

    descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.scene));

    // Sets
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo;
    std::vector<VkWriteDescriptorSet> writeDescriptorSets;

    // Allocate and update descriptor sets
    // Vertical blur
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo = 
        vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.blur, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurVert));
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(descriptorSets.blurVert, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.blurParams.descriptor),
        vks::initializers::writeDescriptorSet(descriptorSets.blurVert, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &offscreenPass.framebuffers[0].descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    
    // Horizontal blur
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurHorz));
    writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.blurParams.descriptor),
        vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &offscreenPass.framebuffers[1].descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    // Other descriptor sets (scene, skybox)
    // Scene rendering
    descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.scene, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.scene));
    writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(descriptorSets.scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.scene.descriptor)							// Binding 0: Vertex shader uniform buffer
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    // Skybox
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.skyBox));
    writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(descriptorSets.skyBox, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.skyBox.descriptor),						// Binding 0: Vertex shader uniform buffer
        vks::initializers::writeDescriptorSet(descriptorSets.skyBox, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	1, &cubemap.descriptor),							// Binding 1: Fragment shader texture sampler
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}
```

## Command Buffer Construction

Now let's build our command buffer:

```cpp
void buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    VkClearValue clearValues[2];
    VkViewport viewport;
    VkRect2D scissor;

    /*
        The blur method used in this example is multi pass and renders the vertical blur first and then the horizontal one
        While it's possible to blur in one pass, this method is widely used as it requires far less samples to generate the blur
    */
    
    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        if (bloom) {
            clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
            clearValues[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
            renderPassBeginInfo.renderPass = offscreenPass.renderPass;
            renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[0].framebuffer;
            renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
            renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
            renderPassBeginInfo.clearValueCount = 2;
            renderPassBeginInfo.pClearValues = clearValues;

            viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            // First render pass: Render glow parts of the model (separate mesh) to an offscreen framebuffer
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.glowPass);
            models.ufoGlow.draw(drawCmdBuffers[i]);
            vkCmdEndRenderPass(drawCmdBuffers[i]);

            // Second render pass: Vertical blur
            // Render contents of the first pass into a second framebuffer and apply a vertical blur
            // This is the first blur pass, the horizontal blur is applied when rendering on top of the scene
            renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[1].framebuffer;
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blur, 0, 1, &descriptorSets.blurVert, 0, NULL);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blurVert);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
            vkCmdEndRenderPass(drawCmdBuffers[i]);
        }

        // Third render pass: Scene rendering with applied vertical blur
        // Renders the scene and the (vertically blurred) contents of the second framebuffer and apply a horizontal blur
        clearValues[0].color = defaultClearColor;
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = frameBuffers[i];
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        // Skybox
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.skyBox, 0, NULL);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.skyBox);
        models.skyBox.draw(drawCmdBuffers[i]);

        // 3D scene
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.phongPass);
        models.ufo.draw(drawCmdBuffers[i]);

        // Apply horizontal blur with additive blending to achieve the bloom effect
        if (bloom)
        {
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blur, 0, 1, &descriptorSets.blurHorz, 0, NULL);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blurHorz);
            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
        }

        drawUI(drawCmdBuffers[i]);
        
        vkCmdEndRenderPass(drawCmdBuffers[i]);
        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}
```

## Loading assets and finishing preparations

Finally, we load our assets and finish the setup:

```cpp
void loadAssets()
{
    const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
    models.ufo.loadFromFile(getAssetPath() + "models/retroufo.gltf", vulkanDevice, queue, glTFLoadingFlags);
    models.ufoGlow.loadFromFile(getAssetPath() + "models/retroufo_glow.gltf", vulkanDevice, queue, glTFLoadingFlags);
    models.skyBox.loadFromFile(getAssetPath() + "models/cube.gltf", vulkanDevice, queue, glTFLoadingFlags);
    cubemap.loadFromFile(getAssetPath() + "textures/cubemap_space.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
}

void prepare()
{
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    prepareOffscreen();
    setupDescriptors();
    preparePipelines();
    buildCommandBuffers();
    prepared = true;
}
```

## Updating the uniform buffers in loop

A missing piece is updating the uniform buffers for the scene:

```cpp
// Update uniform buffers for rendering the 3D scene
void updateUniformBuffersScene()
{
   // UFO
   ubos.scene.projection = camera.matrices.perspective;
   ubos.scene.view = camera.matrices.view;
   
   ubos.scene.model = glm::translate(glm::mat4(1.0f), glm::vec3(sin(glm::radians(timer * 360.0f)) * 0.25f, -1.0f, cos(glm::radians(timer * 360.0f)) * 0.25f));
   ubos.scene.model = glm::rotate(ubos.scene.model, -sinf(glm::radians(timer * 360.0f)) * 0.15f, glm::vec3(1.0f, 0.0f, 0.0f));
   ubos.scene.model = glm::rotate(ubos.scene.model, glm::radians(timer * 360.0f), glm::vec3(0.0f, 1.0f, 0.0f));
   
   memcpy(uniformBuffers.scene.mapped, &ubos.scene, sizeof(ubos.scene));
   
   // Skybox
   ubos.skyBox.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 256.0f);
   ubos.skyBox.view = glm::mat4(glm::mat3(camera.matrices.view));
   ubos.skyBox.model = glm::mat4(1.0f);
   
   memcpy(uniformBuffers.skyBox.mapped, &ubos.skyBox, sizeof(ubos.skyBox));
}
```

## Rendering

We call our trustful `render()` and `draw()` functions:

```cpp
	void draw()
	{
		VulkanExampleBase::prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VulkanExampleBase::submitFrame();
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused || camera.updated)
		{
			updateUniformBuffersScene();
		}
	}
```

## Controlling Bloom Parameters

We use a uniform buffer to control bloom parameters:

```cpp
struct UBOBlurParams {
    float blurScale = 1.0f;
    float blurStrength = 1.5f;
};

// Update blur parameters
void updateUniformBuffersBlur()
{
    memcpy(uniformBuffers.blurParams.mapped, &ubos.blurParams, sizeof(ubos.blurParams));
}

// Add UI controls for bloom parameters
virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
{
    if (overlay->header("Settings")) {
        if (overlay->checkBox("Bloom", &bloom)) {
            buildCommandBuffers();
        }
        if (overlay->inputFloat("Scale", &ubos.blurParams.blurScale, 0.1f, 2)) {
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

2. **Use multiple blur passes with different kernel sizes**: For more advanced bloom effects, consider using multiple
   blur passes with different kernel sizes to create a more natural-looking bloom effect.

3. **HDR rendering**: Bloom works best in a high dynamic range rendering pipeline. Consider implementing HDR rendering
   for even better results.

4. **Performance optimization**: Consider using a lower resolution for the bloom effect. You can render to a smaller
   offscreen framebuffer (e.g., half the screen size) and then upsample when compositing. This can significantly improve
   performance with minimal quality loss.

5. **Tone mapping**: If implementing HDR, be sure to apply proper tone mapping after adding bloom to ensure your final
   image looks correct on standard displays.


## Conclusion

Implementing bloom in Vulkan requires a good understanding of offscreen rendering, multiple render passes, and shader
programming. The multi-pass Gaussian blur approach provides a high-quality bloom effect while maintaining good
performance.

Experiment with different parameters, combine bloom with other effects like HDR rendering and tone mapping, and see how
it can transform the look and feel of your 3D scenes.
