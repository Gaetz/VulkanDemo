# Understanding Deferred Rendering in Vulkan

Deferred rendering is a powerful technique for handling complex lighting scenarios in 3D graphics. Unlike forward rendering, deferred rendering separates the geometry and lighting passes, which can significantly improve performance in scenes with many light sources. In this lesson, we'll break down how deferred rendering works by examining a Vulkan implementation.

## What is Deferred Rendering?

In traditional forward rendering, each object is rendered with all its lighting calculations in a single pass. This means that for every object, we need to consider every light in the scene, which can quickly become inefficient as the number of lights increases.

Deferred rendering takes a different approach:

1. **Geometry Pass**: Render the scene geometry and store various properties (position, normal, albedo, etc.) in multiple render targets, collectively called the G-Buffer
2. **Lighting Pass**: Process the G-Buffer to calculate lighting for the entire scene

This separation allows us to decouple the scene complexity (number of objects) from the lighting complexity (number of lights), often resulting in better performance for scenes with many light sources.

You can find a theorical description of deferred rendering here: https://en.wikipedia.org/wiki/Deferred_shading. Here is an exemple, where we first render a color pass, a depth pass and a normals pass.

![](01_deferred-rendering-pass_col.jpg) ![](02_deferred-rendering-pass_depth.jpg) ![](03_deferred-rendering-pass_normals.jpg)

We then compose everything with in the composition. Note that we compute shadows, which won't be done in this lesson, but in the shadow mapping deferred rendering.

![](04_deferred-rendering-pass_composition.jpg) 

## The G-Buffer

The G-Buffer (Geometry Buffer) consists of several textures, each storing different information about the scene:

- **Position**: World-space position of each fragment
- **Normal**: Surface normal vector at each fragment
- **Albedo**: Base color (and sometimes specular intensity in the alpha channel)
- **Depth**: Depth information for each fragment

Here's how these buffers are defined in our example code:

```cpp
// Framebuffers holding the deferred attachments
struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    VkFormat format;
};

struct FrameBuffer {
    int32_t width, height;
    VkFramebuffer frameBuffer;
    // One attachment for every component required for a deferred rendering setup
    FrameBufferAttachment position, normal, albedo;
    FrameBufferAttachment depth;
    VkRenderPass renderPass;
} offScreenFrameBuf{};
```

## Implementation Walkthrough

Let's walk through the key components of deferred rendering in Vulkan.


### Necessary structs, constructor and destructor

Our code will be contained in this class.

```cpp
#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
	int32_t debugDisplayTarget = 0;

	struct {
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} model;
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} floor;
	} textures;

	struct {
		vkglTF::Model model;
		vkglTF::Model floor;
	} models;

	struct UniformDataOffscreen  {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec4 instancePos[3];
	} uniformDataOffscreen;

	struct Light {
		glm::vec4 position;
		glm::vec3 color;
		float radius;
	};

	struct UniformDataComposition {
		Light lights[6];
		glm::vec4 viewPos;
		int debugDisplayTarget = 0;
	} uniformDataComposition;

	struct {
		vks::Buffer offscreen;
		vks::Buffer composition;
	} uniformBuffers;

	struct {
		VkPipeline offscreen{ VK_NULL_HANDLE };
		VkPipeline composition{ VK_NULL_HANDLE };
	} pipelines;
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };

	struct {
		VkDescriptorSet model{ VK_NULL_HANDLE };
		VkDescriptorSet floor{ VK_NULL_HANDLE };
		VkDescriptorSet composition{ VK_NULL_HANDLE };
	} descriptorSets;

	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

	// One sampler for the frame buffer color attachments
	VkSampler colorSampler{ VK_NULL_HANDLE };

	VkCommandBuffer offScreenCmdBuffer{ VK_NULL_HANDLE };

	// Semaphore used to synchronize between offscreen and final scene rendering
	VkSemaphore offscreenSemaphore{ VK_NULL_HANDLE };

	VulkanExample() : VulkanExampleBase()
	{
		title = "Deferred shading";
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 5.0f;
		camera.rotationSpeed = 0.25f;
		camera.position = { 2.15f, 0.3f, -8.75f };
		camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		if (device) {
			vkDestroySampler(device, colorSampler, nullptr);

			// Frame buffer

			// Color attachments
			vkDestroyImageView(device, offScreenFrameBuf.position.view, nullptr);
			vkDestroyImage(device, offScreenFrameBuf.position.image, nullptr);
			vkFreeMemory(device, offScreenFrameBuf.position.mem, nullptr);

			vkDestroyImageView(device, offScreenFrameBuf.normal.view, nullptr);
			vkDestroyImage(device, offScreenFrameBuf.normal.image, nullptr);
			vkFreeMemory(device, offScreenFrameBuf.normal.mem, nullptr);

			vkDestroyImageView(device, offScreenFrameBuf.albedo.view, nullptr);
			vkDestroyImage(device, offScreenFrameBuf.albedo.image, nullptr);
			vkFreeMemory(device, offScreenFrameBuf.albedo.mem, nullptr);

			// Depth attachment
			vkDestroyImageView(device, offScreenFrameBuf.depth.view, nullptr);
			vkDestroyImage(device, offScreenFrameBuf.depth.image, nullptr);
			vkFreeMemory(device, offScreenFrameBuf.depth.mem, nullptr);

			vkDestroyFramebuffer(device, offScreenFrameBuf.frameBuffer, nullptr);

			vkDestroyPipeline(device, pipelines.composition, nullptr);
			vkDestroyPipeline(device, pipelines.offscreen, nullptr);

			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

			// Uniform buffers
			uniformBuffers.offscreen.destroy();
			uniformBuffers.composition.destroy();

			vkDestroyRenderPass(device, offScreenFrameBuf.renderPass, nullptr);

			textures.model.colorMap.destroy();
			textures.model.normalMap.destroy();
			textures.floor.colorMap.destroy();
			textures.floor.normalMap.destroy();

			vkDestroySemaphore(device, offscreenSemaphore, nullptr);
		}
	}
    ...
```

### 1. G-Buffer Setup

First, we need to create the framebuffer attachments for our G-Buffer:

```cpp
// Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
void prepareOffscreenFramebuffer()
{
    // Note: Instead of using fixed sizes, one could also match the window size and recreate the attachments on resize
    offScreenFrameBuf.width = 2048;
    offScreenFrameBuf.height = 2048;

    // Color attachments

    // (World space) Positions
    createAttachment(
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        &offScreenFrameBuf.position);

    // (World space) Normals
    createAttachment(
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        &offScreenFrameBuf.normal);

    // Albedo (color)
    createAttachment(
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        &offScreenFrameBuf.albedo);

    // Depth attachment

    // Find a suitable depth format
    VkFormat attDepthFormat;
    VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
    assert(validDepthFormat);

    createAttachment(
        attDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        &offScreenFrameBuf.depth);
    
    // Set up separate renderpass with references to the color and depth attachments
    ...
}
```

Let's look at how the attachments are created:

```cpp
void createAttachment(
    VkFormat format,
    VkImageUsageFlagBits usage,
    FrameBufferAttachment *attachment)
{
    VkImageAspectFlags aspectMask = 0;
    VkImageLayout imageLayout;

    attachment->format = format;

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
            aspectMask |=VK_IMAGE_ASPECT_STENCIL_BIT;
        imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    assert(aspectMask > 0);

    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent.width = offScreenFrameBuf.width;
    image.extent.height = offScreenFrameBuf.height;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
    vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

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
```

Note the different formats used for each attachment:
- Position and Normal use higher precision (16-bit per channel) to store accurate spatial information
- Albedo uses standard 8-bit per channel format for color information

Next, we need to set up a render pass for the G-Buffer:

From `prepareOffscreenFramebuffer()`:
```cpp
...
   // Set up separate renderpass with references to the color and depth attachments
   std::array<VkAttachmentDescription, 4> attachmentDescs = {};
   
   // Init attachment properties
   for (uint32_t i = 0; i < 4; ++i)
   {
       attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
       attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
       attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
       attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
       attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
       if (i == 3)
       {
           attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
           attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
       }
       else
       {
           attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
           attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
       }
   }
   
   // Formats
   attachmentDescs[0].format = offScreenFrameBuf.position.format;
   attachmentDescs[1].format = offScreenFrameBuf.normal.format;
   attachmentDescs[2].format = offScreenFrameBuf.albedo.format;
   attachmentDescs[3].format = offScreenFrameBuf.depth.format;
   
   std::vector<VkAttachmentReference> colorReferences;
   colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
   colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
   colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
   
   VkAttachmentReference depthReference = {};
   depthReference.attachment = 3;
   depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   
   VkSubpassDescription subpass = {};
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
   subpass.pColorAttachments = colorReferences.data();
   subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
   subpass.pDepthStencilAttachment = &depthReference;
   
   // Use subpass dependencies for attachment layout transitions
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
   
   VkRenderPassCreateInfo renderPassInfo = {};
   renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
   renderPassInfo.pAttachments = attachmentDescs.data();
   renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
   renderPassInfo.subpassCount = 1;
   renderPassInfo.pSubpasses = &subpass;
   renderPassInfo.dependencyCount = 2;
   renderPassInfo.pDependencies = dependencies.data();
   
   VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offScreenFrameBuf.renderPass));
   
   std::array<VkImageView,4> attachments;
   attachments[0] = offScreenFrameBuf.position.view;
   attachments[1] = offScreenFrameBuf.normal.view;
   attachments[2] = offScreenFrameBuf.albedo.view;
   attachments[3] = offScreenFrameBuf.depth.view;
   
   VkFramebufferCreateInfo fbufCreateInfo = {};
   fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
   fbufCreateInfo.pNext = NULL;
   fbufCreateInfo.renderPass = offScreenFrameBuf.renderPass;
   fbufCreateInfo.pAttachments = attachments.data();
   fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
   fbufCreateInfo.width = offScreenFrameBuf.width;
   fbufCreateInfo.height = offScreenFrameBuf.height;
   fbufCreateInfo.layers = 1;
   VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offScreenFrameBuf.frameBuffer));
   
   // Create sampler to sample from the color attachments
   VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
   sampler.magFilter = VK_FILTER_NEAREST;
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

### 2. Geometry Pass (G-Buffer Generation)

In the geometry pass, we render the scene geometry to populate the G-Buffer. This is done using a special fragment shader that outputs position, normal, and albedo data to multiple render targets:

`mrt.frag` (MRT = Multiple Render Targets)
```glsl
#version 450

layout (binding = 1) uniform sampler2D samplerColor;
layout (binding = 2) uniform sampler2D samplerNormalMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inWorldPos;
layout (location = 4) in vec3 inTangent;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;

void main() 
{
    // Output world position
    outPosition = vec4(inWorldPos, 1.0);

    // Calculate normal in tangent space
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    vec3 tnorm = TBN * normalize(texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0));
    outNormal = vec4(tnorm, 1.0);

    // Output albedo (color)
    outAlbedo = texture(samplerColor, inUV);
}
```

The vertex shader provides the necessary inputs:

`mrt.vert`
```glsl
#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec3 inTangent;

layout (binding = 0) uniform UBO 
{
    mat4 projection;
    mat4 model;
    mat4 view;
    vec4 instancePos[3];
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outWorldPos;
layout (location = 4) out vec3 outTangent;

void main() 
{
    vec4 tmpPos = inPos + ubo.instancePos[gl_InstanceIndex];
    gl_Position = ubo.projection * ubo.view * ubo.model * tmpPos;
    
    outUV = inUV;

    // Vertex position in world space
    outWorldPos = vec3(ubo.model * tmpPos);
    
    // Normal in world space
    mat3 mNormal = transpose(inverse(mat3(ubo.model)));
    outNormal = mNormal * normalize(inNormal);    
    outTangent = mNormal * normalize(inTangent);
    
    // Currently just vertex color
    outColor = inColor;
}
```

The command buffer for this pass renders all scene geometry to the G-Buffer:

```cpp
// Build command buffer for rendering the scene to the offscreen frame buffer attachments
void buildDeferredCommandBuffer()
{
    if (offScreenCmdBuffer == VK_NULL_HANDLE) {
        offScreenCmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);
    }

    // Create a semaphore used to synchronize offscreen rendering and usage
    VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &offscreenSemaphore));

    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    // Clear values for all attachments written in the fragment shader
    std::array<VkClearValue,4> clearValues;
    clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[3].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass =  offScreenFrameBuf.renderPass;
    renderPassBeginInfo.framebuffer = offScreenFrameBuf.frameBuffer;
    renderPassBeginInfo.renderArea.extent.width = offScreenFrameBuf.width;
    renderPassBeginInfo.renderArea.extent.height = offScreenFrameBuf.height;
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    VK_CHECK_RESULT(vkBeginCommandBuffer(offScreenCmdBuffer, &cmdBufInfo));

    vkCmdBeginRenderPass(offScreenCmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = vks::initializers::viewport((float)offScreenFrameBuf.width, (float)offScreenFrameBuf.height, 0.0f, 1.0f);
    vkCmdSetViewport(offScreenCmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(offScreenFrameBuf.width, offScreenFrameBuf.height, 0, 0);
    vkCmdSetScissor(offScreenCmdBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);

    // Floor
    vkCmdBindDescriptorSets(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.floor, 0, nullptr);
    models.floor.draw(offScreenCmdBuffer);

    // We render multiple instances of a model
    vkCmdBindDescriptorSets(offScreenCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.model, 0, nullptr);
    models.model.bindBuffers(offScreenCmdBuffer);
    vkCmdDrawIndexed(offScreenCmdBuffer, models.model.indices.count, 3, 0, 0, 0);

    vkCmdEndRenderPass(offScreenCmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(offScreenCmdBuffer));
}
```

### 3. Setting up Descriptors for G-Buffer Rendering

To render to the G-Buffer and then read from it in the composition phase, we need to set up appropriate descriptor sets:

```cpp
void setupDescriptors()
{
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Layouts
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        // Binding 1 : Position texture target / Scene colormap
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
        // Binding 2 : Normals texture target
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
        // Binding 3 : Albedo texture target
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
        // Binding 4 : Fragment shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));
    
    // Sets
    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

    // Image descriptors for the offscreen color attachments
    VkDescriptorImageInfo texDescriptorPosition =
        vks::initializers::descriptorImageInfo(
            colorSampler,
            offScreenFrameBuf.position.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkDescriptorImageInfo texDescriptorNormal =
        vks::initializers::descriptorImageInfo(
            colorSampler,
            offScreenFrameBuf.normal.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkDescriptorImageInfo texDescriptorAlbedo =
        vks::initializers::descriptorImageInfo(
            colorSampler,
            offScreenFrameBuf.albedo.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Deferred composition descriptor set
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.composition));
    writeDescriptorSets = {
        // Binding 1 : Position texture target
        vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &texDescriptorPosition),
        // Binding 2 : Normals texture target
        vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &texDescriptorNormal),
        // Binding 3 : Albedo texture target
        vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3, &texDescriptorAlbedo),
        // Binding 4 : Fragment shader uniform buffer
        vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, &uniformBuffers.composition.descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    // Offscreen (scene)

     // Model
     VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.model));
     writeDescriptorSets = {
         // Binding 0: Vertex shader uniform buffer
         vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.offscreen.descriptor),
         // Binding 1: Color map
         vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textures.model.colorMap.descriptor),
         // Binding 2: Normal map
         vks::initializers::writeDescriptorSet(descriptorSets.model, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &textures.model.normalMap.descriptor)
     };
     vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

     // Background
     VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.floor));
     writeDescriptorSets = {
         // Binding 0: Vertex shader uniform buffer
         vks::initializers::writeDescriptorSet(descriptorSets.floor, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.offscreen.descriptor),
         // Binding 1: Color map
         vks::initializers::writeDescriptorSet(descriptorSets.floor, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textures.floor.colorMap.descriptor),
         // Binding 2: Normal map
         vks::initializers::writeDescriptorSet(descriptorSets.floor, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, &textures.floor.normalMap.descriptor)
     };
     vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}
```

### 4. Lighting Pass (Composition)

After filling the G-Buffer, we perform the lighting pass. This is where we calculate lighting for the entire scene using the data stored in the G-Buffer. The vertex shader for this pass is very simple - it just generates a full-screen quad:

`deferred.vert`
```glsl
#version 450

layout (location = 0) out vec2 outUV;

void main() 
{
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
```

The fragment shader does the heavy lifting, sampling from the G-Buffer and performing lighting calculations:

`deferred.frag`
```glsl
#version 450

layout (binding = 1) uniform sampler2D samplerposition;
layout (binding = 2) uniform sampler2D samplerNormal;
layout (binding = 3) uniform sampler2D samplerAlbedo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

struct Light {
    vec4 position;
    vec3 color;
    float radius;
};

layout (binding = 4) uniform UBO 
{
    Light lights[6];
    vec4 viewPos;
    int displayDebugTarget;
} ubo;

void main() 
{
    // Get G-Buffer values
    vec3 fragPos = texture(samplerposition, inUV).rgb;
    vec3 normal = texture(samplerNormal, inUV).rgb;
    vec4 albedo = texture(samplerAlbedo, inUV);
    
    // Debug display
    if (ubo.displayDebugTarget > 0) {
        switch (ubo.displayDebugTarget) {
            case 1: 
                outFragcolor.rgb = fragPos;
                break;
            case 2: 
                outFragcolor.rgb = normal;
                break;
            case 3: 
                outFragcolor.rgb = albedo.rgb;
                break;
            case 4: 
                outFragcolor.rgb = albedo.aaa;
                break;
        }        
        outFragcolor.a = 1.0;
        return;
    }

    // Render-target composition

    #define lightCount 6
    #define ambient 0.0
    
    // Ambient part
    vec3 fragcolor  = albedo.rgb * ambient;
    
    for(int i = 0; i < lightCount; ++i)
    {
        // Vector to light
        vec3 L = ubo.lights[i].position.xyz - fragPos;
        // Distance from light to fragment position
        float dist = length(L);

        // Viewer to fragment
        vec3 V = ubo.viewPos.xyz - fragPos;
        V = normalize(V);
        
        //if(dist < ubo.lights[i].radius)
        {
            // Light to fragment
            L = normalize(L);

            // Attenuation
            float atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

            // Diffuse part
            vec3 N = normalize(normal);
            float NdotL = max(0.0, dot(N, L));
            vec3 diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

            // Specular part
            // Specular map values are stored in alpha of albedo mrt
            vec3 R = reflect(-L, N);
            float NdotR = max(0.0, dot(R, V));
            vec3 spec = ubo.lights[i].color * albedo.a * pow(NdotR, 16.0) * atten;

            fragcolor += diff + spec;    
        }    
    }    
   
    outFragcolor = vec4(fragcolor, 1.0);    
}
```

The main composition pass is a full-screen render that applies the lighting calculations:

```cpp
void buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 0.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.composition, 0, nullptr);

        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.composition);
        
        // Final composition
        // This is done by simply drawing a full screen quad
        // The fragment shader then combines the deferred attachments into the final image
        // Note: Also used for debug display if debugDisplayTarget > 0
        vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

        drawUI(drawCmdBuffers[i]);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}
```

### 5. Pipelines creation

We create two pipelines: one for the G-Buffer generation and one for the composition pass. The G-Buffer pipeline uses multiple render targets, while the composition pipeline is a simple full-screen quad.

```cpp
void preparePipelines()
{
   // Pipeline layout
   VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
   VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));
   
   // Pipelines
   VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
   VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
   VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
   VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
   VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
   VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
   VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
   std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
   std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
   
   VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);
   pipelineCI.pInputAssemblyState = &inputAssemblyState;
   pipelineCI.pRasterizationState = &rasterizationState;
   pipelineCI.pColorBlendState = &colorBlendState;
   pipelineCI.pMultisampleState = &multisampleState;
   pipelineCI.pViewportState = &viewportState;
   pipelineCI.pDepthStencilState = &depthStencilState;
   pipelineCI.pDynamicState = &dynamicState;
   pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
   pipelineCI.pStages = shaderStages.data();
   
   // Final fullscreen composition pass pipeline
   rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
   shaderStages[0] = loadShader(getShadersPath() + "deferred/deferred.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
   shaderStages[1] = loadShader(getShadersPath() + "deferred/deferred.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
   // Empty vertex input state, vertices are generated by the vertex shader
   VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
   pipelineCI.pVertexInputState = &emptyInputState;
   VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.composition));
   
   // Vertex input state from glTF model for pipeline rendering models
   pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Tangent});
   rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
   
   // Offscreen pipeline
   shaderStages[0] = loadShader(getShadersPath() + "deferred/mrt.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
   shaderStages[1] = loadShader(getShadersPath() + "deferred/mrt.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
   
   // Separate render pass
   pipelineCI.renderPass = offScreenFrameBuf.renderPass;
   
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
   
   VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.offscreen));
}
```


### 6. Light Management and Animation

The example includes code for managing lights and even animating them:

```cpp
void updateUniformBufferComposition()
{
    // White
    uniformDataComposition.lights[0].position = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
    uniformDataComposition.lights[0].color = glm::vec3(1.5f);
    uniformDataComposition.lights[0].radius = 15.0f * 0.25f;
    // Red
    uniformDataComposition.lights[1].position = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
    uniformDataComposition.lights[1].color = glm::vec3(1.0f, 0.0f, 0.0f);
    uniformDataComposition.lights[1].radius = 15.0f;
    // Blue
    uniformDataComposition.lights[2].position = glm::vec4(2.0f, -1.0f, 0.0f, 0.0f);
    uniformDataComposition.lights[2].color = glm::vec3(0.0f, 0.0f, 2.5f);
    uniformDataComposition.lights[2].radius = 5.0f;
    // Yellow
    uniformDataComposition.lights[3].position = glm::vec4(0.0f, -0.9f, 0.5f, 0.0f);
    uniformDataComposition.lights[3].color = glm::vec3(1.0f, 1.0f, 0.0f);
    uniformDataComposition.lights[3].radius = 2.0f;
    // Green
    uniformDataComposition.lights[4].position = glm::vec4(0.0f, -0.5f, 0.0f, 0.0f);
    uniformDataComposition.lights[4].color = glm::vec3(0.0f, 1.0f, 0.2f);
    uniformDataComposition.lights[4].radius = 5.0f;
    // Yellow
    uniformDataComposition.lights[5].position = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
    uniformDataComposition.lights[5].color = glm::vec3(1.0f, 0.7f, 0.3f);
    uniformDataComposition.lights[5].radius = 25.0f;

    // Animate the lights
    if (!paused) {
        uniformDataComposition.lights[0].position.x = sin(glm::radians(360.0f * timer)) * 5.0f;
        uniformDataComposition.lights[0].position.z = cos(glm::radians(360.0f * timer)) * 5.0f;

        uniformDataComposition.lights[1].position.x = -4.0f + sin(glm::radians(360.0f * timer) + 45.0f) * 2.0f;
        uniformDataComposition.lights[1].position.z = 0.0f + cos(glm::radians(360.0f * timer) + 45.0f) * 2.0f;

        uniformDataComposition.lights[2].position.x = 4.0f + sin(glm::radians(360.0f * timer)) * 2.0f;
        uniformDataComposition.lights[2].position.z = 0.0f + cos(glm::radians(360.0f * timer)) * 2.0f;

        uniformDataComposition.lights[4].position.x = 0.0f + sin(glm::radians(360.0f * timer + 90.0f)) * 5.0f;
        uniformDataComposition.lights[4].position.z = 0.0f - cos(glm::radians(360.0f * timer + 45.0f)) * 5.0f;

        uniformDataComposition.lights[5].position.x = 0.0f + sin(glm::radians(-360.0f * timer + 135.0f)) * 10.0f;
        uniformDataComposition.lights[5].position.z = 0.0f - cos(glm::radians(-360.0f * timer - 45.0f)) * 10.0f;
    }

    // Current view position
    uniformDataComposition.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

    uniformDataComposition.debugDisplayTarget = debugDisplayTarget;

    memcpy(uniformBuffers.composition.mapped, &uniformDataComposition, sizeof(UniformDataComposition));
}
```
### 6. Preparing uniform buffers

We finally need to prepare the uniform buffers for the G-Buffer and composition passes. The update functions will also be called in the main loop to update the uniform data:
```cpp
// Prepare and initialize uniform buffer containing shader uniforms
void prepareUniformBuffers()
{
     // Offscreen vertex shader
     VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformBuffers.offscreen, sizeof(UniformDataOffscreen)));

     // Deferred fragment shader
     VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformBuffers.composition, sizeof(UniformDataComposition)));

     // Map persistent
     VK_CHECK_RESULT(uniformBuffers.offscreen.map());
     VK_CHECK_RESULT(uniformBuffers.composition.map());

     // Setup instanced model positions
     uniformDataOffscreen.instancePos[0] = glm::vec4(0.0f);
     uniformDataOffscreen.instancePos[1] = glm::vec4(-4.0f, 0.0, -4.0f, 0.0f);
     uniformDataOffscreen.instancePos[2] = glm::vec4(4.0f, 0.0, -4.0f, 0.0f);

     // Update
     updateUniformBufferOffscreen();
     updateUniformBufferComposition();
}

// Update matrices used for the offscreen rendering of the scene
void updateUniformBufferOffscreen()
{
     uniformDataOffscreen.projection = camera.matrices.perspective;
     uniformDataOffscreen.view = camera.matrices.view;
     uniformDataOffscreen.model = glm::mat4(1.0f);
     memcpy(uniformBuffers.offscreen.mapped, &uniformDataOffscreen, sizeof(UniformDataOffscreen));
}

// Update lights and parameters passed to the composition shaders
void updateUniformBufferComposition()
{
     // White
     uniformDataComposition.lights[0].position = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
     uniformDataComposition.lights[0].color = glm::vec3(1.5f);
     uniformDataComposition.lights[0].radius = 15.0f * 0.25f;
     // Red
     uniformDataComposition.lights[1].position = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
     uniformDataComposition.lights[1].color = glm::vec3(1.0f, 0.0f, 0.0f);
     uniformDataComposition.lights[1].radius = 15.0f;
     // Blue
     uniformDataComposition.lights[2].position = glm::vec4(2.0f, -1.0f, 0.0f, 0.0f);
     uniformDataComposition.lights[2].color = glm::vec3(0.0f, 0.0f, 2.5f);
     uniformDataComposition.lights[2].radius = 5.0f;
     // Yellow
     uniformDataComposition.lights[3].position = glm::vec4(0.0f, -0.9f, 0.5f, 0.0f);
     uniformDataComposition.lights[3].color = glm::vec3(1.0f, 1.0f, 0.0f);
     uniformDataComposition.lights[3].radius = 2.0f;
     // Green
     uniformDataComposition.lights[4].position = glm::vec4(0.0f, -0.5f, 0.0f, 0.0f);
     uniformDataComposition.lights[4].color = glm::vec3(0.0f, 1.0f, 0.2f);
     uniformDataComposition.lights[4].radius = 5.0f;
     // Yellow
     uniformDataComposition.lights[5].position = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
     uniformDataComposition.lights[5].color = glm::vec3(1.0f, 0.7f, 0.3f);
     uniformDataComposition.lights[5].radius = 25.0f;

     // Animate the lights
     if (!paused) {
         uniformDataComposition.lights[0].position.x = sin(glm::radians(360.0f * timer)) * 5.0f;
         uniformDataComposition.lights[0].position.z = cos(glm::radians(360.0f * timer)) * 5.0f;

         uniformDataComposition.lights[1].position.x = -4.0f + sin(glm::radians(360.0f * timer) + 45.0f) * 2.0f;
         uniformDataComposition.lights[1].position.z = 0.0f + cos(glm::radians(360.0f * timer) + 45.0f) * 2.0f;

         uniformDataComposition.lights[2].position.x = 4.0f + sin(glm::radians(360.0f * timer)) * 2.0f;
         uniformDataComposition.lights[2].position.z = 0.0f + cos(glm::radians(360.0f * timer)) * 2.0f;

         uniformDataComposition.lights[4].position.x = 0.0f + sin(glm::radians(360.0f * timer + 90.0f)) * 5.0f;
         uniformDataComposition.lights[4].position.z = 0.0f - cos(glm::radians(360.0f * timer + 45.0f)) * 5.0f;

         uniformDataComposition.lights[5].position.x = 0.0f + sin(glm::radians(-360.0f * timer + 135.0f)) * 10.0f;
         uniformDataComposition.lights[5].position.z = 0.0f - cos(glm::radians(-360.0f * timer - 45.0f)) * 10.0f;
     }

     // Current view position
     uniformDataComposition.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

     uniformDataComposition.debugDisplayTarget = debugDisplayTarget;

     memcpy(uniformBuffers.composition.mapped, &uniformDataComposition, sizeof(UniformDataComposition));
}
```

### 7. Wrapping up preparations

We can now call all preparation function:
```cpp
 void prepare()
 {
     VulkanExampleBase::prepare();
     loadAssets();
     prepareOffscreenFramebuffer();
     prepareUniformBuffers();
     setupDescriptors();
     preparePipelines();
     buildCommandBuffers();
     buildDeferredCommandBuffer();
     prepared = true;
 }
```


### 8. Rendering Process Flow and Synchronization

A key aspect of deferred rendering is coordinating the sequence of render passes. The example handles this with explicit synchronization:

```cpp
void draw()
{
    VulkanExampleBase::prepareFrame();

    // The scene render command buffer has to wait for the offscreen
    // rendering to be finished before we can use the framebuffer
    // color image for sampling during final rendering
    // To ensure this we use a dedicated offscreen synchronization
    // semaphore that will be signaled when offscreen rendering
    // has been finished
    // This is necessary as an implementation may start both
    // command buffers at the same time, there is no guarantee
    // that command buffers will be executed in the order they
    // have been submitted by the application

    // Offscreen rendering

    // Wait for swap chain presentation to finish
    submitInfo.pWaitSemaphores = &semaphores.presentComplete;
    // Signal ready with offscreen semaphore
    submitInfo.pSignalSemaphores = &offscreenSemaphore;

    // Submit work
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &offScreenCmdBuffer;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    // Scene rendering

    // Wait for offscreen semaphore
    submitInfo.pWaitSemaphores = &offscreenSemaphore;
    // Signal ready with render complete semaphore
    submitInfo.pSignalSemaphores = &semaphores.renderComplete;

    // Submit work
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VulkanExampleBase::submitFrame();
}

virtual void render()
{
    if (!prepared)
       return;
    updateUniformBufferComposition();
    updateUniformBufferOffscreen();
    draw();
}
```

This ensures that the offscreen G-Buffer rendering completes before the composition pass begins.

## The Rendering Process Flow

Here's the complete deferred rendering process:

1. **First Pass (Geometry/G-Buffer Pass)**:
   - Render scene geometry to multiple render targets (position, normal, albedo)
   - No lighting calculations are performed yet
   - Output is saved to the G-Buffer textures

2. **Second Pass (Lighting/Composition Pass)**:
   - Render a full-screen quad
   - Sample from the G-Buffer textures to get position, normal, and albedo data
   - Calculate lighting for each fragment based on the G-Buffer data
   - Output the final lit scene

## Debugging Features

The example includes a useful debugging feature that allows viewing the individual components of the G-Buffer:

`deferred.frag`
```glsl
// Debug display
if (ubo.displayDebugTarget > 0) {
    switch (ubo.displayDebugTarget) {
        case 1: 
            outFragcolor.rgb = fragPos;  // World positions
            break;
        case 2: 
            outFragcolor.rgb = normal;   // Normals
            break;
        case 3: 
            outFragcolor.rgb = albedo.rgb; // Albedo (color)
            break;
        case 4: 
            outFragcolor.rgb = albedo.aaa; // Specular intensity from alpha
            break;
    }        
    outFragcolor.a = 1.0;
    return;
}
```

This is toggled via a UI dropdown:

```cpp
virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
{
    if (overlay->header("Settings")) {
        overlay->comboBox("Display", &debugDisplayTarget, { "Final composition", "Position", "Normals", "Albedo", "Specular" });
    }
}
```

## Advantages of Deferred Rendering

1. **Decoupled complexity**: Scene complexity (number of objects) is separated from lighting complexity (number of lights)
2. **Efficient for many lights**: Lighting calculations are performed once per pixel, not per object-light combination
3. **Consistent performance**: Rendering time is more predictable and less affected by scene complexity
4. **Post-processing friendly**: Having scene data in the G-Buffer makes many post-processing effects easier to implement

## Limitations of Deferred Rendering

1. **Memory usage**: Requires more memory for the G-Buffer textures
2. **Bandwidth**: Higher bandwidth requirements for reading/writing the G-Buffer
3. **Transparency challenges**: Transparent objects typically need to be rendered using forward rendering in a separate pass
4. **Anti-aliasing limitations**: Techniques like MSAA are more complicated to implement with deferred rendering
5. **Material variety**: More complex to implement a wide variety of materials compared to forward rendering

## Performance Considerations

### G-Buffer Format Selection

The format of each G-Buffer attachment is crucial for performance and quality:

```cpp
// (World space) Positions - high precision needed
createAttachment(
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    &offScreenFrameBuf.position);

// (World space) Normals - high precision needed
createAttachment(
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    &offScreenFrameBuf.normal);

// Albedo (color) - standard precision sufficient
createAttachment(
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    &offScreenFrameBuf.albedo);
```

Using 16-bit floating-point (SFLOAT) formats for position and normal data ensures sufficient precision for lighting calculations, while standard 8-bit formats are adequate for color data.

### Memory Bandwidth

Deferred rendering can be bandwidth-intensive due to reading and writing the G-Buffer. The example uses 16-bit precision for position and normal data, which provides a good balance between quality and memory usage.

## Extensions and Further Development

This basic deferred rendering implementation can be extended in several ways:

1. **Shadow mapping**: Add shadow maps to enhance the lighting
2. **Screen-space ambient occlusion (SSAO)**: Improve the look of ambient lighting
3. **Global illumination**: Add indirect lighting using techniques like screen-space reflections
4. **Tone mapping**: Implement HDR rendering with tone mapping
5. **Temporal anti-aliasing**: Add TAA to improve image quality
6. **Material system**: Implement PBR (Physically Based Rendering) materials

## Conclusion

Deferred rendering is a powerful technique for handling scenes with many light sources. By separating the geometry and lighting passes, we can achieve better performance scaling when dealing with complex lighting scenarios.

The implementation in this example demonstrates how to create a G-Buffer in Vulkan, populate it with scene data, and then use that data to perform lighting calculations in a separate pass. The code also shows proper synchronization between the passes using Vulkan semaphores.

Key takeaways:
- Deferred rendering separates geometry and lighting passes
- The G-Buffer stores position, normal, and albedo data
- Lighting calculations are performed in a separate pass
- Multiple lights can be processed efficiently
- Proper synchronization is essential between the passes

By understanding the fundamentals of deferred rendering presented here, you'll be well-prepared to implement and extend this technique in your own graphics applications.
