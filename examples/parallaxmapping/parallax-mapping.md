# Parallax Mapping in Vulkan

Parallax mapping is a technique used to add depth to flat surfaces in 3D graphics. Unlike normal mapping, which only affects the lighting calculations, parallax mapping actually modifies the texture coordinates to create the illusion that the surface has real geometric depth. This tutorial will walk you through a Vulkan implementation of parallax mapping.

## What is Parallax Mapping?

Parallax mapping simulates depth by offsetting texture coordinates based on the view angle and a height map. This creates a convincing illusion of depth without adding extra geometry. There are several variations of this technique:

1. **Basic parallax mapping**: A simple offset based on the height map
2. **Steep parallax mapping**: Uses multiple samples to find a more accurate intersection point
3. **Parallax occlusion mapping (POM)**: An advanced technique that accounts for self-shadowing

In this tutorial, we'll implement all three methods and let you switch between them to see the differences.

## Project Setup

Let's start by examining the class structure and basic setup for our Vulkan parallax mapping example:

```cpp
#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
    // Texture resources that we'll need for parallax mapping
    // We need a color texture (for the visible surface color) and a special texture
    // that combines normal map and height map information
    struct {
        vks::Texture2D colorMap;        // Regular color/diffuse texture - gives the surface its base colors
        // Normals and height are combined into one texture (height = alpha channel)
        // Normal map gives surface detail without extra geometry
        // Height map tells us how "deep" each point should appear for parallax effect
        vks::Texture2D normalHeightMap;
    } textures;

    // Our 3D model - just a simple plane that we'll apply parallax mapping to
    // We use a flat surface to show how parallax mapping creates the illusion of depth
    vkglTF::Model plane;

    // Vertex shader uniform buffer - contains data needed by the vertex shader
    // Uniforms are a way to send data from CPU to GPU shaders
    struct UniformDataVertexShader {
        glm::mat4 projection;    // Projection matrix
        glm::mat4 view;          // View matrix
        glm::mat4 model;         // Model matrix
        glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 0.0f, 1.0f);  // Light position
        glm::vec4 cameraPos;     // Camera position for parallax calculations - needed to calculate view angle
    } uniformDataVertexShader;

    // Fragment shader uniform buffer - contains parallax mapping parameters
    // These control how the parallax effect looks
    struct UniformDataFragmentShader {
        float heightScale = 0.1f;    // Controls how "deep" the parallax effect appears - higher values = deeper effect
        // Basic parallax mapping needs a bias to look good - helps avoid artifacts
        float parallaxBias = -0.02f;
        // Number of layers for steep parallax and parallax occlusion - more layers = better quality but slower
        float numLayers = 48.0f;
        // Which mapping technique to use (0-4)
        // 0 = color only, 1 = normal mapping, 2 = basic parallax, 3 = steep parallax, 4 = parallax occlusion
        int32_t mappingMode = 4;
    } uniformDataFragmentShader;

    // Uniform buffers - these hold our uniform data on the GPU
    struct {
        vks::Buffer vertexShader;
        vks::Buffer fragmentShader;
    } uniformBuffers;

    // Vulkan pipeline objects
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    // Names of the different mapping modes for UI display
    // These techniques become progressively more complex and realistic
    const std::vector<std::string> mappingModes = {
        "Color only",                // Just the base color texture
        "Normal mapping",            // Adds surface detail with normal maps
        "Parallax mapping",          // Basic parallax for simple depth illusion
        "Steep parallax mapping",    // Better parallax with multiple samples
        "Parallax occlusion mapping", // Best quality parallax with sample interpolation
    };

    // Constructor
    VulkanExample() : VulkanExampleBase()
    {
        title = "Parallax Mapping";
        timerSpeed *= 0.5f;  // Slow down the timer for smoother animation
        
        // Set up the camera - we use first-person camera to better see the parallax effect
        camera.type = Camera::CameraType::firstperson;
        camera.setPosition(glm::vec3(0.0f, 1.25f, -1.5f));
        camera.setRotation(glm::vec3(-45.0f, 0.0f, 0.0f));
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
    }

    // Destructor - clean up Vulkan resources
    ~VulkanExample()
    {
        if (device) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            uniformBuffers.vertexShader.destroy();
            uniformBuffers.fragmentShader.destroy();
            textures.colorMap.destroy();
            textures.normalHeightMap.destroy();
        }
    }
    
    // The rest of the implementation...
};
```

This sets up our basic class structure. The key elements are:

- A normal/height map texture where the height is stored in the alpha channel
- Uniform buffers for vertex and fragment shaders with parameters to control the parallax effect
- Support for multiple mapping techniques that can be switched at runtime

## Loading Assets

Next, let's look at how we load the 3D model and textures:

```cpp
void loadAssets()
{
    // Set flags for glTF model loading
    const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | 
                                     vkglTF::FileLoadingFlags::PreMultiplyVertexColors | 
                                     vkglTF::FileLoadingFlags::FlipY;
    
    // Load a simple plane model
    plane.loadFromFile(getAssetPath() + "models/plane.gltf", vulkanDevice, queue, glTFLoadingFlags);
    
    // Load the normal-height map (combined in one texture, height in alpha channel)
    // Normal data is in RGB channels (surface direction) and height data is in alpha channel (depth)
    // This is a common way to store this data efficiently in one texture
    textures.normalHeightMap.loadFromFile(getAssetPath() + "textures/rocks_normal_height_rgba.ktx", 
                                         VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
    
    // Load the color/diffuse map
    textures.colorMap.loadFromFile(getAssetPath() + "textures/rocks_color_rgba.ktx", 
                                  VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
}
```

We're loading three key assets:

1. A simple plane model that will serve as our rendering surface
2. A normal map texture that includes height information in its alpha channel
3. A color (diffuse) texture that provides the base colors

For parallax mapping, the height map is crucial - it tells us how "deep" each point on the surface should appear.

## Setting Up Descriptors

The descriptors connect our shader resources (like uniforms and textures) to the GPU:

```cpp
void setupDescriptors()
{
    // Create a descriptor pool with enough space for our descriptors
    // A descriptor pool is like a memory pool for descriptor sets
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Set up the descriptor layout (binding points for shader resources)
    // This defines what resources the shaders can access and at which binding points
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0: Vertex shader uniform buffer - contains transformation matrices and other data
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
                                                     VK_SHADER_STAGE_VERTEX_BIT, 0),
        // Binding 1: Fragment shader color map image sampler
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
                                                     VK_SHADER_STAGE_FRAGMENT_BIT, 1),
        // Binding 2: Fragment combined normal and heightmap
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
                                                     VK_SHADER_STAGE_FRAGMENT_BIT, 2),
        // Binding 3: Fragment shader uniform buffer - contains parallax parameters
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
                                                     VK_SHADER_STAGE_FRAGMENT_BIT, 3),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = 
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

    // Create the descriptor set - an actual instance of the layout with real resources
    VkDescriptorSetAllocateInfo allocInfo = 
        vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    
    // Update the descriptor set with actual buffer and image info
    // This connects our actual resources to the shader binding points
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
        // Binding 0: Vertex shader uniform buffer - for transforms and lighting
        vks::initializers::writeDescriptorSet(descriptorSet, 
                                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, 
                                             &uniformBuffers.vertexShader.descriptor),
        // Binding 1: Fragment shader color map - for surface color
        vks::initializers::writeDescriptorSet(descriptorSet, 
                                             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, 
                                             &textures.colorMap.descriptor),
        // Binding 2: Combined normal and heightmap - for surface detail and parallax depth
        vks::initializers::writeDescriptorSet(descriptorSet, 
                                             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, 
                                             &textures.normalHeightMap.descriptor),
        // Binding 3: Fragment shader uniform buffer - for parallax parameters
        vks::initializers::writeDescriptorSet(descriptorSet, 
                                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, 
                                             &uniformBuffers.fragmentShader.descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), 
                          writeDescriptorSets.data(), 0, NULL);
}
```

In this function, we:

1. Create a descriptor pool that will hold our descriptors
2. Define the layout of our descriptor set with four bindings:
   - Vertex shader uniforms (matrices, light position)
   - Color texture
   - Normal/height texture
   - Fragment shader uniforms (parallax parameters)
3. Allocate a descriptor set from the pool
4. Update the descriptor set with our actual buffer and texture resources

These descriptors will be used by the shaders to access these resources.

## Creating Pipelines

Now let's create our graphics pipeline:

```cpp
void preparePipelines()
{
    // Create the pipeline layout using our descriptor set layout
    // Pipeline layout defines what descriptor sets the pipeline can access
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = 
        vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    // Set up common pipeline state - these define how rendering will work
    
    // Input assembly - how to interpret vertex data (as triangles, lines, etc.)
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = 
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    
    // Rasterization - how to convert vertices to fragments/pixels
    VkPipelineRasterizationStateCreateInfo rasterizationState = 
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    
    // Color blending - how to combine fragment colors with the framebuffer
    VkPipelineColorBlendAttachmentState blendAttachmentState = 
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    
    VkPipelineColorBlendStateCreateInfo colorBlendState = 
        vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    
    // Depth and stencil - for handling 3D depth information
    VkPipelineDepthStencilStateCreateInfo depthStencilState = 
        vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    
    // Viewport - region of the framebuffer to render to
    VkPipelineViewportStateCreateInfo viewportState = 
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    
    // Multisampling - for anti-aliasing (not used here, just 1 sample per pixel)
    VkPipelineMultisampleStateCreateInfo multisampleState = 
        vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
    
    // Dynamic state - aspects of the pipeline that can change without recreating the whole pipeline
    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT, 
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState = 
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    
    // Array to hold our shader stages (vertex and fragment)
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // Create the main pipeline object
    VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(
        pipelineLayout, renderPass);
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    
    // Set up vertex input state based on the model's vertex format
    // For parallax mapping, we need position, UV coords, normal, and tangent vectors
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ 
        vkglTF::VertexComponent::Position, 
        vkglTF::VertexComponent::UV, 
        vkglTF::VertexComponent::Normal, 
        vkglTF::VertexComponent::Tangent 
    });

    // Load shaders
    shaderStages[0] = loadShader(getShadersPath() + "parallaxmapping/parallax.vert.spv", 
                                VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "parallaxmapping/parallax.frag.spv", 
                                VK_SHADER_STAGE_FRAGMENT_BIT);
    
    // Create the pipeline
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));
}
```

In this function, we:

1. Create a pipeline layout from our descriptor set layout
2. Configure various pipeline states (rasterization, blending, depth test, etc.)
3. Set up the vertex input format - note that we need tangent vectors for parallax mapping
4. Load our shaders for parallax mapping
5. Create the graphics pipeline

## Preparing Uniform Buffers

Now let's set up the uniform buffers that will contain our data for shaders:

```cpp
void prepareUniformBuffers()
{
    // Create the vertex shader uniform buffer - for transformation matrices and lighting
    // This allocates GPU memory for our uniform data
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &uniformBuffers.vertexShader,
        sizeof(UniformDataVertexShader)));

    // Create the fragment shader uniform buffer - for parallax parameters
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &uniformBuffers.fragmentShader,
        sizeof(UniformDataFragmentShader)));
    
    // Map the buffers so we can write to them directly from CPU memory
    // This creates a CPU-accessible pointer to the GPU memory
    VK_CHECK_RESULT(uniformBuffers.vertexShader.map());
    VK_CHECK_RESULT(uniformBuffers.fragmentShader.map());
    
    // Initialize the uniform buffers with current values
    updateUniformBuffers();
}
```

This function:

1. Creates two uniform buffers - one for the vertex shader and one for the fragment shader
2. Maps them to CPU memory so we can update them easily
3. Initializes them with current values

## Updating Uniform Buffers

We need to update our uniform buffers whenever the camera moves or parameters change:

```cpp
void updateUniformBuffers()
{
    // Update vertex shader uniforms
    uniformDataVertexShader.projection = camera.matrices.perspective;
    uniformDataVertexShader.view = camera.matrices.view;
    uniformDataVertexShader.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f));

    // Animate the light position in a circular motion if not paused
    // This makes the lighting change dynamically to better see the parallax effect
    if (!paused) {
        uniformDataVertexShader.lightPos.x = sin(glm::radians(timer * 360.0f)) * 1.5f;
        uniformDataVertexShader.lightPos.z = cos(glm::radians(timer * 360.0f)) * 1.5f;
    }

    // Update the camera position
    // The camera position is needed to calculate view direction for parallax effect
    uniformDataVertexShader.cameraPos = glm::vec4(camera.position, -1.0f) * -1.0f;
    
    // Copy the data to the mapped buffer - this updates the GPU memory with our new values
    memcpy(uniformBuffers.vertexShader.mapped, &uniformDataVertexShader, sizeof(UniformDataVertexShader));

    // Update fragment shader uniforms (parallax mapping parameters)
    // We just copy the current state of the fragment shader uniforms
    memcpy(uniformBuffers.fragmentShader.mapped, &uniformDataFragmentShader, sizeof(UniformDataFragmentShader));
}
```

This function:

1. Updates the transform matrices based on camera state
2. Animates the light position in a circular motion
3. Updates the camera position (needed for parallax calculations)
4. Copies all the data to the mapped uniform buffers

## Command Buffer Building

Let's look at how we build the command buffer for rendering:

```cpp
void buildCommandBuffers()
{
    // Command buffers contain the actual rendering commands that the GPU will execute
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    // Set clear values for the framebuffer
    VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;           // Background color
    clearValues[1].depthStencil = { 1.0f, 0 };          // Depth clear value (1.0 = farthest)

    // Set up render pass
    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    // For each frame buffer (double/triple buffering) - we need to record commands for each
    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        // Set target frame buffer for this command buffer
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        // Begin command buffer recording - start adding commands to the buffer
        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        // Begin the render pass - starts rendering to the framebuffer
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Set viewport - the region of the framebuffer to render to
        VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        // Set scissor rectangle - cropping region
        VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        // Bind descriptor sets - connect our resources (uniforms, textures) to the pipeline
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, 
                               pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        
        // Bind pipeline
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        
        // Draw the plane model - this sends the actual draw command to render our model
        plane.draw(drawCmdBuffers[i]);

        // Draw UI overlay
        drawUI(drawCmdBuffers[i]);

        // End render pass - finish rendering to the framebuffer
        vkCmdEndRenderPass(drawCmdBuffers[i]);

        // End command buffer recording - finalize the command buffer
        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}
```

This function:

1. Prepares command buffer recording
2. Sets up framebuffer clearing
3. Begins the render pass
4. Sets viewport and scissor
5. Binds descriptors and pipeline
6. Draws our plane model
7. Draws UI
8. Finalizes command buffer recording

## Main Draw and Render Functions

Here are the main rendering functions:

```cpp
void draw()
{
    // Prepare frame (acquire next swap chain image)
    // This gets the next available image from the swap chain to render to
    VulkanExampleBase::prepareFrame();
    
    // Submit the command buffer
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    
    // Present the frame
    VulkanExampleBase::submitFrame();
}

virtual void render()
{
    if (!prepared)
        return;
    
    // Update uniforms if needed (camera moved or not paused)
    if (!paused || camera.updated) {
        updateUniformBuffers();
    }
    
    // Draw the frame
    draw();
}
```

These functions:

1. Prepare the frame by acquiring the next image from the swap chain
2. Submit the command buffer for rendering
3. Present the rendered frame to the display
4. Update uniform buffers when necessary

## UI Overlay

We also have a UI for switching between different mapping modes:

```cpp
virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
{
    if (overlay->header("Settings")) {
        // Create a combo box to switch between mapping modes
        if (overlay->comboBox("Mode", &uniformDataFragmentShader.mappingMode, mappingModes)) {
            // Update uniforms when the mapping mode changes
            updateUniformBuffers();
        }
    }
}
```

This function adds a combo box to the UI that lets us switch between different mapping modes.

## Program Entry Point

Finally, we have the main entry point:

```cpp
VULKAN_EXAMPLE_MAIN()
```

This macro defines the program's entry point and creates our `VulkanExample` class.

## Prepare Function

This is our main setup function that calls all the preparatory functions:

```cpp
void prepare()
{
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    buildCommandBuffers();
    prepared = true;
}
```

This function:

1. Calls the base class preparation
2. Loads 3D models and textures
3. Sets up uniform buffers
4. Sets up descriptors
5. Creates pipelines
6. Builds command buffers
7. Marks the application as prepared

## Understanding the Parallax Mapping Shader

To fully understand how parallax mapping works, we need to examine the fragment shader:

`parallax.frag`
```glsl
#version 450

// Input texture samplers
layout (binding = 1) uniform sampler2D sColorMap;         // Diffuse/color texture
layout (binding = 2) uniform sampler2D sNormalHeightMap;  // Combined normal map and height map

// Uniform buffer for parallax mapping parameters
layout (binding = 3) uniform UBO
{
   float heightScale;   // Controls how "deep" the parallax effect appears
   float parallaxBias;  // Bias adjustment for basic parallax mapping
   float numLayers;     // Number of layers to sample in steep/occlusion parallax mapping
   int mappingMode;     // Which mapping technique to use (0-4)
} ubo;

// Input variables from the vertex shader (interpolated across the triangle)
layout (location = 0) in vec2 inUV;                // Texture coordinates
layout (location = 1) in vec3 inTangentLightPos;   // Light position in tangent space
layout (location = 2) in vec3 inTangentViewPos;    // Camera position in tangent space
layout (location = 3) in vec3 inTangentFragPos;    // Fragment position in tangent space

// Output color for this fragment
layout (location = 0) out vec4 outColor;

// Basic Parallax Mapping function
// This is the simplest form of parallax mapping
vec2 parallaxMapping(vec2 uv, vec3 viewDir)
{
   // Get height from the alpha channel of the normal-height map
   // We invert it (1.0 - value) because in the texture:
   // - 0 typically means deepest point 
   // - 1 typically means highest point
   float height = 1.0 - textureLod(sNormalHeightMap, uv, 0.0).a;
   
   // Calculate the UV offset:
   // - viewDir.xy / viewDir.z controls how much shift based on view angle
   // - height * heightScale * 0.5 controls how much depth effect
   // - parallaxBias is added to adjust the overall effect
   vec2 p = viewDir.xy * (height * (ubo.heightScale * 0.5) + ubo.parallaxBias) / viewDir.z;
   
   // Return shifted texture coordinates
   // We subtract the offset because looking at an angle should shift UVs in the opposite direction
   return uv - p;
}

// Steep Parallax Mapping function
// Better quality than basic parallax by using multiple samples
vec2 steepParallaxMapping(vec2 uv, vec3 viewDir)
{
   // Calculate the depth of each layer (divide total 1.0 depth into uniform layers)
   float layerDepth = 1.0 / ubo.numLayers;
   
   // Start at the surface (depth 0)
   float currLayerDepth = 0.0;
   
   // Calculate how much to shift UVs at each step based on view angle
   vec2 deltaUV = viewDir.xy * ubo.heightScale / (viewDir.z * ubo.numLayers);
   
   // Start from the original texture coordinates
   vec2 currUV = uv;
   
   // Get initial height at the current position
   float height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;
   
   // Step through layers until the ray intersects the height field
   // This is like a linear search to find the intersection point
   for (int i = 0; i < ubo.numLayers; i++) {
      // Move to the next layer depth
      currLayerDepth += layerDepth;
      
      // Shift UVs inward at each step (deeper into the surface)
      currUV -= deltaUV;
      
      // Sample the height at this new position
      height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;
      
      // If we've hit or passed through the height field, we've found our intersection
      if (height < currLayerDepth) {
         break;
      }
   }
   
   // Return the UV coordinates where the ray intersects the height field
   return currUV;
}

// Parallax Occlusion Mapping function
// Highest quality parallax mapping with smooth transitions
vec2 parallaxOcclusionMapping(vec2 uv, vec3 viewDir)
{
   // Initial setup - same as steep parallax mapping
   float layerDepth = 1.0 / ubo.numLayers;
   float currLayerDepth = 0.0;
   vec2 deltaUV = viewDir.xy * ubo.heightScale / (viewDir.z * ubo.numLayers);
   vec2 currUV = uv;
   float height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;
   
   // Find intersection point - same as steep parallax mapping
   for (int i = 0; i < ubo.numLayers; i++) {
      currLayerDepth += layerDepth;
      currUV -= deltaUV;
      height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;
      if (height < currLayerDepth) {
         break;
      }
   }
   
   // Here's where POM differs from steep parallax mapping:
   // Instead of just using the intersection point, we interpolate between two layers
   
   // Get the UV coordinates of the previous layer (before we hit the surface)
   vec2 prevUV = currUV + deltaUV;
   
   // Calculate how far the view ray is inside the surface at the current layer
   float nextDepth = height - currLayerDepth;
   
   // Calculate how far the view ray was above the surface at the previous layer
   float prevDepth = 1.0 - textureLod(sNormalHeightMap, prevUV, 0.0).a - currLayerDepth + layerDepth;
   
   // Interpolate between the two UV sets for smoother results
   // The weight is based on how close we are to the actual intersection point
   return mix(currUV, prevUV, nextDepth / (nextDepth - prevDepth));
}

void main(void)
{
   // Calculate view direction (normalized vector from fragment to viewer)
   vec3 V = normalize(inTangentViewPos - inTangentFragPos);
   
   // Start with original texture coordinates
   vec2 uv = inUV;

   if (ubo.mappingMode == 0) {
      // Mode 0: Color only (no depth effects)
      // Just use the color texture with original UVs
      outColor = texture(sColorMap, inUV);
   } else {
      // For other modes, we apply normal mapping and/or parallax mapping
      
      // Choose which parallax mapping technique to use based on mode
      switch(ubo.mappingMode) {
         case 2: // Basic parallax mapping
              uv = parallaxMapping(inUV, V);
              break;
         case 3: // Steep parallax mapping
              uv = steepParallaxMapping(inUV, V);
              break;
         case 4: // Parallax occlusion mapping
              uv = parallaxOcclusionMapping(inUV, V);
              break;
         // Note: Mode 1 (normal mapping only) doesn't modify UVs
      }

      // Perform sampling before potentially discarding fragments
      // This is important for GPU performance reasons - avoids derivative issues
      vec3 normalHeightMapLod = textureLod(sNormalHeightMap, uv, 0.0).rgb;
      vec3 color = texture(sColorMap, uv).rgb;

      // Discard fragments that would sample outside the texture
      // This prevents artifacts at the edges of surfaces
      if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
         discard;
      }

      // Extract normal from the normal map
      // Convert from [0,1] color range to [-1,1] normal vector range
      vec3 N = normalize(normalHeightMapLod * 2.0 - 1.0);
      
      // Calculate light direction (from fragment to light)
      vec3 L = normalize(inTangentLightPos - inTangentFragPos);
      
      // Calculate reflection vector (for specular lighting)
      vec3 R = reflect(-L, N);
      
      // Calculate half vector (used for specular in Blinn-Phong model)
      vec3 H = normalize(L + V);

      // Calculate lighting components
      vec3 ambient = 0.2 * color;                           // Ambient light (constant low level)
      vec3 diffuse = max(dot(L, N), 0.0) * color;          // Diffuse light (depends on light angle)
      vec3 specular = vec3(0.15) * pow(max(dot(N, H), 0.0), 32.0); // Specular highlights

      // Combine all lighting components for final color
      outColor = vec4(ambient + diffuse + specular, 1.0f);
   }
}

```

The fragment shader implements three different parallax mapping techniques:

1. **Basic Parallax Mapping (Mode 2)**: Simply shifts the texture coordinates based on the height map and view direction. This is the simplest technique but can lead to artifacts.

2. **Steep Parallax Mapping (Mode 3)**: Uses multiple samples along the view ray to find the intersection with the height map. This provides better quality but is more expensive.

3. **Parallax Occlusion Mapping (Mode 4)**: Similar to steep parallax mapping but adds interpolation between the layers for smoother results.

The shader also has a mode for normal mapping only (Mode 1) and basic color mapping (Mode 0).

## Understanding the Vertex Shader

The vertex shader prepares data for the fragment shader:

`parallax.vert`
```glsl
#version 450

// Input attributes from the vertex buffer
layout (location = 0) in vec3 inPos;      // Vertex position in model space (x, y, z)
layout (location = 1) in vec2 inUV;       // Texture coordinates (u, v)
layout (location = 2) in vec3 inNormal;   // Vertex normal vector
layout (location = 3) in vec4 inTangent;  // Tangent vector (xyz) with handedness in w component

// Uniform buffer object - contains transformation matrices and positions
// Uniforms are constant for all vertices in a draw call
layout (binding = 0) uniform UBO
{
   mat4 projection;   // Projection matrix - transforms from view space to clip space
   mat4 view;         // View matrix - transforms from world space to view space (camera)
   mat4 model;        // Model matrix - transforms from model space to world space
   vec4 lightPos;     // Light position in world space
   vec4 cameraPos;    // Camera position in world space
} ubo;

// Output variables to pass to the fragment shader
layout (location = 0) out vec2 outUV;               // Pass texture coordinates to fragment shader
layout (location = 1) out vec3 outTangentLightPos;  // Light position in tangent space
layout (location = 2) out vec3 outTangentViewPos;   // Camera position in tangent space
layout (location = 3) out vec3 outTangentFragPos;   // Fragment position in tangent space

void main(void)
{
   // Calculate final position in clip space using all transformation matrices
   gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos, 1.0f);

   // Transform the vertex position to world space (for lighting calculations)
   outTangentFragPos = vec3(ubo.model * vec4(inPos, 1.0));

   // Pass texture coordinates to the fragment shader
   outUV = inUV;

   // Create the TBN matrix to transform from world space to tangent space
   // Tangent space is a coordinate system relative to the surface of the model
   // It's essential for normal mapping and parallax mapping

   // Transform normal to world space and normalize it
   vec3 N = normalize(mat3(ubo.model) * inNormal);

   // Transform tangent to world space and normalize it
   vec3 T = normalize(mat3(ubo.model) * inTangent.xyz);

   // Calculate the bitangent vector (perpendicular to both normal and tangent)
   vec3 B = normalize(cross(N, T));

   // Create the TBN matrix (transpose is used to invert the transformation)
   // This matrix will transform vectors from world space to tangent space
   mat3 TBN = transpose(mat3(T, B, N));

   // Transform light position, camera position, and fragment position to tangent space
   // This allows us to perform all lighting calculations in tangent space
   outTangentLightPos = TBN * ubo.lightPos.xyz;  // Light position in tangent space
   outTangentViewPos  = TBN * ubo.cameraPos.xyz; // Camera position in tangent space
   outTangentFragPos  = TBN * outTangentFragPos; // Fragment position in tangent space
}
```

This vertex shader:

1. Receives vertex position, texture coordinates, normal, and tangent from the model
2. Calculates the tangent space vectors (normal, tangent, and bitangent)
3. Calculates the world position of the vertex
4. Computes vectors to the camera and light in world space
5. Passes all these to the fragment shader for parallax mapping

## How Parallax Mapping Works

Let's dive deeper into how the different parallax mapping techniques work:

### Basic Parallax Mapping

The simplest form of parallax mapping just shifts the texture coordinates based on view angle and height map:

```glsl
// Get height from the alpha channel of the normal-height map
// We invert it (1.0 - value) because in the texture:
// - 0 typically means deepest point 
// - 1 typically means highest point
float height = 1.0 - textureLod(sNormalHeightMap, uv, 0.0).a;

// Calculate the UV offset:
// - viewDir.xy / viewDir.z controls how much shift based on view angle
// - height * heightScale * 0.5 controls how much depth effect
// - parallaxBias is added to adjust the overall effect
vec2 p = viewDir.xy * (height * (ubo.heightScale * 0.5) + ubo.parallaxBias) / viewDir.z;

// Return shifted texture coordinates
// We subtract the offset because looking at an angle should shift UVs in the opposite direction
return uv - p;
```

This creates a basic parallax effect but can have artifacts. It works like this:

1. Get the height value from the height map (0 to 1)
2. Calculate how much to shift the texture coordinates based on view direction and height
3. Apply the shift to create an illusion of depth

Imagine looking at a brick wall from an angle:
- Without parallax: The bricks look flat
- With parallax: The bricks appear to have depth because the texture coordinates shift based on height

### Steep Parallax Mapping

This technique uses multiple samples along the view ray to find the intersection with the height field:

```glsl
   // Calculate the depth of each layer (divide total 1.0 depth into uniform layers)
   float layerDepth = 1.0 / ubo.numLayers;

   // Start at the surface (depth 0)
   float currLayerDepth = 0.0;

   // Calculate how much to shift UVs at each step based on view angle
   vec2 deltaUV = viewDir.xy * ubo.heightScale / (viewDir.z * ubo.numLayers);

   // Start from the original texture coordinates
   vec2 currUV = uv;

   // Get initial height at the current position
   float height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;

   // Step through layers until the ray intersects the height field
   // This is like a linear search to find the intersection point
   for (int i = 0; i < ubo.numLayers; i++) {
      // Move to the next layer depth
      currLayerDepth += layerDepth;
      
      // Shift UVs inward at each step (deeper into the surface)
      currUV -= deltaUV;
      
      // Sample the height at this new position
      height = 1.0 - textureLod(sNormalHeightMap, currUV, 0.0).a;
      
      // If we've hit or passed through the height field, we've found our intersection
      if (height < currLayerDepth) {
          break;
      }
   }

   // Return the UV coordinates where the ray intersects the height field
   return currUV;
```

This provides better quality by:

1. Dividing the height range into multiple layers
2. Starting at the surface and stepping deeper until we find where the view ray intersects the height field
3. This is like a linear search for the intersection point

### Parallax Occlusion Mapping

This technique extends steep parallax mapping by interpolating between layers for smoother results:

```glsl
// Find intersection point (same as steep parallax)
// ...

// Get the UV coordinates of the previous layer (before we hit the surface)
vec2 prevUV = currUV + deltaUV;

// Calculate how far the view ray is inside the surface at the current layer
float nextDepth = height - currLayerDepth;

// Calculate how far the view ray was above the surface at the previous layer
float prevDepth = 1.0 - textureLod(sNormalHeightMap, prevUV, 0.0).a - currLayerDepth + layerDepth;

// Interpolate between the two UV sets for smoother results
// The weight is based on how close we are to the actual intersection point
return mix(currUV, prevUV, nextDepth / (nextDepth - prevDepth));
```

This provides the highest quality but is also the most expensive.

## Conclusion

Parallax mapping is a powerful technique to add perceived depth to surfaces without adding geometry. By offsetting texture coordinates based on height information and viewing angle, we can create convincing depth illusions.

The implementation we've examined offers several techniques:

1. **Basic color mapping** - No depth effect, just texture
2. **Normal mapping** - Adds lighting detail but no depth
3. **Simple parallax mapping** - Adds basic depth illusion
4. **Steep parallax mapping** - Improves depth accuracy with multiple samples
5. **Parallax occlusion mapping** - Highest quality with interpolated samples

When implementing parallax mapping, the key considerations are:

- Using height maps (often stored in the alpha channel of normal maps)
- Calculating proper view vectors in tangent space
- Finding the right balance between quality and performance
- Tuning parameters for the best visual result

By understanding the principles behind parallax mapping, you can enhance the visual quality of your 3D applications with minimal performance impact compared to adding actual geometry.