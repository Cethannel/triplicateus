const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const zlm = @import("zlm");

const builtin = @import("builtin");

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;

const Device = vk.DeviceProxy;

allocator: std.mem.Allocator,
window: ?*glfw.Window = null,
instance: vk.Instance = .null_handle,
debugMessenger: vk.DebugUtilsMessengerEXT = .null_handle,

physical_device: vk.PhysicalDevice = .null_handle,
device: vk.Device = .null_handle,
graphics_queue: vk.Queue = .null_handle,
present_queue: vk.Queue = .null_handle,
surface: vk.SurfaceKHR = .null_handle,

swapchain: vk.SwapchainKHR = .null_handle,
swapchain_images: std.ArrayList(vk.Image) = .empty,
swapchain_image_format: vk.Format = .undefined,
swapchain_extent: vk.Extent2D = .{ .height = undefined, .width = undefined },
swapchain_image_views: std.ArrayList(vk.ImageView) = .empty,
swapchain_framebuffers: std.ArrayList(vk.Framebuffer) = .empty,

render_pass: vk.RenderPass = .null_handle,
pipeline_layout: vk.PipelineLayout = .null_handle,
graphics_pipeline: vk.Pipeline = .null_handle,

command_pool: vk.CommandPool = .null_handle,
command_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer = @splat(.null_handle),

vkb: BaseWrapper = undefined,
vki: InstanceWrapper = undefined,
vkd: DeviceWrapper = undefined,
dev: Device = undefined,

image_available_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = @splat(.null_handle),
render_finished_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = @splat(.null_handle),
in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = @splat(.null_handle),

frame_buffer_resized: bool = false,

current_frame: usize = 0,

const Self = @This();

const WIDTH = 800;
const HEIGHT = 600;
const MAX_FRAMES_IN_FLIGHT = 2;

const validation_layers: []const [*:0]const u8 = ([_][*:0]const u8{"VK_LAYER_KHRONOS_validation"})[0..];
const device_extensions = [_][:0]const u8{vk.extensions.khr_swapchain.name};
const _device_extension_names_arr = blk: {
    var out: [device_extensions.len][*:0]const u8 = undefined;

    for (device_extensions, 0..) |extension, i| {
        out[i] = extension.ptr;
    }

    break :blk out;
};
const device_extension_names: []const [*:0]const u8 = _device_extension_names_arr[0..];

const enable_validation_layers = switch (builtin.mode) {
    .Debug => true,
    .ReleaseFast => false,
    .ReleaseSafe => true,
    .ReleaseSmall => false,
};

pub fn run(self: *Self) !void {
    try self.initWindow();
    try self.initVulkan();
    try self.mainLoop();
    self.cleanup();
}

fn initWindow(self: *Self) !void {
    try glfw.init();
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);

    self.window = try glfw.createWindow(WIDTH, HEIGHT, "Vulkan", null, null);
    glfw.setWindowUserPointer(self.window, self);
    _ = glfw.setFramebufferSizeCallback(self.window, &framebufferResizeCallback);
}

fn framebufferResizeCallback(
    window: *glfw.Window,
    width: c_int,
    height: c_int,
) callconv(.c) void {
    _ = width; // autofix
    _ = height; // autofix
    const self: *Self = @ptrCast(@alignCast(glfw.getWindowUserPointer(window).?));

    self.frame_buffer_resized = true;
}

fn getInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) callconv(.c) ?*const fn () callconv(.c) void {
    return glfw.getInstanceProcAddress(@intFromEnum(instance), procname);
}

fn initVulkan(self: *Self) !void {
    self.vkb = vk.BaseWrapper.load(getInstanceProcAddress);

    try self.createInstance();
    try self.setupDebugMessenger();
    try self.createSurface();
    try self.pickPhysicalDevice();
    try self.createLogicalDevice();
    try self.createSwapChain();
    try self.createImageViews();
    try self.createRenderPass();
    try self.createGraphicsPipeline();
    try self.createFramebuffers();
    try self.createCommandPool();
    try self.createCommandBuffer();
    try self.createSyncObjects();
}

fn createInstance(self: *Self) !void {
    if (enable_validation_layers and !try self.checkValidationLayerSupport()) {
        return error.NoValidationLayers;
    }

    const app_info: vk.ApplicationInfo = .{
        .p_application_name = "Hello Triangle",
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .p_engine_name = "No Engine",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.API_VERSION_1_0),
    };

    var create_info: vk.InstanceCreateInfo = .{
        .p_application_info = &app_info,
    };

    create_info.flags.enumerate_portability_bit_khr = true;

    var required_extensions = try self.getRequiredExtensions();
    defer required_extensions.deinit(self.allocator);
    create_info.enabled_extension_count = @intCast(required_extensions.items.len);
    create_info.pp_enabled_extension_names = required_extensions.items.ptr;

    var debug_create_info: vk.DebugUtilsMessengerCreateInfoEXT = undefined;
    if (enable_validation_layers) {
        create_info.enabled_layer_count = @intCast(validation_layers.len);
        create_info.pp_enabled_layer_names = validation_layers.ptr;

        debug_create_info = populateDebugMessengerCreateInfo();
        create_info.p_next = &debug_create_info;
    } else {
        create_info.enabled_layer_count = 0;

        create_info.p_next = null;
    }

    self.instance = try self.vkb.createInstance(&create_info, null);

    self.vki = .load(self.instance, getInstanceProcAddress);
}

fn checkValidationLayerSupport(self: *Self) !bool {
    var layer_count: u32 = 0;
    _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

    const available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
    _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

    for (validation_layers) |layer_name| {
        const layer_len = std.mem.len(layer_name);
        var padded_name = try std.ArrayList(u8).initCapacity(self.allocator, 256);
        defer padded_name.deinit(self.allocator);
        padded_name.appendSliceAssumeCapacity(layer_name[0..layer_len]);
        padded_name.appendNTimesAssumeCapacity(0, 256 - layer_len);

        var layer_found = false;
        for (available_layers) |layer_props| {
            if (std.mem.eql(u8, padded_name.items, &layer_props.layer_name)) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            return false;
        }
    }

    return true;
}

fn getRequiredExtensions(self: *Self) !std.ArrayList([*:0]const u8) {
    var glfw_extension_count: u32 = 0;
    const glfw_extensions_ptr = glfw.getRequiredInstanceExtensions(&glfw_extension_count) orelse {
        return error.NoGlfwExtensions;
    };
    const glfw_extensions = glfw_extensions_ptr[0..glfw_extension_count];

    var extensions = try std.ArrayList([*:0]const u8).initCapacity(
        self.allocator,
        glfw_extension_count + 1,
    );
    errdefer extensions.deinit(self.allocator);

    extensions.appendSliceAssumeCapacity(glfw_extensions);

    if (enable_validation_layers) {
        extensions.appendAssumeCapacity(vk.extensions.ext_debug_utils.name.ptr);
        try extensions.append(self.allocator, vk.extensions.ext_debug_report.name.ptr);
    }

    return extensions;
}

fn debugCallback(
    messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(.c) vk.Bool32 {
    _ = messageSeverity; // autofix
    _ = message_type; // autofix
    _ = p_user_data; // autofix
    if (p_callback_data) |data| {
        if (data.p_message) |msg| {
            std.log.warn("Validation layer: {s}", .{msg});
        } else {
            std.log.warn("Validation layer: NO MESSAGE", .{});
        }
    }

    return .false;
}

fn setupDebugMessenger(self: *Self) !void {
    if (!enable_validation_layers) {
        return;
    }

    const create_info: vk.DebugUtilsMessengerCreateInfoEXT = populateDebugMessengerCreateInfo();

    std.debug.assert(self.vki.dispatch.vkCreateDebugUtilsMessengerEXT != null);
    self.debugMessenger = try self.vki.createDebugUtilsMessengerEXT(self.instance, &create_info, null);
}

fn populateDebugMessengerCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return vk.DebugUtilsMessengerCreateInfoEXT{
        .message_severity = .{
            .verbose_bit_ext = true,
            .warning_bit_ext = true,
            .error_bit_ext = true,
        },
        .message_type = .{
            .general_bit_ext = true,
            .validation_bit_ext = true,
            .performance_bit_ext = true,
        },
        .pfn_user_callback = &debugCallback,
        .p_user_data = null,
    };
}

fn pickPhysicalDevice(self: *Self) !void {
    var device_count: u32 = 0;

    _ = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);

    if (device_count == 0) {
        std.log.err("Failed to find GPUs with Vulkan support!", .{});
        return error.NoGPU;
    }

    const devices = try self.allocator.alloc(vk.PhysicalDevice, device_count);
    defer self.allocator.free(devices);

    _ = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, devices.ptr);

    var candidates = std.AutoArrayHashMap(i32, vk.PhysicalDevice).init(self.allocator);
    defer candidates.deinit();

    for (devices) |device| {
        const score = self.rateDeviceSuitablility(device);
        try candidates.put(score, device);
    }

    candidates.sort(DeviceSorter{ .hashmap = candidates });

    const first = candidates.keys()[0];
    const first_device = candidates.get(first).?;
    if (try self.isDeviceSuitable(first_device)) {
        self.physical_device = first_device;
    } else {
        std.log.err("Failed to find suitable GPU!", .{});
        return error.NoSuitableGPU;
    }
}

const QueueFamilies = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    fn is_complete(self: *const @This()) bool {
        var out = true;
        inline for (std.meta.fields(@This())) |field| {
            out = out and @field(self, field.name) != null;
        }
        return out;
    }
};

fn findQueueFamilies(self: *const Self, device: vk.PhysicalDevice) !QueueFamilies {
    var indices = QueueFamilies{};

    var queue_family_count: u32 = 0;
    self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

    const queue_families = try self.allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
    defer self.allocator.free(queue_families);
    self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    var i: u32 = 0;

    for (queue_families) |queue_family| {
        if (queue_family.queue_flags.graphics_bit) {
            indices.graphics_family = i;
        }

        if (try self.vki.getPhysicalDeviceSurfaceSupportKHR(device, i, self.surface) == .true) {
            indices.present_family = i;
        }

        if (indices.is_complete()) {
            break;
        }

        i += 1;
    }

    return indices;
}

fn isDeviceSuitable(self: *const Self, device: vk.PhysicalDevice) !bool {
    var indices = try self.findQueueFamilies(device);

    const extensionsSupported = try self.checkDeviceExtensionSupport(device);

    var swapChainAdaquate = false;
    if (extensionsSupported) {
        var swapChainSupport = try self.querySwapChainSupport(device);
        defer swapChainSupport.deinit(self.allocator);
        swapChainAdaquate = (swapChainSupport.formats.items.len != 0) and //
            (swapChainSupport.present_modes.items.len != 0);
    }

    return swapChainAdaquate and extensionsSupported and indices.is_complete();
}

fn checkDeviceExtensionSupport(self: *const Self, device: vk.PhysicalDevice) !bool {
    var extension_count: u32 = 0;

    _ = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, null);

    const available_extensions = try self.allocator.alloc(vk.ExtensionProperties, extension_count);
    defer self.allocator.free(available_extensions);

    _ = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

    var required_extensions = std.StringArrayHashMap(void).init(self.allocator);
    defer required_extensions.deinit();

    for (device_extensions) |extension| {
        const extension_len = std.mem.len(extension.ptr);
        try required_extensions.put(extension[0..extension_len], undefined);
    }

    for (available_extensions) |extension| {
        for (required_extensions.keys()) |key| {
            if (std.mem.eql(u8, key, extension.extension_name[0..key.len])) {
                _ = required_extensions.orderedRemove(key);
                break;
            }
        }
    }

    return required_extensions.keys().len == 0;
}

const DeviceSorter = struct {
    hashmap: std.AutoArrayHashMap(i32, vk.PhysicalDevice),

    pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
        const keys = ctx.hashmap.keys();
        const a_key = keys[a_index];
        const b_key = keys[b_index];

        std.log.info("{d} < {d}", .{ a_key, b_key });

        return a_key < b_key;
    }
};

fn rateDeviceSuitablility(self: *const Self, device: vk.PhysicalDevice) i32 {
    const device_properties = self.vki.getPhysicalDeviceProperties(device);

    const device_features = self.vki.getPhysicalDeviceFeatures(device);

    var score: i32 = 0;

    if (device_properties.device_type == .discrete_gpu) {
        score += 1000;
    }

    score += @intCast(device_properties.limits.max_image_dimension_2d);

    if (device_features.geometry_shader == .false) {
        return 0;
    }

    return score;
}

fn createLogicalDevice(self: *Self) !void {
    const indices = try self.findQueueFamilies(self.physical_device);

    var queue_create_infos = std.AutoArrayHashMap(u32, vk.DeviceQueueCreateInfo).init(self.allocator);

    const queue_priority: f32 = 1.0;
    inline for (std.meta.fields(QueueFamilies)) |field| {
        const queue_family: u32 = @field(indices, field.name).?;
        if (!queue_create_infos.contains(queue_family)) {
            const queue_create_info: vk.DeviceQueueCreateInfo = .{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = @ptrCast(&queue_priority),
            };
            try queue_create_infos.put(queue_family, queue_create_info);
        }
    }

    const device_features: vk.PhysicalDeviceFeatures = .{};
    var create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = queue_create_infos.values().ptr,
        .queue_create_info_count = @intCast(queue_create_infos.values().len),
        .p_enabled_features = &device_features,
        .enabled_extension_count = device_extension_names.len,
        .pp_enabled_extension_names = device_extension_names.ptr,
    };

    if (enable_validation_layers) {
        create_info.enabled_layer_count = @intCast(validation_layers.len);
        create_info.pp_enabled_layer_names = validation_layers.ptr;
    } else {
        create_info.enabled_layer_count = 0;
    }

    self.device = try self.vki.createDevice(self.physical_device, &create_info, null);

    self.vkd = DeviceWrapper.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr.?);
    self.dev = Device.init(self.device, &self.vkd);

    self.graphics_queue = self.vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
    self.present_queue = self.vkd.getDeviceQueue(self.device, indices.present_family.?, 0);
}

fn createSurface(self: *Self) !void {
    if (glfw.createWindowSurface(
        @intFromEnum(self.instance),
        self.window.?,
        null,
        @ptrCast(&self.surface),
    ) != .success) {
        std.log.err("Failed to create window surface", .{});
        return error.NoWindowSurface;
    }
}

const SwapChainSuppoertDetails = struct {
    capabilites: vk.SurfaceCapabilitiesKHR,
    formats: std.ArrayList(vk.SurfaceFormatKHR),
    present_modes: std.ArrayList(vk.PresentModeKHR),

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        self.formats.deinit(allocator);
        self.present_modes.deinit(allocator);
    }
};

fn querySwapChainSupport(
    self: *const Self,
    device: vk.PhysicalDevice,
) !SwapChainSuppoertDetails {
    var details: SwapChainSuppoertDetails = .{
        .capabilites = undefined,
        .formats = .empty,
        .present_modes = .empty,
    };
    errdefer details.formats.deinit(self.allocator);
    errdefer details.present_modes.deinit(self.allocator);

    details.capabilites = try self.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface);

    var format_count: u32 = 0;
    _ = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, null);

    if (format_count != 0) {
        try details.formats.resize(self.allocator, format_count);
        _ = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(
            device,
            self.surface,
            &format_count,
            details.formats.items.ptr,
        );
    }

    var present_mode_count: u32 = 0;
    _ = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, null);

    if (format_count != 0) {
        try details.present_modes.resize(self.allocator, present_mode_count);
        _ = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(
            device,
            self.surface,
            &present_mode_count,
            details.present_modes.items.ptr,
        );
    }

    return details;
}

fn chooseSwapSurfaceFormat(availableFormats: []const vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    for (availableFormats) |available_format| {
        if (available_format.format == .b8g8r8a8_srgb and available_format.color_space == .srgb_nonlinear_khr) {
            return available_format;
        }
    }

    return availableFormats[0];
}

fn chooseSwapPresentMode(
    available_present_modes: []const vk.PresentModeKHR,
) vk.PresentModeKHR {
    for (available_present_modes) |present_mode| {
        if (present_mode == .mailbox_khr) {
            return present_mode;
        }
    }

    return .fifo_khr;
}

fn chooseSwapExtent(self: *const Self, capabilities: *const vk.SurfaceCapabilitiesKHR) vk.Extent2D {
    if (capabilities.current_extent.width != std.math.maxInt(u32)) {
        return capabilities.current_extent;
    } else {
        var width: c_int = 0;
        var height: c_int = 0;

        glfw.getFramebufferSize(self.window, &width, &height);

        var actual_extent: vk.Extent2D = .{
            .width = @intCast(width),
            .height = @intCast(height),
        };

        actual_extent.width = std.math.clamp(
            actual_extent.width,
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );

        actual_extent.height = std.math.clamp(
            actual_extent.height,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );

        return actual_extent;
    }
}

fn createSwapChain(self: *Self) !void {
    var swap_chain_support = try self.querySwapChainSupport(self.physical_device);
    defer swap_chain_support.deinit(self.allocator);

    const surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats.items);
    const present_mode = chooseSwapPresentMode(swap_chain_support.present_modes.items);
    const extent = self.chooseSwapExtent(&swap_chain_support.capabilites);

    var image_count = swap_chain_support.capabilites.min_image_count + 1;
    if (swap_chain_support.capabilites.max_image_count > 0 and //
        image_count > swap_chain_support.capabilites.max_image_count)
    {
        image_count = swap_chain_support.capabilites.max_image_count;
    }

    var create_info: vk.SwapchainCreateInfoKHR = .{
        .surface = self.surface,
        .min_image_count = image_count,
        .image_format = surface_format.format,
        .image_color_space = surface_format.color_space,
        .image_extent = extent,
        .image_array_layers = 1,
        .image_usage = .{
            .color_attachment_bit = true,
        },
        .pre_transform = swap_chain_support.capabilites.current_transform,
        .composite_alpha = .{
            .opaque_bit_khr = true,
        },
        .present_mode = present_mode,
        .clipped = .true,
        .image_sharing_mode = undefined,
        .old_swapchain = .null_handle,
    };

    const indices = try self.findQueueFamilies(self.physical_device);
    var queue_family_indices = [_]u32{ indices.graphics_family.?, indices.present_family.? };

    if (indices.graphics_family != indices.present_family) {
        create_info.image_sharing_mode = .concurrent;
        create_info.queue_family_index_count = 2;
        create_info.p_queue_family_indices = queue_family_indices[0..].ptr;
    } else {
        create_info.image_sharing_mode = .exclusive;
        create_info.queue_family_index_count = 0;
        create_info.p_queue_family_indices = null;
    }

    self.swapchain = try self.dev.createSwapchainKHR(&create_info, null);

    self.swapchain_images = .fromOwnedSlice(try self.dev.getSwapchainImagesAllocKHR(self.swapchain, self.allocator));
    errdefer self.swapchain_images.deinit(self.allocator);

    self.swapchain_image_format = surface_format.format;
    self.swapchain_extent = extent;
}

fn createImageViews(self: *Self) !void {
    try self.swapchain_image_views.resize(self.allocator, self.swapchain_images.items.len);
    errdefer self.swapchain_image_views.deinit(self.allocator);

    for (self.swapchain_images.items, 0..) |image, i| {
        const create_info: vk.ImageViewCreateInfo = .{
            .image = image,
            .view_type = .@"2d",
            .format = self.swapchain_image_format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };

        self.swapchain_image_views.items[i] = try self.dev.createImageView(&create_info, null);
    }
}

fn createGraphicsPipeline(self: *Self) !void {
    const vertex_shader align(4) = @embedFile("vertex_shader").*;
    const fragment_shader align(4) = @embedFile("fragment_shader").*;

    const vert_shader_module = try self.createShaderModule(&vertex_shader);
    defer self.dev.destroyShaderModule(vert_shader_module, null);
    const frag_shader_module = try self.createShaderModule(&fragment_shader);
    defer self.dev.destroyShaderModule(frag_shader_module, null);

    const vert_shader_stage_info: vk.PipelineShaderStageCreateInfo = .{
        .stage = .{ .vertex_bit = true },
        .module = vert_shader_module,
        .p_name = "main",
    };

    const frag_shader_stage_info: vk.PipelineShaderStageCreateInfo = .{
        .stage = .{ .fragment_bit = true },
        .module = frag_shader_module,
        .p_name = "main",
    };

    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        vert_shader_stage_info,
        frag_shader_stage_info,
    };

    const dynamic_states = [_]vk.DynamicState{
        .viewport,
        .scissor,
    };

    const dynamic_state: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = @intCast(dynamic_states.len),
        .p_dynamic_states = dynamic_states[0..].ptr,
    };

    const vertex_input_info: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = 0,
        .p_vertex_binding_descriptions = null,
        .vertex_attribute_description_count = 0,
        .p_vertex_attribute_descriptions = null,
    };

    const input_assembly: vk.PipelineInputAssemblyStateCreateInfo = .{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };

    const viewport: vk.Viewport = .{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(self.swapchain_extent.width),
        .height = @floatFromInt(self.swapchain_extent.height),
        .min_depth = 0.0,
        .max_depth = 1.0,
    };

    const scisor: vk.Rect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.swapchain_extent,
    };

    const viewport_state: vk.PipelineViewportStateCreateInfo = .{
        .viewport_count = 1,
        .scissor_count = 1,
        .p_viewports = @ptrCast(&viewport),
        .p_scissors = @ptrCast(&scisor),
    };

    const rasterizer: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_bias_enable = .false,
        .depth_clamp_enable = .false,
        .depth_bias_constant_factor = 0.0,
        .depth_bias_clamp = 0.0,
        .depth_bias_slope_factor = 0.0,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .line_width = 1.0,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
    };

    const multisampling: vk.PipelineMultisampleStateCreateInfo = .{
        .sample_shading_enable = .true,
        .rasterization_samples = .{ .@"1_bit" = true },
        .min_sample_shading = 1.0,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };

    const color_blend_attachment: vk.PipelineColorBlendAttachmentState = .{
        .color_write_mask = .{
            .r_bit = true,
            .g_bit = true,
            .b_bit = true,
            .a_bit = true,
        },
        .blend_enable = .false,
        .src_color_blend_factor = .src_alpha,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
    };

    const color_blending: vk.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_blend_attachment),
        .blend_constants = @splat(0.0),
    };

    const pipeline_layout_info: vk.PipelineLayoutCreateInfo = .{};

    self.pipeline_layout = try self.dev.createPipelineLayout(&pipeline_layout_info, null);

    const pipeline_info: vk.GraphicsPipelineCreateInfo = .{
        .stage_count = 2,
        .p_stages = shader_stages[0..].ptr,
        .p_vertex_input_state = &vertex_input_info,
        .p_input_assembly_state = &input_assembly,
        .p_rasterization_state = &rasterizer,
        .p_multisample_state = &multisampling,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &color_blending,
        .p_dynamic_state = &dynamic_state,
        .p_viewport_state = &viewport_state,
        .layout = self.pipeline_layout,
        .render_pass = self.render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    _ = try self.dev.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&pipeline_info),
        null,
        @ptrCast(&self.graphics_pipeline),
    );
}

fn createRenderPass(self: *Self) !void {
    const color_attachment: vk.AttachmentDescription = .{
        .format = self.swapchain_image_format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref: vk.AttachmentReference = .{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    const dependency: vk.SubpassDependency = .{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    };

    const render_pass_info: vk.RenderPassCreateInfo = .{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    };

    self.render_pass = try self.dev.createRenderPass(&render_pass_info, null);
}

fn createShaderModule(self: *const Self, code: []align(4) const u8) !vk.ShaderModule {
    const create_info: vk.ShaderModuleCreateInfo = .{
        .code_size = code.len,
        .p_code = @ptrCast(code.ptr),
    };

    return self.dev.createShaderModule(&create_info, null);
}

fn createFramebuffers(self: *Self) !void {
    try self.swapchain_framebuffers.resize(
        self.allocator,
        self.swapchain_image_views.items.len,
    );
    errdefer self.swapchain_framebuffers.deinit(self.allocator);

    for (self.swapchain_image_views.items, 0..) |image_view, i| {
        const attackments = [_]vk.ImageView{
            image_view,
        };

        const frame_buffer_info: vk.FramebufferCreateInfo = .{
            .render_pass = self.render_pass,
            .attachment_count = 1,
            .p_attachments = attackments[0..].ptr,
            .width = self.swapchain_extent.width,
            .height = self.swapchain_extent.height,
            .layers = 1,
        };

        self.swapchain_framebuffers.items[i] = try self.dev.createFramebuffer(
            &frame_buffer_info,
            null,
        );
    }
}

fn createCommandPool(self: *Self) !void {
    const queue_family_indices = try self.findQueueFamilies(self.physical_device);
    const pool_info: vk.CommandPoolCreateInfo = .{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queue_family_indices.graphics_family.?,
    };

    self.command_pool = try self.dev.createCommandPool(&pool_info, null);
}

fn createCommandBuffer(self: *Self) !void {
    const alloc_info: vk.CommandBufferAllocateInfo = .{
        .command_pool = self.command_pool,
        .level = .primary,
        .command_buffer_count = self.command_buffers.len,
    };

    _ = try self.dev.allocateCommandBuffers(&alloc_info, self.command_buffers[0..].ptr);
}

fn recordCommandBuffer(
    self: *Self,
    command_buffer: vk.CommandBuffer,
    image_index: u32,
) !void {
    const begin_info: vk.CommandBufferBeginInfo = .{};

    _ = try self.dev.beginCommandBuffer(command_buffer, &begin_info);

    const clear_color: vk.ClearValue = .{
        .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 1.0 } },
    };

    const render_pass_info: vk.RenderPassBeginInfo = .{
        .render_pass = self.render_pass,
        .framebuffer = self.swapchain_framebuffers.items[image_index],
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        },
        .clear_value_count = 1,
        .p_clear_values = @ptrCast(&clear_color),
    };

    self.vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");

    self.vkd.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);

    const viewport: vk.Viewport = .{
        .x = 0.0,
        .y = 0.0,
        .width = @floatFromInt(self.swapchain_extent.width),
        .height = @floatFromInt(self.swapchain_extent.height),
        .min_depth = 0.0,
        .max_depth = 1.0,
    };

    self.vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&viewport));

    const scissor: vk.Rect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.swapchain_extent,
    };

    self.vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&scissor));

    self.vkd.cmdDraw(command_buffer, 3, 1, 0, 0);

    self.vkd.cmdEndRenderPass(command_buffer);

    try self.vkd.endCommandBuffer(command_buffer);
}

fn createSyncObjects(self: *Self) !void {
    const semaphore_info: vk.SemaphoreCreateInfo = .{};
    const fence_info: vk.FenceCreateInfo = .{
        .flags = .{ .signaled_bit = true },
    };

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        self.image_available_semaphores[i] = try self.dev.createSemaphore(&semaphore_info, null);
        self.render_finished_semaphores[i] = try self.dev.createSemaphore(&semaphore_info, null);
        self.in_flight_fences[i] = try self.dev.createFence(&fence_info, null);
    }
}

fn mainLoop(self: *Self) !void {
    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
        try self.drawFrame();
    }
    try self.dev.deviceWaitIdle();
}

fn drawFrame(self: *Self) !void {
    _ = try self.dev.waitForFences(1, self.in_flight_fences[self.current_frame..].ptr, .true, std.math.maxInt(u64));

    const next_image_result = (try self.dev.acquireNextImageKHR(
        self.swapchain,
        std.math.maxInt(u64),
        self.image_available_semaphores[self.current_frame],
        self.in_flight_fences[self.current_frame],
    ));

    if (next_image_result.result == .error_out_of_date_khr) {
        try self.recreateSwapChain();
        return;
    }

    try self.dev.resetFences(1, self.in_flight_fences[self.current_frame..].ptr);

    const image_index = next_image_result.image_index;

    try self.vkd.resetCommandBuffer(self.command_buffers[self.current_frame], .{});
    try self.recordCommandBuffer(self.command_buffers[self.current_frame], image_index);

    const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores[self.current_frame]};
    const wait_stages = [_]vk.PipelineStageFlags{
        .{ .color_attachment_output_bit = true },
    };

    const singal_sempahores = [_]vk.Semaphore{self.render_finished_semaphores[self.current_frame]};

    const submit_info: vk.SubmitInfo = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = wait_semaphores[0..].ptr,
        .p_wait_dst_stage_mask = wait_stages[0..].ptr,
        .command_buffer_count = 1,
        .p_command_buffers = self.command_buffers[self.current_frame..].ptr,
        .signal_semaphore_count = 1,
        .p_signal_semaphores = singal_sempahores[0..].ptr,
    };

    try self.vkd.queueSubmit(
        self.graphics_queue,
        1,
        @ptrCast(&submit_info),
        self.in_flight_fences[self.current_frame],
    );

    const swapchains = [_]vk.SwapchainKHR{self.swapchain};

    const present_info: vk.PresentInfoKHR = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = singal_sempahores[0..].ptr,
        .swapchain_count = 1,
        .p_swapchains = swapchains[0..].ptr,
        .p_image_indices = @ptrCast(&image_index),
    };

    const queue_result = try self.vkd.queuePresentKHR(self.present_queue, &present_info);

    switch (queue_result) {
        .error_out_of_date_khr, .suboptimal_khr => {
            self.frame_buffer_resized = false;
            try self.recreateSwapChain();
        },
        else => {
            if (self.frame_buffer_resized) {
                self.frame_buffer_resized = false;
                try self.recreateSwapChain();
            }
        },
    }

    self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

fn recreateSwapChain(self: *Self) !void {
    var width: c_int = 0;
    var height: c_int = 0;

    glfw.getFramebufferSize(self.window, &width, &height);

    while (width == 0 or height == 0) {
        glfw.getFramebufferSize(self.window, &width, &height);
        glfw.waitEvents();
    }

    try self.dev.deviceWaitIdle();

    self.creanupSwapChain();

    try self.createSwapChain();
    try self.createImageViews();
    try self.createFramebuffers();
}

fn creanupSwapChain(self: *Self) void {
    for (self.swapchain_framebuffers.items) |framebuffer| {
        self.dev.destroyFramebuffer(framebuffer, null);
    }

    for (self.swapchain_image_views.items) |view| {
        self.dev.destroyImageView(view, null);
    }

    self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);
}

fn cleanup(self: *Self) void {
    self.creanupSwapChain();

    self.dev.destroyPipeline(self.graphics_pipeline, null);
    self.dev.destroyPipelineLayout(self.pipeline_layout, null);

    self.dev.destroyRenderPass(self.render_pass, null);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        self.dev.destroySemaphore(self.image_available_semaphores[i], null);
        self.dev.destroySemaphore(self.render_finished_semaphores[i], null);
        self.dev.destroyFence(self.in_flight_fences[i], null);
    }

    self.dev.destroyCommandPool(self.command_pool, null);

    defer self.swapchain_framebuffers.deinit(self.allocator);

    defer self.swapchain_image_views.deinit(self.allocator);
    defer self.swapchain_images.deinit(self.allocator);

    self.vkd.destroyDevice(self.device, null);

    if (enable_validation_layers) {
        self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debugMessenger, null);
    }

    self.vki.destroySurfaceKHR(self.instance, self.surface, null);
    self.vki.destroyInstance(self.instance, null);

    glfw.destroyWindow(self.window);

    glfw.terminate();
}
