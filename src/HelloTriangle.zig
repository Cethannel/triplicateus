const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const zlm = @import("zlm");
const zignal = @import("zignal");

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
descriptor_set_layout: vk.DescriptorSetLayout = .null_handle,
pipeline_layout: vk.PipelineLayout = .null_handle,
graphics_pipeline: vk.Pipeline = .null_handle,

command_pool: vk.CommandPool = .null_handle,
descriptor_pool: vk.DescriptorPool = .null_handle,
descriptor_sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet = @splat(.null_handle),
command_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer = @splat(.null_handle),

vertex_buffer: vk.Buffer = .null_handle,
vertex_buffer_memory: vk.DeviceMemory = .null_handle,
index_buffer: vk.Buffer = .null_handle,
index_buffer_memory: vk.DeviceMemory = .null_handle,

uniform_buffers: [MAX_FRAMES_IN_FLIGHT]vk.Buffer = @splat(.null_handle),
uniform_buffers_memory: [MAX_FRAMES_IN_FLIGHT]vk.DeviceMemory = @splat(.null_handle),
uniform_buffers_mapped: [MAX_FRAMES_IN_FLIGHT]?*anyopaque = @splat(null),

texture_image: vk.Image = .null_handle,
texture_image_memory: vk.DeviceMemory = .null_handle,
texture_image_view: vk.ImageView = .null_handle,
texture_image_sampler: vk.Sampler = .null_handle,

depth_image: vk.Image = .null_handle,
depth_image_memory: vk.DeviceMemory = .null_handle,
depth_image_view: vk.ImageView = .null_handle,

vkb: BaseWrapper = undefined,
vki: InstanceWrapper = undefined,
vkd: DeviceWrapper = undefined,
dev: Device = undefined,

image_available_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = @splat(.null_handle),
render_finished_semaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = @splat(.null_handle),
in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = @splat(.null_handle),

frame_buffer_resized: bool = false,

current_frame: usize = 0,

start_time: std.time.Instant = undefined,

const Self = @This();

const Vertex = extern struct {
    pos: zlm.Vec3,
    color: zlm.Vec3,
    tex_coord: zlm.Vec2,

    pub fn getBindingDescription() vk.VertexInputBindingDescription {
        return vk.VertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(@This()),
            .input_rate = .vertex,
        };
    }

    pub fn getAttributeDescriptions() [3]vk.VertexInputAttributeDescription {
        return [3]vk.VertexInputAttributeDescription{
            vk.VertexInputAttributeDescription{
                .binding = 0,
                .location = 0,
                .format = .r32g32b32_sfloat,
                .offset = @offsetOf(@This(), "pos"),
            },
            vk.VertexInputAttributeDescription{
                .binding = 0,
                .location = 1,
                .format = .r32g32b32_sfloat,
                .offset = @offsetOf(@This(), "color"),
            },
            vk.VertexInputAttributeDescription{
                .binding = 0,
                .location = 2,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(@This(), "tex_coord"),
            },
        };
    }
};

const vertices = [_]Vertex{
    .{ .pos = .new(-0.5, -0.5, 0.0), .color = .new(1.0, 0.0, 0.0), .tex_coord = .new(1.0, 0.0) },
    .{ .pos = .new(0.5, -0.5, 0.0), .color = .new(0.0, 1.0, 0.0), .tex_coord = .new(0.0, 0.0) },
    .{ .pos = .new(0.5, 0.5, 0.0), .color = .new(0.0, 0.0, 1.0), .tex_coord = .new(0.0, 1.0) },
    .{ .pos = .new(-0.5, 0.5, 0.0), .color = .new(1.0, 1.0, 1.0), .tex_coord = .new(1.0, 1.0) },

    .{ .pos = .new(-0.5, -0.5, -0.5), .color = .new(1.0, 0.0, 0.0), .tex_coord = .new(1.0, 0.0) },
    .{ .pos = .new(0.5, -0.5, -0.5), .color = .new(0.0, 1.0, 0.0), .tex_coord = .new(0.0, 0.0) },
    .{ .pos = .new(0.5, 0.5, -0.5), .color = .new(0.0, 0.0, 1.0), .tex_coord = .new(0.0, 1.0) },
    .{ .pos = .new(-0.5, 0.5, -0.5), .color = .new(1.0, 1.0, 1.0), .tex_coord = .new(1.0, 1.0) },
};

const indices = [_]u16{
    0, 1, 2, 2, 3, 0, //
    4, 5, 6, 6, 7, 4,
};

const UniformBufferObject = extern struct {
    model: zlm.Mat4 align(16),
    view: zlm.Mat4 align(16),
    proj: zlm.Mat4 align(16),
};

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
    self.start_time = try std.time.Instant.now();
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
    try self.createDescriptorSetLayout();
    try self.createGraphicsPipeline();
    try self.createCommandPool();
    try self.createDepthResources();
    try self.createFramebuffers();
    try self.createTextureImage();
    try self.createTextureImageView();
    try self.createTextureSampler();
    try self.createVertexBuffer();
    try self.createIndexBuffer();
    try self.createUniformBuffers();
    try self.createDescriptorPool();
    try self.createDescriptorSets();
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
    var queue_indices = QueueFamilies{};

    var queue_family_count: u32 = 0;
    self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

    const queue_families = try self.allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
    defer self.allocator.free(queue_families);
    self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    var i: u32 = 0;

    for (queue_families) |queue_family| {
        if (queue_family.queue_flags.graphics_bit) {
            queue_indices.graphics_family = i;
        }

        if (try self.vki.getPhysicalDeviceSurfaceSupportKHR(device, i, self.surface) == .true) {
            queue_indices.present_family = i;
        }

        if (queue_indices.is_complete()) {
            break;
        }

        i += 1;
    }

    return queue_indices;
}

fn isDeviceSuitable(self: *const Self, device: vk.PhysicalDevice) !bool {
    var queue_indices = try self.findQueueFamilies(device);

    const extensionsSupported = try self.checkDeviceExtensionSupport(device);

    var swapChainAdaquate = false;
    if (extensionsSupported) {
        var swapChainSupport = try self.querySwapChainSupport(device);
        defer swapChainSupport.deinit(self.allocator);
        swapChainAdaquate = (swapChainSupport.formats.items.len != 0) and //
            (swapChainSupport.present_modes.items.len != 0);
    }

    const supported_features = self.vki.getPhysicalDeviceFeatures(device);

    return swapChainAdaquate //
    and extensionsSupported //
    and queue_indices.is_complete() //
    and supported_features.sampler_anisotropy == .true;
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
    const queue_indices = try self.findQueueFamilies(self.physical_device);

    var queue_create_infos = std.AutoArrayHashMap(u32, vk.DeviceQueueCreateInfo).init(self.allocator);

    const queue_priority: f32 = 1.0;
    inline for (std.meta.fields(QueueFamilies)) |field| {
        const queue_family: u32 = @field(queue_indices, field.name).?;
        if (!queue_create_infos.contains(queue_family)) {
            const queue_create_info: vk.DeviceQueueCreateInfo = .{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = @ptrCast(&queue_priority),
            };
            try queue_create_infos.put(queue_family, queue_create_info);
        }
    }

    const device_features: vk.PhysicalDeviceFeatures = .{
        .sampler_anisotropy = .true,
    };

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

    self.graphics_queue = self.vkd.getDeviceQueue(self.device, queue_indices.graphics_family.?, 0);
    self.present_queue = self.vkd.getDeviceQueue(self.device, queue_indices.present_family.?, 0);
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

    const queue_indices = try self.findQueueFamilies(self.physical_device);
    var queue_family_indices = [_]u32{ queue_indices.graphics_family.?, queue_indices.present_family.? };

    if (queue_indices.graphics_family != queue_indices.present_family) {
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
        self.swapchain_image_views.items[i] = try self.createImageView(
            image,
            self.swapchain_image_format,
            .{ .color_bit = true },
        );
    }
}

fn createVertexBuffer(self: *Self) !void {
    const buffer_size: vk.DeviceSize = @sizeOf(Vertex) * vertices.len;

    var staging_buffer: vk.Buffer = .null_handle;
    var staging_buffer_memory: vk.DeviceMemory = .null_handle;

    try self.createBuffer(
        buffer_size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
        &staging_buffer,
        &staging_buffer_memory,
    );

    const data = try self.dev.mapMemory(staging_buffer_memory, 0, buffer_size, .{});
    const data_arr: *@TypeOf(vertices) = @ptrCast(@alignCast(data.?));
    @memcpy(data_arr, vertices[0..]);
    self.dev.unmapMemory(staging_buffer_memory);

    try self.createBuffer(
        buffer_size,
        .{
            .vertex_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        .{ .device_local_bit = true },
        &self.vertex_buffer,
        &self.vertex_buffer_memory,
    );

    try self.copyBuffer(staging_buffer, self.vertex_buffer, buffer_size);

    self.dev.destroyBuffer(staging_buffer, null);
    self.dev.freeMemory(staging_buffer_memory, null);
}

fn transitionImageLayout(
    self: *Self,
    image: vk.Image,
    format: vk.Format,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
) !void {
    const command_buffer = try self.beginSingleTimeCommands();

    var barrier: vk.ImageMemoryBarrier = .{
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .src_access_mask = undefined,
        .dst_access_mask = undefined,
    };

    if (new_layout == .depth_stencil_attachment_optimal) {
        barrier.subresource_range.aspect_mask = .{ .depth_bit = true };

        if (hasStencilComponent(format)) {
            barrier.subresource_range.aspect_mask.stencil_bit = true;
        }
    } else {
        barrier.subresource_range.aspect_mask = .{ .color_bit = true };
    }

    var source_stage: vk.PipelineStageFlags = .{};
    var destination_stage: vk.PipelineStageFlags = .{};

    if (old_layout == .undefined and new_layout == .transfer_dst_optimal) {
        barrier.src_access_mask = .{};
        barrier.dst_access_mask = .{ .transfer_write_bit = true };

        source_stage = .{ .top_of_pipe_bit = true };
        destination_stage = .{ .transfer_bit = true };
    } else if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) {
        barrier.src_access_mask = .{ .transfer_write_bit = true };
        barrier.dst_access_mask = .{ .shader_read_bit = true };

        source_stage = .{ .transfer_bit = true };
        destination_stage = .{ .fragment_shader_bit = true };
    } else if (old_layout == .undefined and new_layout == .depth_stencil_attachment_optimal) {
        barrier.src_access_mask = .{};
        barrier.dst_access_mask = .{
            .depth_stencil_attachment_read_bit = true,
            .depth_stencil_attachment_write_bit = true,
        };

        source_stage = .{ .top_of_pipe_bit = true };
        destination_stage = .{ .early_fragment_tests_bit = true };
    } else {
        unreachable;
    }

    self.vkd.cmdPipelineBarrier(
        command_buffer,
        source_stage,
        destination_stage,
        .{},
        0,
        null,
        0,
        null,
        1,
        @ptrCast(&barrier),
    );

    try self.endSingleTimeCommands(command_buffer);
}

fn copyBufferToImage(
    self: *Self,
    buffer: vk.Buffer,
    image: vk.Image,
    width: u32,
    height: u32,
) !void {
    const command_buffer = try self.beginSingleTimeCommands();

    const region: vk.BufferImageCopy = .{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,

        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },

        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = width, .height = height, .depth = 1 },
    };

    self.vkd.cmdCopyBufferToImage(
        command_buffer,
        buffer,
        image,
        .transfer_dst_optimal,
        1,
        @ptrCast(&region),
    );

    try self.endSingleTimeCommands(command_buffer);
}

fn createIndexBuffer(self: *Self) !void {
    const buffer_size: vk.DeviceSize = @sizeOf(u16) * indices.len;

    var staging_buffer: vk.Buffer = .null_handle;
    var staging_buffer_memory: vk.DeviceMemory = .null_handle;

    try self.createBuffer(
        buffer_size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
        &staging_buffer,
        &staging_buffer_memory,
    );

    const data = try self.dev.mapMemory(staging_buffer_memory, 0, buffer_size, .{});
    const data_arr: *@TypeOf(indices) = @ptrCast(@alignCast(data.?));
    @memcpy(data_arr, indices[0..]);
    self.dev.unmapMemory(staging_buffer_memory);

    try self.createBuffer(
        buffer_size,
        .{
            .index_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        .{ .device_local_bit = true },
        &self.index_buffer,
        &self.index_buffer_memory,
    );

    try self.copyBuffer(staging_buffer, self.index_buffer, buffer_size);

    self.dev.destroyBuffer(staging_buffer, null);
    self.dev.freeMemory(staging_buffer_memory, null);
}

fn createUniformBuffers(self: *Self) !void {
    const buffer_size = @sizeOf(UniformBufferObject);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        try self.createBuffer(
            buffer_size,
            .{
                .uniform_buffer_bit = true,
            },
            .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            },
            &self.uniform_buffers[i],
            &self.uniform_buffers_memory[i],
        );

        self.uniform_buffers_mapped[i] = try self.dev.mapMemory(
            self.uniform_buffers_memory[i],
            0,
            buffer_size,
            .{},
        );
    }
}

fn createDescriptorPool(self: *Self) !void {
    const pool_sizes = [2]vk.DescriptorPoolSize{
        .{
            .type = .uniform_buffer,
            .descriptor_count = MAX_FRAMES_IN_FLIGHT,
        },
        .{
            .type = .combined_image_sampler,
            .descriptor_count = MAX_FRAMES_IN_FLIGHT,
        },
    };

    const pool_info: vk.DescriptorPoolCreateInfo = .{
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = pool_sizes[0..].ptr,
        .max_sets = MAX_FRAMES_IN_FLIGHT,
    };

    self.descriptor_pool = try self.dev.createDescriptorPool(&pool_info, null);
}

fn createDescriptorSets(self: *Self) !void {
    var layouts: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout = @splat(self.descriptor_set_layout);
    const alloc_info: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = MAX_FRAMES_IN_FLIGHT,
        .p_set_layouts = layouts[0..].ptr,
    };

    try self.dev.allocateDescriptorSets(&alloc_info, self.descriptor_sets[0..].ptr);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        const buffer_info: vk.DescriptorBufferInfo = .{
            .buffer = self.uniform_buffers[i],
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        const image_info: vk.DescriptorImageInfo = .{
            .image_layout = .shader_read_only_optimal,
            .image_view = self.texture_image_view,
            .sampler = self.texture_image_sampler,
        };

        const descriptor_writes = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.descriptor_sets[i],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .p_buffer_info = @ptrCast(&buffer_info),
                .p_image_info = ([_]vk.DescriptorImageInfo{})[0..].ptr,
                .p_texel_buffer_view = ([_]vk.BufferView{})[0..].ptr,
            },
            .{
                .dst_set = self.descriptor_sets[i],
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
                .p_buffer_info = ([_]vk.DescriptorBufferInfo{})[0..].ptr,
                .p_image_info = @ptrCast(&image_info),
                .p_texel_buffer_view = ([_]vk.BufferView{})[0..].ptr,
            },
        };

        self.dev.updateDescriptorSets(
            @intCast(descriptor_writes.len),
            descriptor_writes[0..].ptr,
            0,
            null,
        );
    }
}

fn copyBuffer(
    self: *Self,
    src_buffer: vk.Buffer,
    dst_buffer: vk.Buffer,
    size: vk.DeviceSize,
) !void {
    const command_buffer = try self.beginSingleTimeCommands();

    const copy_region: vk.BufferCopy = .{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };

    self.vkd.cmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, @ptrCast(&copy_region));

    try self.endSingleTimeCommands(command_buffer);
}

fn createBuffer(
    self: *Self,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    properties: vk.MemoryPropertyFlags,
    buffer: *vk.Buffer,
    buffer_memory: *vk.DeviceMemory,
) !void {
    const buffer_info: vk.BufferCreateInfo = .{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    };

    buffer.* = try self.dev.createBuffer(&buffer_info, null);

    const mem_requirements = self.dev.getBufferMemoryRequirements(buffer.*);

    const alloc_info: vk.MemoryAllocateInfo = .{
        .allocation_size = mem_requirements.size,
        .memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, properties),
    };

    buffer_memory.* = try self.dev.allocateMemory(&alloc_info, null);

    try self.dev.bindBufferMemory(buffer.*, buffer_memory.*, 0);
}

fn findMemoryType(self: *Self, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
    const mem_properties = self.vki.getPhysicalDeviceMemoryProperties(self.physical_device);

    for (0..mem_properties.memory_type_count) |i| {
        if (type_filter & (@as(u32, 1) << @intCast(i)) != 0 and //
            @as(u32, @bitCast(mem_properties.memory_types[i].property_flags)) //
            & @as(u32, @bitCast(properties)) == @as(u32, @bitCast(properties)))
        {
            return @intCast(i);
        }
    }

    std.log.err("Failed to find suitable memory type", .{});
    return error.NoSuitableMemoryType;
}

fn createDescriptorSetLayout(self: *Self) !void {
    const ubo_layout_binding: vk.DescriptorSetLayoutBinding = .{
        .binding = 0,
        .descriptor_type = .uniform_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex_bit = true },
        .p_immutable_samplers = null,
    };

    const sampler_layout_binding: vk.DescriptorSetLayoutBinding = .{
        .binding = 1,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_immutable_samplers = null,
        .stage_flags = .{ .fragment_bit = true },
    };

    const bindings = [_]vk.DescriptorSetLayoutBinding{ ubo_layout_binding, sampler_layout_binding };

    const layout_info: vk.DescriptorSetLayoutCreateInfo = .{
        .binding_count = @intCast(bindings.len),
        .p_bindings = bindings[0..].ptr,
    };

    self.descriptor_set_layout = try self.dev.createDescriptorSetLayout(
        &layout_info,
        null,
    );
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

    const binding_description = Vertex.getBindingDescription();
    const attribute_description = Vertex.getAttributeDescriptions();

    const vertex_input_info: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&binding_description),
        .vertex_attribute_description_count = attribute_description.len,
        .p_vertex_attribute_descriptions = attribute_description[0..].ptr,
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
        .front_face = .counter_clockwise,
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

    const pipeline_layout_info: vk.PipelineLayoutCreateInfo = .{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&self.descriptor_set_layout),
    };

    self.pipeline_layout = try self.dev.createPipelineLayout(
        &pipeline_layout_info,
        null,
    );

    const depth_stencil: vk.PipelineDepthStencilStateCreateInfo = .{
        .depth_test_enable = .true,
        .depth_write_enable = .true,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = .false,
        .min_depth_bounds = 0.0,
        .max_depth_bounds = 1.0,
        .stencil_test_enable = .true,
        .front = undefined,
        .back = undefined,
    };

    const pipeline_info: vk.GraphicsPipelineCreateInfo = .{
        .stage_count = 2,
        .p_stages = shader_stages[0..].ptr,
        .p_vertex_input_state = &vertex_input_info,
        .p_input_assembly_state = &input_assembly,
        .p_rasterization_state = &rasterizer,
        .p_multisample_state = &multisampling,
        .p_depth_stencil_state = &depth_stencil,
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

    const depth_attachment: vk.AttachmentDescription = .{
        .format = self.findDepthFormat(),
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .dont_care,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };

    const depth_attachment_ref: vk.AttachmentReference = .{
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_depth_stencil_attachment = &depth_attachment_ref,
    };

    const dependency: vk.SubpassDependency = .{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{
            .color_attachment_output_bit = true,
            .late_fragment_tests_bit = true,
        },
        .src_access_mask = .{
            .depth_stencil_attachment_write_bit = true,
        },
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
        },
        .dst_access_mask = .{
            .color_attachment_write_bit = true,
            .depth_stencil_attachment_write_bit = true,
        },
    };

    const attachments = [_]vk.AttachmentDescription{
        color_attachment,
        depth_attachment,
    };

    const render_pass_info: vk.RenderPassCreateInfo = .{
        .attachment_count = attachments.len,
        .p_attachments = attachments[0..].ptr,
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
        const attachments = [_]vk.ImageView{
            image_view,
            self.depth_image_view,
        };

        const frame_buffer_info: vk.FramebufferCreateInfo = .{
            .render_pass = self.render_pass,
            .attachment_count = attachments.len,
            .p_attachments = attachments[0..].ptr,
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

fn createDepthResources(self: *Self) !void {
    const depth_format = self.findDepthFormat();

    try self.createImage(
        self.swapchain_extent.width,
        self.swapchain_extent.height,
        depth_format,
        .optimal,
        .{ .depth_stencil_attachment_bit = true },
        .{ .device_local_bit = true },
        &self.depth_image,
        &self.depth_image_memory,
    );

    self.depth_image_view = try self.createImageView(
        self.depth_image,
        depth_format,
        .{ .depth_bit = true },
    );

    try self.transitionImageLayout(
        self.depth_image,
        depth_format,
        .undefined,
        .depth_stencil_attachment_optimal,
    );
}

fn findSupportedFormat(
    self: *Self,
    candidates: []const vk.Format,
    tiling: vk.ImageTiling,
    featrues: vk.FormatFeatureFlags,
) vk.Format {
    for (candidates) |format| {
        const props = self.vki.getPhysicalDeviceFormatProperties(self.physical_device, format);

        if (tiling == .linear and atLeast(props.linear_tiling_features, featrues)) {
            return format;
        } else if (tiling == .optimal and atLeast(props.optimal_tiling_features, featrues)) {
            return format;
        }
    }

    unreachable;
}

fn hasStencilComponent(format: vk.Format) bool {
    return format == .d32_sfloat_s8_uint or format == .d24_unorm_s8_uint;
}

fn findDepthFormat(self: *Self) vk.Format {
    const candidates = [_]vk.Format{
        .d32_sfloat,
        .d32_sfloat_s8_uint,
        .d24_unorm_s8_uint,
    };

    return self.findSupportedFormat(
        candidates[0..],
        .optimal,
        .{ .depth_stencil_attachment_bit = true },
    );
}

fn atLeast(input: anytype, at_least: @TypeOf(input)) bool {
    const T = @TypeOf(input);
    const t_info = @typeInfo(T);
    const struct_info = t_info.@"struct";

    const T_int = struct_info.backing_integer.?;

    return (@as(T_int, @bitCast(input)) & @as(T_int, @bitCast(at_least))) //
    == @as(T_int, @bitCast(at_least));
}

fn createTextureImage(self: *Self) !void {
    var image = try zignal.jpeg.load(zignal.Rgba, self.allocator, "textures/texture.jpg");
    defer image.deinit(self.allocator);

    const tex_width = image.cols;
    const tex_height = image.rows;

    const image_size = tex_width * tex_height * 4;

    var staging_buffer: vk.Buffer = .null_handle;
    var staging_buffer_memory: vk.DeviceMemory = .null_handle;

    try self.createBuffer(
        image_size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
        &staging_buffer,
        &staging_buffer_memory,
    );

    try self.copyData(u8, image.asBytes(), staging_buffer_memory, image_size);

    try self.createImage(
        @intCast(tex_width),
        @intCast(tex_height),
        .r8g8b8a8_srgb,
        .optimal,
        .{ .transfer_dst_bit = true, .sampled_bit = true },
        .{ .device_local_bit = true },
        &self.texture_image,
        &self.texture_image_memory,
    );

    try self.transitionImageLayout(
        self.texture_image,
        .r8g8b8a8_srgb,
        .undefined,
        .transfer_dst_optimal,
    );

    try self.copyBufferToImage(
        staging_buffer,
        self.texture_image,
        @intCast(tex_width),
        @intCast(tex_height),
    );

    try self.transitionImageLayout(
        self.texture_image,
        .r8g8b8a8_srgb,
        .transfer_dst_optimal,
        .shader_read_only_optimal,
    );

    self.dev.destroyBuffer(staging_buffer, null);
    self.dev.freeMemory(staging_buffer_memory, null);
}

fn createTextureImageView(self: *Self) !void {
    self.texture_image_view = try self.createImageView(
        self.texture_image,
        .r8g8b8a8_srgb,
        .{ .color_bit = true },
    );
}

fn createTextureSampler(self: *Self) !void {
    const properties = self.vki.getPhysicalDeviceProperties(self.physical_device);

    const sampler_info: vk.SamplerCreateInfo = .{
        .mag_filter = .linear,
        .min_filter = .linear,
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
        .anisotropy_enable = .true,
        .max_anisotropy = properties.limits.max_sampler_anisotropy,
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
        .compare_enable = .false,
        .compare_op = .always,
        .mipmap_mode = .linear,
        .mip_lod_bias = 0.0,
        .min_lod = 0.0,
        .max_lod = 0.0,
    };

    self.texture_image_sampler = try self.dev.createSampler(&sampler_info, null);
}

fn createImageView(
    self: *Self,
    image: vk.Image,
    format: vk.Format,
    aspect_flags: vk.ImageAspectFlags,
) !vk.ImageView {
    const view_info: vk.ImageViewCreateInfo = .{
        .image = image,
        .view_type = .@"2d",
        .format = format,
        .subresource_range = .{
            .aspect_mask = aspect_flags,
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
    };

    return self.dev.createImageView(&view_info, null);
}

fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
    const alloc_info: vk.CommandBufferAllocateInfo = .{
        .level = .primary,
        .command_pool = self.command_pool,
        .command_buffer_count = 1,
    };

    var command_buffer: vk.CommandBuffer = .null_handle;
    try self.dev.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));

    const begin_info: vk.CommandBufferBeginInfo = .{
        .flags = .{ .one_time_submit_bit = true },
    };

    try self.vkd.beginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
}

fn endSingleTimeCommands(self: *Self, command_buffer: vk.CommandBuffer) !void {
    try self.vkd.endCommandBuffer(command_buffer);

    const submit_info: vk.SubmitInfo = .{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&command_buffer),
    };

    try self.vkd.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit_info), .null_handle);
    try self.vkd.queueWaitIdle(self.graphics_queue);

    self.dev.freeCommandBuffers(self.command_pool, 1, @ptrCast(&command_buffer));
}

fn createImage(
    self: *Self,
    width: u32,
    height: u32,
    format: vk.Format,
    tiling: vk.ImageTiling,
    usage: vk.ImageUsageFlags,
    properties: vk.MemoryPropertyFlags,
    image: *vk.Image,
    image_memory: *vk.DeviceMemory,
) !void {
    const image_info: vk.ImageCreateInfo = .{
        .image_type = .@"2d",
        .extent = .{
            .width = width,
            .height = height,
            .depth = 1,
        },
        .mip_levels = 1,
        .array_layers = 1,
        .format = format,
        .tiling = tiling,
        .initial_layout = .undefined,
        .usage = usage,
        .samples = .{ .@"1_bit" = true },
        .sharing_mode = .exclusive,
    };

    image.* = try self.dev.createImage(&image_info, null);

    const mem_requirements = self.dev.getImageMemoryRequirements(image.*);

    const alloc_info: vk.MemoryAllocateInfo = .{
        .allocation_size = mem_requirements.size,
        .memory_type_index = try self.findMemoryType(
            mem_requirements.memory_type_bits,
            properties,
        ),
    };

    image_memory.* = try self.dev.allocateMemory(&alloc_info, null);

    try self.dev.bindImageMemory(image.*, image_memory.*, 0);
}

fn copyData(
    self: *const Self,
    comptime T: type,
    data: []const T,
    memory: vk.DeviceMemory,
    size: usize,
) !void {
    if (data.len != size) {
        return error.WrongLen;
    }
    const mapped_memory = try self.dev.mapMemory(memory, 0, size, .{});
    const mapped_memory_buffer: [*]T = @ptrCast(mapped_memory);
    @memcpy(mapped_memory_buffer[0..size], data);
    self.dev.unmapMemory(memory);
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

    const clear_values = [_]vk.ClearValue{
        .{
            .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 1.0 } },
        },
        .{
            .depth_stencil = .{ .depth = 1.0, .stencil = 0 },
        },
    };

    const render_pass_info: vk.RenderPassBeginInfo = .{
        .render_pass = self.render_pass,
        .framebuffer = self.swapchain_framebuffers.items[image_index],
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        },
        .clear_value_count = clear_values.len,
        .p_clear_values = clear_values[0..].ptr,
    };

    self.vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");

    self.vkd.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);

    const vertex_buffers = [_]vk.Buffer{self.vertex_buffer};
    const offsets = [_]vk.DeviceSize{0};

    self.vkd.cmdBindVertexBuffers(
        command_buffer,
        0,
        1,
        vertex_buffers[0..].ptr,
        offsets[0..].ptr,
    );

    self.vkd.cmdBindIndexBuffer(command_buffer, self.index_buffer, 0, .uint16);

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

    self.vkd.cmdBindDescriptorSets(
        command_buffer,
        .graphics,
        self.pipeline_layout,
        0,
        1,
        self.descriptor_sets[self.current_frame .. self.current_frame + 1].ptr,
        0,
        null,
    );

    self.vkd.cmdDrawIndexed(command_buffer, @intCast(indices.len), 1, 0, 0, 0);

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

    try self.updateUniformBuffer(self.current_frame);

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

fn updateUniformBuffer(self: *Self, current_image: usize) !void {
    const current_time = try std.time.Instant.now();

    const time: f32 = @as(f32, @floatFromInt(current_time.since(self.start_time))) / @as(f32, @floatFromInt(std.time.ns_per_s));

    var ubo: UniformBufferObject = .{
        .model = zlm.Mat4.createAngleAxis(.unitZ, time * zlm.toRadians(90.0)),
        .view = .createLookAt(.all(2.0), .zero, .unitZ),
        .proj = .createPerspective(
            zlm.toRadians(45.0),
            @as(f32, @floatFromInt(self.swapchain_extent.width)) / @as(f32, @floatFromInt(self.swapchain_extent.height)),
            0.1,
            10.0,
        ),
    };

    ubo.proj.fields[1][1] *= -1;

    const dest: *UniformBufferObject = @ptrCast(@alignCast(self.uniform_buffers_mapped[current_image].?));
    dest.* = ubo;
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

    self.cleanupSwapChain();

    try self.createSwapChain();
    try self.createImageViews();
    try self.createDepthResources();
    try self.createFramebuffers();
}

fn cleanupSwapChain(self: *Self) void {
    self.dev.destroyImageView(self.depth_image_view, null);
    self.dev.destroyImage(self.depth_image, null);
    self.dev.freeMemory(self.depth_image_memory, null);

    for (self.swapchain_framebuffers.items) |framebuffer| {
        self.dev.destroyFramebuffer(framebuffer, null);
    }

    for (self.swapchain_image_views.items) |view| {
        self.dev.destroyImageView(view, null);
    }

    self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);
}

fn cleanup(self: *Self) void {
    self.cleanupSwapChain();

    self.dev.destroySampler(self.texture_image_sampler, null);
    self.dev.destroyImageView(self.texture_image_view, null);

    self.dev.destroyImage(self.texture_image, null);
    self.dev.freeMemory(self.texture_image_memory, null);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        self.dev.destroyBuffer(self.uniform_buffers[i], null);
        self.dev.freeMemory(self.uniform_buffers_memory[i], null);
    }

    self.dev.destroyDescriptorPool(self.descriptor_pool, null);

    self.dev.destroyDescriptorSetLayout(self.descriptor_set_layout, null);

    self.dev.destroyBuffer(self.index_buffer, null);
    self.dev.freeMemory(self.index_buffer_memory, null);

    self.dev.destroyBuffer(self.vertex_buffer, null);
    self.dev.freeMemory(self.vertex_buffer_memory, null);

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
