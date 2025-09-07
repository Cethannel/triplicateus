const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");

const Window = @import("Window.zig");

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;

pub const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR,
    formats: []vk.SurfaceFormatKHR,
    present_modes: []vk.PresentModeKHR,
};

pub const QueueFamilyIndixces = struct {
    graphics_family: ?u32,
    present_family: ?u32,
    pub fn is_complete(self: *const @This()) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

enable_validation_layers: bool = false,
properties: vk.PhysicalDeviceProperties,
instance: vk.Instance,
debugMessenger: vk.DebugUtilsMessengerEXT,
window: *Window,
command_pool: vk.CommandPool,
device: vk.Device,
surface: vk.SurfaceKHR,
graphics_queue: vk.Queue,
present_queue: vk.Queue,
vkb: vk.BaseWrapper,
vki: InstanceWrapper,
allocator: std.mem.Allocator,

const validationLayers: []const [*:0]const u8 = ([_][*:0]const u8{"VK_LAYER_KHRONOS_validation"})[0..];
const deviceExtensions: []const [*:0]const u8 = ([_][*:0]const u8{vk.extensions.khr_swapchain.name})[0..];

const Self = @This();

pub fn init(window: *Window, allocator: std.mem.Allocator) !Self {
    var self: Self = undefined;
    self.window = window;
    self.allocator = allocator;

    try self.createInstance();
    try self.setupDebugMessenger();
    try self.createSurface();
    try self.pickPhysicalDevice();
}

pub fn deinit(self: *Self) void {
    _ = self; // autofix
}

fn debug_callback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    p_calllback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(.c) vk.Bool32 {
    _ = message_severity; // autofix
    _ = message_type; // autofix
    _ = p_user_data; // autofix
    if (p_calllback_data.?.p_message) |msg| {
        std.debug.print("Validation layer: {s}\n", .{msg});
    } else {
        std.debug.print("Validation layer no msg\n", .{});
    }

    return .false;
}

pub fn createSurface(self: *Self) !void {
    return self.window.createWindowSurface(self.instance, &self.surface);
}

fn createInstance(self: *Self) !void {
    if (self.enable_validation_layers and !try self.checkValidationLayerSupport()) {
        std.debug.panic("validation layers requested, but not available!", .{});
    }

    const app_info: vk.ApplicationInfo = .{
        .s_type = .application_info,
        .p_application_name = "VulkanEngine App",
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .p_engine_name = "No Engine",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.API_VERSION_1_0),
    };

    const extensions = try self.getRequiredExtensions();
    defer self.allocator.free(extensions);
    var create_info: vk.InstanceCreateInfo = .{
        .p_application_info = &app_info,
        .enabled_extension_count = @intCast(extensions.len),
        .pp_enabled_layer_names = extensions.ptr,
    };

    var debug_create_info: vk.DebugUtilsMessengerCreateInfoEXT = undefined;
    if (self.enable_validation_layers) {
        create_info.enabled_layer_count = validationLayers.len;
        create_info.pp_enabled_layer_names = validationLayers.ptr;

        debug_create_info = populateDebugMessengerCreateInfo();
        create_info.p_next = @ptrCast(&debug_create_info);
    } else {
        create_info.enabled_extension_count = 0;
        create_info.p_next = null;
    }

    self.instance = try self.vkb.createInstance(&create_info, null);

    return self.hasGflwRequiredInstanceExtensions();
}

fn checkValidationLayerSupport(self: *const Self) !bool {
    var layer_count: u32 = 0;

    _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

    const available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
    defer self.allocator.free(available_layers);

    _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

    for (validationLayers) |layerName| {
        var found = false;
        const layer_len = std.mem.len(layerName);
        var padded_name = try std.ArrayList(u8).initCapacity(self.allocator, 256);
        defer padded_name.deinit(self.allocator);
        padded_name.appendSliceAssumeCapacity(layerName[0..layer_len]);
        padded_name.appendNTimesAssumeCapacity(0, 256 - layer_len);

        for (available_layers) |layer| {
            if (std.mem.eql(u8, padded_name.items, &layer.layer_name)) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;
        }
    }

    return true;
}

fn getRequiredExtensions(self: *const Self) ![][*:0]const u8 {
    var glfw_extention_count: u32 = 0;
    const glfw_extentions = glfw.getRequiredInstanceExtensions(&glfw_extention_count) orelse {
        return error.NoExtentions;
    };

    const len = glfw_extention_count + @as(u32, if (self.enable_validation_layers) 1 else 0);
    const extensions = try self.allocator.alloc([*:0]const u8, len);
    errdefer self.allocator.free(extensions);
    for (0..glfw_extention_count) |i| {
        extensions[i] = glfw_extentions[i];
    }

    if (self.enable_validation_layers) {
        extensions[len - 1] = vk.extensions.ext_debug_utils.name;
    }

    return extensions;
}

fn populateDebugMessengerCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return vk.DebugUtilsMessengerCreateInfoEXT{
        .message_severity = .{
            .error_bit_ext = true,
            .warning_bit_ext = true,
        },
        .message_type = .{
            .general_bit_ext = true,
            .validation_bit_ext = true,
            .performance_bit_ext = true,
        },
        .pfn_user_callback = &debug_callback,
    };
}

fn hasGflwRequiredInstanceExtensions(self: *const Self) !void {
    var extension_count: u32 = 0;
    _ = try self.vkb.enumerateInstanceExtensionProperties(null, &extension_count, null);

    const extensions = try self.allocator.alloc(vk.ExtensionProperties, extension_count);
    defer self.allocator.free(extensions);

    _ = try self.vkb.enumerateInstanceExtensionProperties(null, &extension_count, extensions.ptr);

    std.debug.print("available extensions:\n", .{});
    var available = std.StringArrayHashMap(void).init(self.allocator);
    for (extensions) |extension| {
        std.debug.print("\t{s}\n", .{extension.extension_name});
        try available.put(&extension.extension_name, undefined);
    }

    std.debug.print("required extensions:\n", .{});
    const requiredExtensions = try self.getRequiredExtensions();
    defer self.allocator.free(requiredExtensions);
    for (requiredExtensions) |required| {
        const required_len = std.mem.len(required);
        std.debug.print("\t{s}\n", .{required});
        if (!available.contains(required[0..required_len])) {
            return error.MissingRequiredGLFWExtension;
        }
    }
}

fn setupDebugMessenger(self: *Self) !void {
    if (!self.enable_validation_layers) {
        return;
    }

    var createInfo = populateDebugMessengerCreateInfo();
    try self.createDebugUtilsMessengerEXT(self.instance, &createInfo, null, &self.debugMessenger);
}

fn createDebugUtilsMessengerEXT(
    self: *const Self,
    instance: vk.Instance,
    p_create_info: *const vk.DebugUtilsMessengerCreateInfoEXT,
    p_allocator: ?*const vk.AllocationCallbacks,
    p_debug_messenger: *vk.DebugUtilsMessengerEXT,
) !void {
    const func = self.vkb.getInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) |fun| {
        return fun(instance, p_create_info, p_allocator, p_debug_messenger);
    } else {
        error.ExtensionNotPresent;
    }
}

fn pickPhysicalDevice(self: *Self) !void {
    self.vki = InstanceWrapper.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);

    var device_count = 0;
    try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);
}
