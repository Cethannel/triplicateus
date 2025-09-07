const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const zlm = @import("zlm");

const BaseWrapper = vk.BaseWrapper;

const HelloTriangle = @import("HelloTriangle.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var app = HelloTriangle{
        .allocator = allocator,
    };
    try app.run();
}

fn customGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction {
    return glfw.getInstanceProcAddress(@intFromEnum(instance), procname);
}

fn println(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}
