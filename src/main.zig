const std = @import("std");
const vk = @import("vulkan");
const glfw = @import("glfw");
const zlm = @import("zlm");

const BaseWrapper = vk.BaseWrapper;

const App = @import("App.zig");

pub fn main() !void {
    var app = try App.init();

    app.run();
}

fn customGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction {
    return glfw.getInstanceProcAddress(@intFromEnum(instance), procname);
}

fn println(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}
