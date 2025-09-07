const vert: []const u8 = @embedFile("vertex_shader");
const frag: []const u8 = @embedFile("fragment_shader");

const std = @import("std");

const Self = @This();

pub fn init() Self {
    std.debug.print("Vertex size: {d}\n", .{vert.len});
    std.debug.print("Fragment size: {d}\n", .{frag.len});
    return .{};
}
