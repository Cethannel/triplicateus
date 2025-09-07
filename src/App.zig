const Window = @import("Window.zig");
const glfw = @import("glfw");

const Pipeline = @import("Pipeline.zig");

window: Window,
pipeline: Pipeline,

pub const WIDTH = 800;
pub const HEIGHT = 600;

const Self = @This();

pub fn init() !Self {
    return Self{ .window = try .init(WIDTH, HEIGHT, "Hello Vulkan!"), .pipeline = .init() };
}

pub fn run(self: *Self) void {
    while (!glfw.windowShouldClose(self.window.window)) {
        glfw.pollEvents();
    }
}
