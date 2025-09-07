const glfw = @import("glfw");
const vk = @import("vulkan");

window: *glfw.Window,
width: u32,
height: u32,
name: [:0]const u8,

const Self = @This();

pub fn init(width: u32, height: u32, name: [:0]const u8) !Self {
    var out = Self{
        .width = width,
        .height = height,
        .name = name,
        .window = undefined,
    };

    try out.initWindow();

    return out;
}

fn initWindow(self: *Self) !void {
    try glfw.init();
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);
    glfw.windowHint(glfw.Resizable, @intFromBool(false));

    self.window = try glfw.createWindow(@intCast(self.width), @intCast(self.height), self.name, null, null);
}

pub fn createWindowSurface(self: *Self, instance: vk.Instance, surface: *vk.SurfaceKHR) !void {
    if (glfw.createWindowSurface(instance, self.window, null, surface) != .success) {
        return error.CreateWindowSurfaceFailed;
    }
}

pub fn deinit(self: *Self) void {
    glfw.destroyWindow(self.window);
    glfw.terminate();
}
