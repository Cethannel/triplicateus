const std = @import("std");
const zlm = @import("zlm");
const gpu = std.gpu;

extern const a_pos: @Vector(2, f32) addrspace(.input);
extern const a_color: @Vector(3, f32) addrspace(.input);

extern const view_matrix: zlm.Mat4 addrspace(.uniform);

extern var v_color: @Vector(3, f32) addrspace(.output);

export fn main() callconv(.spirv_vertex) void {
    gpu.location(&a_pos, 0);
    gpu.location(&a_color, 1);
    gpu.location(&v_color, 0);

    gpu.binding(&view_matrix, 0, 0);

    gpu.position_out.* = .{ a_pos[0], a_pos[1], 0.0, 1.0 };
    v_color = a_color;
}
