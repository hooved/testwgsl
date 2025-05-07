if (!navigator.gpu) throw new Error("WebGPU not supported.");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
	requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [], powerPreference: "high-performance",
  requiredLimits: {maxStorageBufferBindingSize: 262668288, maxBufferSize: 262668288,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup},
});

const createEmptyBuf = (size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};
const createUniformBuf = (size) => { return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}) }
const addComputePass = (commandEncoder, pipeline, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((buffer, index) => ({ binding: index, resource: { buffer } }))
  });
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

// adapted from https://gist.github.com/wpmed92/c045e98fdb5916670c31383096706406
const test_multiply_bitcasted_inf = `fn INFINITY() -> f32 { let bits = 0x7F800000u; return bitcast<f32>(bits); }
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let i: u32 = index.x;
  data0[i] = data1[i] * INFINITY();
}}`;

const test_multiply_runtime_bitcasted_inf = `
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let bits: u32 = 0x7F800000u;
  let inf: f32 = bitcast<f32>(bits);
  let i: u32 = index.x;
  data0[i] = data1[i] * inf;
}}`;

const test_multiply_runtime_bitcasted_inf2 = `
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let i: u32 = index.x;
  let bits: u32 = i + 0x7F800000u - i;
  let inf:  f32 = bitcast<f32>(bits);
  data0[i] = data1[i] * inf;
}}`;

const test_multiply_runtime_overflow_inf = `
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let i: u32 = index.x;
  let inf: f32 = exp2(data1[i] * 0.0 + 129.0);
  data0[i] = data1[i] * inf;
}}`;

const test_multiply_uniform_inf = `@group(0) @binding(0) var<uniform> INFINITY : f32;
@group(0) @binding(1) var<storage, read_write> data0: array<f32>;
@group(0) @binding(2) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let i: u32 = index.x;
  data0[i] = data1[i] * INFINITY;
}}`;

const test_assign_bitcasted_inf = `fn INFINITY() -> f32 { let bits = 0x7F800000u; return bitcast<f32>(bits); }
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
  let i: u32 = index.x;
  data0[i] = INFINITY();
}}`;

const setupTests = async () => {
  const buf_0 = createEmptyBuf(4);
  const buf_1 = createEmptyBuf(4);
  const buf_inf = createUniformBuf(4);
  device.queue.writeBuffer(buf_inf, 0, new Float32Array([Infinity]));
  const read_buf = device.createBuffer({size:buf_1.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const kernels = [
    test_multiply_bitcasted_inf, test_multiply_runtime_bitcasted_inf, test_multiply_runtime_bitcasted_inf2, test_multiply_runtime_overflow_inf,
    test_multiply_uniform_inf, test_assign_bitcasted_inf
  ];
  const pipelines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({
    layout: "auto", compute: { module: device.createShaderModule({ code: name }), entryPoint: "main" }})));

  const execute = async (...args) => {
    const commandEncoder = device.createCommandEncoder();
    device.queue.writeBuffer(buf_0, 0, new Float32Array([5]));

    addComputePass(commandEncoder, ...args);

    commandEncoder.copyBufferToBuffer(buf_1, 0, read_buf, 0, 4);
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    await read_buf.mapAsync(GPUMapMode.READ);
    const ret = new Float32Array(read_buf.size/4);
    ret.set(new Float32Array(read_buf.getMappedRange()));
    read_buf.unmap();
    return ret
  };

  const assertEqual = (ret, expected, testName) => {
    console.log(`\n`)
    console.log(testName);
    console.log(`ret = ${ret}`);
    const failMessage = `FAILED, expected ret=${expected}`;
    const worked = (ret === expected) ? "Success!" : failMessage;
    console.log(worked);
    //if (ret !== expected) throw new Error(failMessage);
  };

  const testMultiplyBitcastedInfinity = async () => {
    const ret = await execute(pipelines[0], [buf_1, buf_0], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testMultiplyBitcastedInfinity");
  };

  const testMultiplyRuntimeBitcastedInfinity = async () => {
    const ret = await execute(pipelines[1], [buf_1, buf_0], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testMultiplyRuntimeBitcastedInfinity");
  };

  const testMultiplyRuntimeBitcastedInfinity2 = async () => {
    const ret = await execute(pipelines[2], [buf_1, buf_0], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testMultiplyRuntimeBitcastedInfinity2");
  };

  const testMultiplyRuntimeOverflowInfinity = async () => {
    const ret = await execute(pipelines[3], [buf_1, buf_0], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testMultiplyRuntimeOverflowInfinity");
  };

  const testMultiplyUniformInfinity = async () => {
    const ret = await execute(pipelines[4], [buf_inf, buf_1, buf_0], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testMultiplyUniformInfinity");
  };

  const testAssignBitcastedInfinity = async () => {
    const ret = await execute(pipelines[5], [buf_1], [1, 1, 1]);
    const expected = Infinity;
    assertEqual(ret[0], expected, "testAssignBitcastedInfinity");
  };

  return { testMultiplyBitcastedInfinity, testMultiplyRuntimeBitcastedInfinity, testMultiplyRuntimeBitcastedInfinity2, testMultiplyRuntimeOverflowInfinity, 
    testMultiplyUniformInfinity, testAssignBitcastedInfinity };
};

export default setupTests;
