import React, { useEffect, useRef, useState, useCallback } from 'react';

// ============= SIMPLEX NOISE =============
class SimplexNoise {
  private perm: number[] = [];
  constructor(seed: number = Math.random() * 65536) {
    const p: number[] = [];
    for (let i = 0; i < 256; i++) p[i] = i;
    let s = seed;
    for (let i = 255; i > 0; i--) {
      s = (s * 16807 + 0) % 2147483647;
      const j = s % (i + 1);
      [p[i], p[j]] = [p[j], p[i]];
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
  }
  private grad(hash: number, x: number, y: number): number {
    const h = hash & 7;
    const u = h < 4 ? x : y;
    const v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
  }
  noise2D(x: number, y: number): number {
    const F2 = 0.5 * (Math.sqrt(3) - 1);
    const G2 = (3 - Math.sqrt(3)) / 6;
    const s = (x + y) * F2;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const t = (i + j) * G2;
    const X0 = i - t, Y0 = j - t;
    const x0 = x - X0, y0 = y - Y0;
    const i1 = x0 > y0 ? 1 : 0;
    const j1 = x0 > y0 ? 0 : 1;
    const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
    const ii = i & 255, jj = j & 255;
    let n0 = 0, n1 = 0, n2 = 0;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 >= 0) { t0 *= t0; n0 = t0 * t0 * this.grad(this.perm[ii + this.perm[jj]], x0, y0); }
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 >= 0) { t1 *= t1; n1 = t1 * t1 * this.grad(this.perm[ii + i1 + this.perm[jj + j1]], x1, y1); }
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 >= 0) { t2 *= t2; n2 = t2 * t2 * this.grad(this.perm[ii + 1 + this.perm[jj + 1]], x2, y2); }
    return 70 * (n0 + n1 + n2);
  }
  octave2D(x: number, y: number, octaves: number, persistence: number): number {
    let total = 0, frequency = 1, amplitude = 1, maxVal = 0;
    for (let i = 0; i < octaves; i++) {
      total += this.noise2D(x * frequency, y * frequency) * amplitude;
      maxVal += amplitude;
      amplitude *= persistence;
      frequency *= 2;
    }
    return total / maxVal;
  }
}

// ============= BLOCK DEFINITIONS =============
interface BlockDef {
  id: number;
  name: string;
  color: [number, number, number];
  transparent: boolean;
  solid: boolean;
  light: number;
  hardness: number;
}

const BLOCKS: Record<number, BlockDef> = {
  0: { id: 0, name: 'Air', color: [0, 0, 0], transparent: true, solid: false, light: 0, hardness: 0 },
  1: { id: 1, name: 'Stone', color: [128, 128, 128], transparent: false, solid: true, light: 0, hardness: 3 },
  2: { id: 2, name: 'Grass', color: [80, 180, 60], transparent: false, solid: true, light: 0, hardness: 1 },
  3: { id: 3, name: 'Dirt', color: [139, 90, 43], transparent: false, solid: true, light: 0, hardness: 1 },
  4: { id: 4, name: 'Cobblestone', color: [100, 100, 100], transparent: false, solid: true, light: 0, hardness: 3 },
  5: { id: 5, name: 'Oak Planks', color: [160, 120, 60], transparent: false, solid: true, light: 0, hardness: 2 },
  6: { id: 6, name: 'Oak Log', color: [100, 70, 30], transparent: false, solid: true, light: 0, hardness: 2 },
  7: { id: 7, name: 'Leaves', color: [40, 140, 30], transparent: true, solid: true, light: 0, hardness: 0.5 },
  8: { id: 8, name: 'Sand', color: [220, 200, 130], transparent: false, solid: true, light: 0, hardness: 1 },
  9: { id: 9, name: 'Water', color: [30, 80, 200], transparent: true, solid: false, light: 0, hardness: 0 },
  10: { id: 10, name: 'Coal Ore', color: [60, 60, 60], transparent: false, solid: true, light: 0, hardness: 4 },
  11: { id: 11, name: 'Iron Ore', color: [160, 130, 110], transparent: false, solid: true, light: 0, hardness: 4 },
  12: { id: 12, name: 'Gold Ore', color: [200, 180, 50], transparent: false, solid: true, light: 0, hardness: 4 },
  13: { id: 13, name: 'Diamond Ore', color: [80, 200, 220], transparent: false, solid: true, light: 0, hardness: 5 },
  14: { id: 14, name: 'Bedrock', color: [40, 40, 40], transparent: false, solid: true, light: 0, hardness: -1 },
  15: { id: 15, name: 'Gravel', color: [140, 130, 130], transparent: false, solid: true, light: 0, hardness: 1 },
  16: { id: 16, name: 'Oak Leaves', color: [50, 150, 40], transparent: true, solid: true, light: 0, hardness: 0.3 },
  17: { id: 17, name: 'Glass', color: [200, 220, 255], transparent: true, solid: true, light: 0, hardness: 0.5 },
  18: { id: 18, name: 'Torch', color: [255, 200, 50], transparent: true, solid: false, light: 14, hardness: 0 },
  19: { id: 19, name: 'Crafting Table', color: [140, 90, 40], transparent: false, solid: true, light: 0, hardness: 2 },
  20: { id: 20, name: 'Furnace', color: [110, 110, 110], transparent: false, solid: true, light: 0, hardness: 3 },
  21: { id: 21, name: 'Snow', color: [240, 245, 255], transparent: false, solid: true, light: 0, hardness: 0.5 },
  22: { id: 22, name: 'Ice', color: [150, 200, 255], transparent: true, solid: true, light: 0, hardness: 1 },
  23: { id: 23, name: 'Cactus', color: [30, 130, 30], transparent: false, solid: true, light: 0, hardness: 1 },
  24: { id: 24, name: 'Clay', color: [160, 165, 175], transparent: false, solid: true, light: 0, hardness: 1 },
  25: { id: 25, name: 'Glowstone', color: [255, 230, 130], transparent: false, solid: true, light: 15, hardness: 1 },
  26: { id: 26, name: 'Obsidian', color: [20, 10, 30], transparent: false, solid: true, light: 0, hardness: 50 },
  27: { id: 27, name: 'Netherrack', color: [120, 30, 30], transparent: false, solid: true, light: 0, hardness: 1 },
  28: { id: 28, name: 'Lava', color: [255, 100, 0], transparent: true, solid: false, light: 15, hardness: 0 },
  29: { id: 29, name: 'Emerald Ore', color: [50, 200, 80], transparent: false, solid: true, light: 0, hardness: 5 },
  30: { id: 30, name: 'Redstone Ore', color: [180, 30, 30], transparent: false, solid: true, light: 0, hardness: 4 },
  31: { id: 31, name: 'Lapis Ore', color: [30, 50, 180], transparent: false, solid: true, light: 0, hardness: 4 },
  32: { id: 32, name: 'Bricks', color: [160, 80, 60], transparent: false, solid: true, light: 0, hardness: 3 },
  33: { id: 33, name: 'Wool White', color: [230, 230, 230], transparent: false, solid: true, light: 0, hardness: 1 },
  34: { id: 34, name: 'Wool Red', color: [180, 40, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  35: { id: 35, name: 'Wool Blue', color: [40, 40, 180], transparent: false, solid: true, light: 0, hardness: 1 },
  36: { id: 36, name: 'Wool Green', color: [40, 140, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  37: { id: 37, name: 'Concrete', color: [200, 200, 200], transparent: false, solid: true, light: 0, hardness: 2 },
  38: { id: 38, name: 'Birch Log', color: [200, 195, 180], transparent: false, solid: true, light: 0, hardness: 2 },
  39: { id: 39, name: 'Spruce Log', color: [60, 40, 20], transparent: false, solid: true, light: 0, hardness: 2 },
  40: { id: 40, name: 'Copper Ore', color: [180, 120, 80], transparent: false, solid: true, light: 0, hardness: 4 },
  41: { id: 41, name: 'Sandstone', color: [210, 195, 140], transparent: false, solid: true, light: 0, hardness: 2 },
  42: { id: 42, name: 'Bookshelf', color: [120, 90, 50], transparent: false, solid: true, light: 0, hardness: 2 },
  43: { id: 43, name: 'Mossy Cobble', color: [90, 110, 80], transparent: false, solid: true, light: 0, hardness: 3 },
  44: { id: 44, name: 'Stone Bricks', color: [120, 120, 120], transparent: false, solid: true, light: 0, hardness: 3 },
  45: { id: 45, name: 'Nether Brick', color: [50, 20, 20], transparent: false, solid: true, light: 0, hardness: 3 },
  46: { id: 46, name: 'End Stone', color: [220, 220, 160], transparent: false, solid: true, light: 0, hardness: 3 },
  47: { id: 47, name: 'Redstone Lamp', color: [150, 100, 50], transparent: false, solid: true, light: 0, hardness: 1 },
  48: { id: 48, name: 'Redstone Block', color: [180, 30, 30], transparent: false, solid: true, light: 0, hardness: 3 },
  49: { id: 49, name: 'Iron Block', color: [200, 200, 200], transparent: false, solid: true, light: 0, hardness: 5 },
  50: { id: 50, name: 'Gold Block', color: [240, 200, 50], transparent: false, solid: true, light: 0, hardness: 4 },
  51: { id: 51, name: 'Diamond Block', color: [80, 220, 240], transparent: false, solid: true, light: 0, hardness: 5 },
  52: { id: 52, name: 'Emerald Block', color: [50, 200, 80], transparent: false, solid: true, light: 0, hardness: 5 },
  53: { id: 53, name: 'Lapis Block', color: [30, 50, 180], transparent: false, solid: true, light: 0, hardness: 4 },
  54: { id: 54, name: 'Redstone Wire', color: [200, 30, 30], transparent: true, solid: false, light: 0, hardness: 0 },
  55: { id: 55, name: 'Redstone Torch', color: [200, 50, 50], transparent: true, solid: false, light: 8, hardness: 0 },
  56: { id: 56, name: 'Lever', color: [100, 80, 60], transparent: true, solid: false, light: 0, hardness: 0 },
  57: { id: 57, name: 'Button', color: [120, 120, 120], transparent: true, solid: false, light: 0, hardness: 0 },
  58: { id: 58, name: 'Pressure Plate', color: [140, 140, 140], transparent: true, solid: false, light: 0, hardness: 0 },
  59: { id: 59, name: 'Piston', color: [160, 140, 100], transparent: false, solid: true, light: 0, hardness: 2 },
  60: { id: 60, name: 'Sticky Piston', color: [120, 180, 80], transparent: false, solid: true, light: 0, hardness: 2 },
  61: { id: 61, name: 'Dispenser', color: [110, 110, 110], transparent: false, solid: true, light: 0, hardness: 3 },
  62: { id: 62, name: 'Dropper', color: [100, 100, 100], transparent: false, solid: true, light: 0, hardness: 3 },
  63: { id: 63, name: 'Hopper', color: [80, 80, 80], transparent: false, solid: true, light: 0, hardness: 3 },
  64: { id: 64, name: 'Chest', color: [140, 100, 50], transparent: false, solid: true, light: 0, hardness: 2 },
  65: { id: 65, name: 'Anvil', color: [60, 60, 60], transparent: false, solid: true, light: 0, hardness: 5 },
  66: { id: 66, name: 'Enchanting Table', color: [80, 30, 100], transparent: false, solid: true, light: 7, hardness: 5 },
  67: { id: 67, name: 'Brewing Stand', color: [100, 80, 60], transparent: false, solid: true, light: 1, hardness: 3 },
  68: { id: 68, name: 'Cauldron', color: [50, 50, 50], transparent: false, solid: true, light: 0, hardness: 3 },
  69: { id: 69, name: 'Portal Frame', color: [40, 80, 40], transparent: false, solid: true, light: 1, hardness: -1 },
  70: { id: 70, name: 'Portal', color: [100, 0, 200], transparent: true, solid: false, light: 12, hardness: 0 },
  71: { id: 71, name: 'End Portal', color: [10, 10, 30], transparent: true, solid: false, light: 15, hardness: 0 },
  72: { id: 72, name: 'Dragon Egg', color: [20, 0, 30], transparent: false, solid: true, light: 1, hardness: 3 },
  73: { id: 73, name: 'Beacon', color: [150, 220, 255], transparent: true, solid: true, light: 15, hardness: 3 },
  74: { id: 74, name: 'Hay Bale', color: [180, 160, 50], transparent: false, solid: true, light: 0, hardness: 1 },
  75: { id: 75, name: 'Wool Yellow', color: [220, 200, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  76: { id: 76, name: 'Wool Black', color: [20, 20, 20], transparent: false, solid: true, light: 0, hardness: 1 },
  77: { id: 77, name: 'Wool Brown', color: [120, 80, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  78: { id: 78, name: 'Wool Pink', color: [230, 150, 180], transparent: false, solid: true, light: 0, hardness: 1 },
  79: { id: 79, name: 'Wool Orange', color: [230, 140, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  80: { id: 80, name: 'Wool Purple', color: [120, 50, 180], transparent: false, solid: true, light: 0, hardness: 1 },
  81: { id: 81, name: 'Wool Cyan', color: [50, 180, 200], transparent: false, solid: true, light: 0, hardness: 1 },
  82: { id: 82, name: 'Wool Lime', color: [120, 220, 50], transparent: false, solid: true, light: 0, hardness: 1 },
  83: { id: 83, name: 'Wool Gray', color: [80, 80, 80], transparent: false, solid: true, light: 0, hardness: 1 },
  84: { id: 84, name: 'Wool Light Gray', color: [160, 160, 160], transparent: false, solid: true, light: 0, hardness: 1 },
  85: { id: 85, name: 'Wool Magenta', color: [200, 80, 200], transparent: false, solid: true, light: 0, hardness: 1 },
  86: { id: 86, name: 'Pumpkin', color: [200, 120, 30], transparent: false, solid: true, light: 0, hardness: 1 },
  87: { id: 87, name: 'Melon', color: [100, 180, 50], transparent: false, solid: true, light: 0, hardness: 1 },
  88: { id: 88, name: 'Mycelium', color: [100, 80, 100], transparent: false, solid: true, light: 0, hardness: 1 },
  89: { id: 89, name: 'Podzol', color: [80, 60, 30], transparent: false, solid: true, light: 0, hardness: 1 },
  90: { id: 90, name: 'Coarse Dirt', color: [120, 80, 40], transparent: false, solid: true, light: 0, hardness: 1 },
  91: { id: 91, name: 'Prismarine', color: [80, 160, 150], transparent: false, solid: true, light: 0, hardness: 3 },
  92: { id: 92, name: 'Sea Lantern', color: [150, 200, 220], transparent: false, solid: true, light: 15, hardness: 1 },
  93: { id: 93, name: 'Magma Block', color: [180, 60, 20], transparent: false, solid: true, light: 3, hardness: 1 },
  94: { id: 94, name: 'Bone Block', color: [220, 210, 180], transparent: false, solid: true, light: 0, hardness: 2 },
  95: { id: 95, name: 'Terracotta', color: [160, 100, 70], transparent: false, solid: true, light: 0, hardness: 2 },
  96: { id: 96, name: 'Quartz Block', color: [230, 225, 220], transparent: false, solid: true, light: 0, hardness: 2 },
  97: { id: 97, name: 'Purpur Block', color: [160, 120, 160], transparent: false, solid: true, light: 0, hardness: 2 },
  98: { id: 98, name: 'Concrete Red', color: [160, 50, 50], transparent: false, solid: true, light: 0, hardness: 2 },
  99: { id: 99, name: 'Concrete Blue', color: [50, 50, 160], transparent: false, solid: true, light: 0, hardness: 2 },
  100: { id: 100, name: 'Concrete Green', color: [50, 140, 50], transparent: false, solid: true, light: 0, hardness: 2 },
};

// ============= CHUNK =============
const CHUNK_SIZE = 16;
const CHUNK_HEIGHT = 128;

interface Chunk {
  x: number;
  z: number;
  blocks: Uint8Array;
  mesh: WebGLBuffer | null;
  vertexCount: number;
  dirty: boolean;
}

function createChunk(x: number, z: number): Chunk {
  return {
    x, z,
    blocks: new Uint8Array(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE),
    mesh: null,
    vertexCount: 0,
    dirty: true
  };
}

function getBlockIndex(lx: number, ly: number, lz: number): number {
  return ly * CHUNK_SIZE * CHUNK_SIZE + lz * CHUNK_SIZE + lx;
}

// ============= WORLD GENERATION =============
function generateChunk(chunk: Chunk, noise: SimplexNoise, seed: number): void {
  const wx = chunk.x * CHUNK_SIZE;
  const wz = chunk.z * CHUNK_SIZE;
  
  for (let lx = 0; lx < CHUNK_SIZE; lx++) {
    for (let lz = 0; lz < CHUNK_SIZE; lz++) {
      const x = wx + lx;
      const z = wz + lz;
      
      // Biome determination
      const temp = noise.noise2D(x * 0.003, z * 0.003);
      const moisture = noise.noise2D(x * 0.004 + 100, z * 0.004 + 100);
      
      // Height
      let height = 64;
      height += noise.octave2D(x * 0.01, z * 0.01, 4, 0.5) * 20;
      height += noise.octave2D(x * 0.05, z * 0.05, 2, 0.5) * 5;
      
      // Mountains
      if (temp < -0.3) {
        height += noise.octave2D(x * 0.02, z * 0.02, 3, 0.6) * 30;
      }
      
      height = Math.floor(Math.max(1, Math.min(CHUNK_HEIGHT - 1, height)));
      
      // Surface block
      let surfaceBlock = 2; // grass
      if (temp > 0.4 && moisture < -0.2) surfaceBlock = 8; // desert sand
      if (temp < -0.4) surfaceBlock = 21; // snow
      
      for (let y = 0; y < CHUNK_HEIGHT; y++) {
        const idx = getBlockIndex(lx, y, lz);
        
        if (y === 0) {
          chunk.blocks[idx] = 14; // bedrock
        } else if (y < height - 4) {
          chunk.blocks[idx] = 1; // stone
          // Ores
          const oreNoise = noise.noise2D(x * 0.1 + y * 0.1, z * 0.1);
          if (y < 16 && oreNoise > 0.7) chunk.blocks[idx] = 13; // diamond
          else if (y < 32 && oreNoise > 0.6) chunk.blocks[idx] = 29; // emerald
          else if (y < 40 && oreNoise > 0.55) chunk.blocks[idx] = 12; // gold
          else if (y < 64 && oreNoise > 0.5) chunk.blocks[idx] = 11; // iron
          else if (oreNoise > 0.45) chunk.blocks[idx] = 10; // coal
          
          // Caves
          const cave = noise.noise2D(x * 0.05 + y * 0.05, z * 0.05 + y * 0.03);
          if (cave > 0.5 && y > 5 && y < height - 5) {
            chunk.blocks[idx] = 0;
          }
        } else if (y < height) {
          chunk.blocks[idx] = 3; // dirt
        } else if (y === height) {
          chunk.blocks[idx] = surfaceBlock;
        } else if (y <= 62 && y > height) {
          chunk.blocks[idx] = 9; // water
        } else {
          chunk.blocks[idx] = 0; // air
        }
      }
      
      // Trees
      if (height > 62 && height < 90 && surfaceBlock === 2) {
        const treeNoise = noise.noise2D(x * 0.5, z * 0.5);
        if (treeNoise > 0.6 && lx > 2 && lx < 13 && lz > 2 && lz < 13) {
          const treeHeight = 4 + Math.floor(Math.abs(noise.noise2D(x * 2, z * 2)) * 3);
          for (let ty = 1; ty <= treeHeight; ty++) {
            const tidx = getBlockIndex(lx, height + ty, lz);
            if (height + ty < CHUNK_HEIGHT) chunk.blocks[tidx] = 6; // log
          }
          // Leaves
          for (let ly2 = -2; ly2 <= 2; ly2++) {
            for (let lz2 = -2; lz2 <= 2; lz2++) {
              for (let lyy = treeHeight - 1; lyy <= treeHeight + 2; lyy++) {
                const nlx = lx + ly2, nlz = lz + lz2, nly = height + lyy;
                if (nlx >= 0 && nlx < CHUNK_SIZE && nlz >= 0 && nlz < CHUNK_SIZE && nly < CHUNK_HEIGHT) {
                  if (Math.abs(ly2) + Math.abs(lz2) < 4) {
                    const lidx = getBlockIndex(nlx, nly, nlz);
                    if (chunk.blocks[lidx] === 0) chunk.blocks[lidx] = 7;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ============= MESH GENERATION =============
function buildChunkMesh(chunk: Chunk, chunks: Map<string, Chunk>): Float32Array {
  const vertices: number[] = [];
  
  const faces = [
    { dir: [0, 1, 0], corners: [[0,1,0],[1,1,0],[1,1,1],[0,1,1]], normal: [0,1,0] },   // top
    { dir: [0, -1, 0], corners: [[0,0,1],[1,0,1],[1,0,0],[0,0,0]], normal: [0,-1,0] },  // bottom
    { dir: [1, 0, 0], corners: [[1,0,0],[1,1,0],[1,1,1],[1,0,1]], normal: [1,0,0] },    // right
    { dir: [-1, 0, 0], corners: [[0,0,1],[0,1,1],[0,1,0],[0,0,0]], normal: [-1,0,0] },  // left
    { dir: [0, 0, 1], corners: [[0,0,1],[1,0,1],[1,1,1],[0,1,1]], normal: [0,0,1] },    // front
    { dir: [0, 0, -1], corners: [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], normal: [0,0,-1] },  // back
  ];
  
  for (let y = 0; y < CHUNK_HEIGHT; y++) {
    for (let z = 0; z < CHUNK_SIZE; z++) {
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const idx = getBlockIndex(x, y, z);
        const blockId = chunk.blocks[idx];
        if (blockId === 0) continue;
        
        const block = BLOCKS[blockId];
        if (!block) continue;
        
        for (const face of faces) {
          const nx = x + face.dir[0];
          const ny = y + face.dir[1];
          const nz = z + face.dir[2];
          
          let neighborId = 0;
          if (nx >= 0 && nx < CHUNK_SIZE && ny >= 0 && ny < CHUNK_HEIGHT && nz >= 0 && nz < CHUNK_SIZE) {
            neighborId = chunk.blocks[getBlockIndex(nx, ny, nz)];
          } else if (ny >= 0 && ny < CHUNK_HEIGHT) {
            // Check neighboring chunk
            let cx = chunk.x, cz = chunk.z;
            let lx = nx, lz = nz;
            if (nx < 0) { cx--; lx = CHUNK_SIZE - 1; }
            if (nx >= CHUNK_SIZE) { cx++; lx = 0; }
            if (nz < 0) { cz--; lz = CHUNK_SIZE - 1; }
            if (nz >= CHUNK_SIZE) { cz++; lz = 0; }
            const nkey = `${cx},${cz}`;
            const nchunk = chunks.get(nkey);
            if (nchunk) neighborId = nchunk.blocks[getBlockIndex(lx, ny, lz)];
          }
          
          const neighbor = BLOCKS[neighborId];
          if (neighbor && !neighbor.transparent) continue;
          if (blockId === neighborId && block.transparent) continue;
          
          // Add face vertices
          const color = block.color;
          const shade = face.dir[1] === 1 ? 1.0 : face.dir[1] === -1 ? 0.5 : 
                       face.dir[0] !== 0 ? 0.7 : 0.8;
          
          for (const corner of face.corners) {
            vertices.push(
              chunk.x * CHUNK_SIZE + x + corner[0],
              y + corner[1],
              chunk.z * CHUNK_SIZE + z + corner[2],
              color[0] / 255 * shade,
              color[1] / 255 * shade,
              color[2] / 255 * shade,
              face.normal[0],
              face.normal[1],
              face.normal[2],
              block.transparent ? 0.7 : 1.0
            );
          }
        }
      }
    }
  }
  
  return new Float32Array(vertices);
}

// ============= MOB =============
interface Mob {
  id: number;
  type: string;
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  health: number;
  maxHealth: number;
  hostile: boolean;
  color: [number, number, number];
  size: number;
  aiTimer: number;
  targetX: number;
  targetZ: number;
}

// ============= MAIN APP =============
export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const gameRef = useRef<any>(null);
  
  const [screen, setScreen] = useState<'loading' | 'menu' | 'worlds' | 'create' | 'settings' | 'game' | 'servers' | 'pause' | 'death'>('menu');
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [settingsTab, setSettingsTab] = useState(0);
  const [showInventory, setShowInventory] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [hotbarSlot, setHotbarSlot] = useState(0);
  const [health, setHealth] = useState(20);
  const [hunger, setHunger] = useState(20);
  const [fps, setFps] = useState(60);
  const [coords, setCoords] = useState({ x: 0, y: 64, z: 0 });
  const [chatMessages, setChatMessages] = useState<string[]>([]);
  const [chatInput, setChatInput] = useState('');
  
  // Settings state
  const [settings, setSettings] = useState({
    difficulty: 2,
    renderDistance: 8,
    fov: 70,
    brightness: 1.0,
    mouseSens: 0.5,
    showCoords: true,
    showFps: true,
    gamemode: 0, // 0=survival, 1=creative, 2=adventure, 3=spectator
    clouds: true,
    particles: true,
    smoothLighting: true,
    ao: true,
    language: 'ru',
    volume: 0.7,
    autoSave: true,
    autoSaveInterval: 300,
    vsync: true,
    maxFps: 60,
    shadows: true,
    fog: true,
    flyMode: false,
  });
  
  // Inventory
  const [inventory, setInventory] = useState<(null | { id: number; count: number })[]>(
    Array(36).fill(null)
  );
  const [hotbar, setHotbar] = useState<(null | { id: number; count: number })[]>(
    Array(9).fill(null).map((_, i) => i === 0 ? { id: 1, count: 64 } : null)
  );
  
  // Mods
  const [mods, setMods] = useState<any[]>([]);
  const [texturePacks, setTexturePacks] = useState<any[]>([]);
  const [shaderPacks, setShaderPacks] = useState<any[]>([]);
  
  // Key bindings
  const [keyBindings, setKeyBindings] = useState<Record<string, string>>({
    forward: 'KeyW',
    back: 'KeyS',
    left: 'KeyA',
    right: 'KeyD',
    jump: 'Space',
    sneak: 'ShiftLeft',
    sprint: 'ControlLeft',
    attack: 'Mouse0',
    use: 'Mouse2',
    inventory: 'KeyE',
    drop: 'KeyQ',
    chat: 'KeyT',
    command: 'Slash',
    debug: 'F3',
    thirdPerson: 'F5',
    fullscreen: 'F11',
    pause: 'Escape',
    pickBlock: 'KeyF',
  });
  
  // Worlds
  const [worlds, setWorlds] = useState<{ name: string; seed: number; created: number }[]>([]);
  
  // Servers
  const [servers, setServers] = useState<{ name: string; address: string }[]>([]);
  
  // Achievements
  const [achievements, setAchievements] = useState<string[]>([]);
  const [stats, setStats] = useState({
    blocksBroken: 0,
    blocksPlaced: 0,
    mobsKilled: 0,
    distanceWalked: 0,
    timePlayed: 0,
    jumps: 0,
  });
  
  const unlockAchievement = (id: string) => {
    if (!achievements.includes(id)) {
      setAchievements(prev => [...prev, id]);
      setChatMessages(prev => [...prev, `§6🏆 Достижение получено: ${id}`]);
      localStorage.setItem('mc_achievements', JSON.stringify([...achievements, id]));
    }
  };

  // ============= CRAFTING RECIPES =============
  const recipes = [
    { input: [{ id: 6, count: 1 }], output: { id: 5, count: 4 }, name: 'Доски из бревна' },
    { input: [{ id: 5, count: 4 }], output: { id: 19, count: 1 }, name: 'Верстак' },
    { input: [{ id: 4, count: 8 }], output: { id: 20, count: 1 }, name: 'Печь' },
    { input: [{ id: 5, count: 2 }, { id: 1, count: 1 }], output: { id: 18, count: 4 }, name: 'Факелы' },
    { input: [{ id: 8, count: 4 }], output: { id: 41, count: 1 }, name: 'Песчаник' },
    { input: [{ id: 5, count: 6 }, { id: 1, count: 3 }], output: { id: 42, count: 1 }, name: 'Книжная полка' },
    { input: [{ id: 1, count: 9 }], output: { id: 49, count: 1 }, name: 'Железный блок' },
    { input: [{ id: 12, count: 9 }], output: { id: 50, count: 1 }, name: 'Золотой блок' },
    { input: [{ id: 13, count: 9 }], output: { id: 51, count: 1 }, name: 'Алмазный блок' },
    { input: [{ id: 29, count: 9 }], output: { id: 52, count: 1 }, name: 'Изумрудный блок' },
    { input: [{ id: 31, count: 9 }], output: { id: 53, count: 1 }, name: 'Лазуритовый блок' },
    { input: [{ id: 33, count: 1 }], output: { id: 33, count: 1 }, name: 'Белая шерсть' },
    { input: [{ id: 4, count: 4 }], output: { id: 44, count: 4 }, name: 'Каменные кирпичи' },
    { input: [{ id: 25, count: 4 }, { id: 30, count: 4 }], output: { id: 47, count: 1 }, name: 'Редстоун лампа' },
    { input: [{ id: 30, count: 9 }], output: { id: 48, count: 1 }, name: 'Редстоун блок' },
    { input: [{ id: 5, count: 7 }, { id: 1, count: 2 }], output: { id: 64, count: 1 }, name: 'Сундук' },
    { input: [{ id: 49, count: 3 }, { id: 1, count: 4 }], output: { id: 65, count: 1 }, name: 'Наковальня' },
    { input: [{ id: 49, count: 2 }, { id: 13, count: 1 }, { id: 32, count: 2 }], output: { id: 66, count: 1 }, name: 'Стол зачарований' },
  ];

  const canCraft = (recipe: typeof recipes[0]): boolean => {
    const available = [...hotbar, ...inventory].filter(Boolean) as { id: number; count: number }[];
    for (const inp of recipe.input) {
      const total = available.filter(item => item.id === inp.id).reduce((sum, item) => sum + item.count, 0);
      if (total < inp.count) return false;
    }
    return true;
  };

  const craft = (recipe: typeof recipes[0]) => {
    if (!canCraft(recipe)) return;
    
    // Remove inputs
    for (const inp of recipe.input) {
      let remaining = inp.count;
      for (let i = 0; i < hotbar.length && remaining > 0; i++) {
        if (hotbar[i] && hotbar[i]!.id === inp.id) {
          const take = Math.min(remaining, hotbar[i]!.count);
          hotbar[i] = { ...hotbar[i]!, count: hotbar[i]!.count - take };
          if (hotbar[i]!.count <= 0) hotbar[i] = null;
          remaining -= take;
        }
      }
      for (let i = 0; i < inventory.length && remaining > 0; i++) {
        if (inventory[i] && inventory[i]!.id === inp.id) {
          const take = Math.min(remaining, inventory[i]!.count);
          inventory[i] = { ...inventory[i]!, count: inventory[i]!.count - take };
          if (inventory[i]!.count <= 0) inventory[i] = null;
          remaining -= take;
        }
      }
    }
    setHotbar([...hotbar]);
    setInventory([...inventory]);
    
    // Add output
    addToInventory(recipe.output.id, recipe.output.count);
    playSound('place');
    setChatMessages(prev => [...prev, `Скрафчено: ${BLOCKS[recipe.output.id]?.name} x${recipe.output.count}`]);
  };
  
  // World data refs
  const worldDataRef = useRef({
    chunks: new Map<string, Chunk>(),
    noise: new SimplexNoise(12345),
    seed: 12345,
    mobs: [] as Mob[],
    time: 0,
    weather: 'clear' as 'clear' | 'rain' | 'snow' | 'storm',
    playerPos: { x: 8, y: 80, z: 8 },
    playerVel: { x: 0, y: 0, z: 0 },
    playerRot: { x: 0, y: 0 },
    keys: new Set<string>(),
    mouseDown: [false, false, false],
    rightHandItem: null as null | { id: number; count: number },
    flying: false,
  });

  // Death check
  useEffect(() => {
    if (health <= 0 && screen === 'game') {
      setScreen('death');
      document.exitPointerLock();
    }
  }, [health, screen]);

  // Auto-save
  useEffect(() => {
    if (screen !== 'game' || !settings.autoSave) return;
    const interval = setInterval(() => {
      const wd = worldDataRef.current;
      const saveData = {
        playerPos: wd.playerPos,
        playerRot: wd.playerRot,
        health,
        hunger,
        hotbar,
        inventory,
        time: wd.time,
        seed: wd.seed,
      };
      localStorage.setItem('mc_autosave', JSON.stringify(saveData));
    }, settings.autoSaveInterval * 1000);
    return () => clearInterval(interval);
  }, [screen, settings.autoSave, settings.autoSaveInterval, health, hunger, hotbar, inventory]);

  // ============= AUDIO SYSTEM =============
  const audioCtxRef = useRef<AudioContext | null>(null);
  
  const playSound = useCallback((type: 'break' | 'place' | 'step' | 'hurt' | 'click') => {
    if (!audioCtxRef.current) {
      try { audioCtxRef.current = new AudioContext(); } catch { return; }
    }
    const ctx = audioCtxRef.current;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    gain.gain.value = settings.volume * 0.3;
    
    switch (type) {
      case 'break':
        osc.frequency.value = 200;
        osc.type = 'square';
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        break;
      case 'place':
        osc.frequency.value = 400;
        osc.type = 'square';
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
        break;
      case 'step':
        osc.frequency.value = 100 + Math.random() * 50;
        osc.type = 'triangle';
        gain.gain.value = settings.volume * 0.1;
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
        break;
      case 'hurt':
        osc.frequency.value = 150;
        osc.type = 'sawtooth';
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
        break;
      case 'click':
        osc.frequency.value = 800;
        osc.type = 'sine';
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.03);
        break;
    }
    osc.start();
    osc.stop(ctx.currentTime + 0.2);
  }, [settings.volume]);

  // Load saved data
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('mc_settings');
      if (savedSettings) setSettings(JSON.parse(savedSettings));
      
      const savedWorlds = localStorage.getItem('mc_worlds');
      if (savedWorlds) setWorlds(JSON.parse(savedWorlds));
      
      const savedMods = localStorage.getItem('minecraft_mods_cache');
      if (savedMods) setMods(JSON.parse(savedMods));
      
      const savedPacks = localStorage.getItem('mc_texture_packs');
      if (savedPacks) setTexturePacks(JSON.parse(savedPacks));
      
      const savedShaders = localStorage.getItem('mc_shader_packs');
      if (savedShaders) setShaderPacks(JSON.parse(savedShaders));
      
      const savedKeys = localStorage.getItem('mc_keybindings');
      if (savedKeys) setKeyBindings(JSON.parse(savedKeys));
    } catch (e) { console.error('Load error:', e); }
  }, []);

  // Save settings
  useEffect(() => {
    localStorage.setItem('mc_settings', JSON.stringify(settings));
  }, [settings]);

  // ============= WebGL INIT =============
  const initGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' });
    if (!gl) { 
      alert('WebGL2 не поддерживается вашим браузером!');
      setScreen('menu');
      return; 
    }
    glRef.current = gl;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    // Vertex shader
    const vsSource = `#version 300 es
    in vec3 aPos;
    in vec3 aColor;
    in vec3 aNormal;
    in float aAlpha;
    
    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform float uTime;
    uniform float uBrightness;
    
    out vec3 vColor;
    out vec3 vNormal;
    out float vAlpha;
    out float vDist;
    out vec3 vWorldPos;
    
    void main() {
      vec4 worldPos = vec4(aPos, 1.0);
      vec4 viewPos = uView * worldPos;
      gl_Position = uProjection * viewPos;
      vColor = aColor * uBrightness;
      vNormal = aNormal;
      vAlpha = aAlpha;
      vDist = length(viewPos.xyz);
      vWorldPos = aPos;
    }`;
    
    // Fragment shader
    const fsSource = `#version 300 es
    precision highp float;
    
    in vec3 vColor;
    in vec3 vNormal;
    in float vAlpha;
    in float vDist;
    in vec3 vWorldPos;
    
    uniform vec3 uSunDir;
    uniform vec3 uSkyColor;
    uniform float uFogDist;
    uniform float uTime;
    uniform int uFogEnabled;
    
    out vec4 fragColor;
    
    void main() {
      // Lighting
      float ambient = 0.4;
      float diffuse = max(dot(vNormal, uSunDir), 0.0) * 0.6;
      float light = ambient + diffuse;
      
      vec3 color = vColor * light;
      
      // Fog
      if (uFogEnabled == 1) {
        float fogFactor = clamp((vDist - uFogDist * 0.5) / (uFogDist * 0.5), 0.0, 1.0);
        color = mix(color, uSkyColor, fogFactor);
      }
      
      fragColor = vec4(color, vAlpha);
    }`;
    
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('VS:', gl.getShaderInfoLog(vs));
    }
    
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('FS:', gl.getShaderInfoLog(fs));
    }
    
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program:', gl.getProgramInfoLog(program));
    }
    
    gl.useProgram(program);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    gameRef.current = {
      program,
      uProjection: gl.getUniformLocation(program, 'uProjection'),
      uView: gl.getUniformLocation(program, 'uView'),
      uTime: gl.getUniformLocation(program, 'uTime'),
      uBrightness: gl.getUniformLocation(program, 'uBrightness'),
      uSunDir: gl.getUniformLocation(program, 'uSunDir'),
      uSkyColor: gl.getUniformLocation(program, 'uSkyColor'),
      uFogDist: gl.getUniformLocation(program, 'uFogDist'),
      uFogEnabled: gl.getUniformLocation(program, 'uFogEnabled'),
      aPos: gl.getAttribLocation(program, 'aPos'),
      aColor: gl.getAttribLocation(program, 'aColor'),
      aNormal: gl.getAttribLocation(program, 'aNormal'),
      aAlpha: gl.getAttribLocation(program, 'aAlpha'),
    };
  }, []);

  // ============= MATRIX MATH =============
  const perspective = (fov: number, aspect: number, near: number, far: number): Float32Array => {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0
    ]);
  };
  
  const lookAt = (eye: number[], center: number[], up: number[]): Float32Array => {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = Math.sqrt(zx * zx + zy * zy + zz * zz);
    const z = [zx / len, zy / len, zz / len];
    const xx = up[1] * z[2] - up[2] * z[1];
    const xy = up[2] * z[0] - up[0] * z[2];
    const xz = up[0] * z[1] - up[1] * z[0];
    len = Math.sqrt(xx * xx + xy * xy + xz * xz);
    const x = [xx / len, xy / len, xz / len];
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
      -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
      -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]),
      1
    ]);
  };

  // ============= GAME LOOP =============
  const startGame = useCallback((worldName: string, seed?: number) => {
    const gameSeed = seed || Math.floor(Math.random() * 999999);
    const wd = worldDataRef.current;
    wd.seed = gameSeed;
    wd.noise = new SimplexNoise(gameSeed);
    wd.chunks.clear();
    wd.mobs = [];
    wd.playerPos = { x: 8, y: 80, z: 8 };
    wd.playerVel = { x: 0, y: 0, z: 0 };
    wd.playerRot = { x: 0, y: 0 };
    wd.time = 0;
    wd.flying = settings.gamemode === 1 || settings.gamemode === 3;
    
    // Initialize hotbar with some blocks
    setHotbar([
      { id: 1, count: 64 },
      { id: 2, count: 64 },
      { id: 4, count: 64 },
      { id: 5, count: 64 },
      { id: 17, count: 64 },
      { id: 18, count: 64 },
      { id: 25, count: 64 },
      { id: 8, count: 64 },
      { id: 6, count: 64 },
    ]);
    
    // Save world
    const newWorlds = [...worlds, { name: worldName, seed: gameSeed, created: Date.now() }];
    setWorlds(newWorlds);
    localStorage.setItem('mc_worlds', JSON.stringify(newWorlds));
    
    setScreen('loading');
    setLoadingProgress(0);
    setShowInventory(false);
    
    // Simulate loading with progress
    let progress = 0;
    const loadInterval = setInterval(() => {
      progress += 5 + Math.random() * 10;
      if (progress >= 100) {
        progress = 100;
        clearInterval(loadInterval);
        setScreen('game');
        // Init GL
        setTimeout(() => {
          initGL();
          const loop = () => {
            gameLoop(performance.now());
            requestAnimationFrame(loop);
          };
          requestAnimationFrame(loop);
        }, 100);
      }
      setLoadingProgress(Math.min(100, Math.floor(progress)));
    }, 100);
  }, [worlds, settings, initGL]);

  // ============= CHUNK MANAGEMENT =============
  const updateChunks = useCallback(() => {
    const gl = glRef.current;
    const wd = worldDataRef.current;
    if (!gl || !gameRef.current) return;
    
    const px = Math.floor(wd.playerPos.x / CHUNK_SIZE);
    const pz = Math.floor(wd.playerPos.z / CHUNK_SIZE);
    const rd = settingsRef.current.renderDistance;
    
    // Generate needed chunks
    for (let cx = px - rd; cx <= px + rd; cx++) {
      for (let cz = pz - rd; cz <= pz + rd; cz++) {
        const key = `${cx},${cz}`;
        if (!wd.chunks.has(key)) {
          const chunk = createChunk(cx, cz);
          generateChunk(chunk, wd.noise, wd.seed);
          wd.chunks.set(key, chunk);
        }
      }
    }
    
    // Build meshes for dirty chunks (limit per frame)
    let built = 0;
    for (const [key, chunk] of wd.chunks) {
      if (chunk.dirty && built < 3) {
        const verts = buildChunkMesh(chunk, wd.chunks);
        if (chunk.mesh) gl.deleteBuffer(chunk.mesh);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
        chunk.mesh = buffer;
        chunk.vertexCount = verts.length / 10;
        chunk.dirty = false;
        built++;
      }
    }
    
    // Unload far chunks
    for (const [key, chunk] of wd.chunks) {
      const dx = chunk.x - px, dz = chunk.z - pz;
      if (dx * dx + dz * dz > (rd + 4) * (rd + 4)) {
        if (chunk.mesh) gl.deleteBuffer(chunk.mesh);
        wd.chunks.delete(key);
      }
    }
  }, []);

  // ============= PHYSICS =============
  const updatePhysics = useCallback((dt: number) => {
    const wd = worldDataRef.current;
    const pos = wd.playerPos;
    const vel = wd.playerVel;
    
    // Gravity
    if (!wd.flying) {
      vel.y -= 25 * dt;
    }
    
    // Movement
    const speed = wd.keys.has(keyBindings.sprint) ? 8 : 4.5;
    const flySpeed = wd.keys.has(keyBindings.sprint) ? 20 : 10;
    const moveSpeed = wd.flying ? flySpeed : speed;
    
    const forward = [
      -Math.sin(wd.playerRot.y),
      0,
      -Math.cos(wd.playerRot.y)
    ];
    const right = [
      Math.cos(wd.playerRot.y),
      0,
      -Math.sin(wd.playerRot.y)
    ];
    
    let mx = 0, mz = 0, my = 0;
    if (wd.keys.has(keyBindings.forward)) { mx += forward[0]; mz += forward[2]; }
    if (wd.keys.has(keyBindings.back)) { mx -= forward[0]; mz -= forward[2]; }
    if (wd.keys.has(keyBindings.left)) { mx -= right[0]; mz -= right[2]; }
    if (wd.keys.has(keyBindings.right)) { mx += right[0]; mz += right[2]; }
    
    const len = Math.sqrt(mx * mx + mz * mz);
    if (len > 0) { mx /= len; mz /= len; }
    
    vel.x = mx * moveSpeed;
    vel.z = mz * moveSpeed;
    
    if (wd.flying) {
      if (wd.keys.has(keyBindings.jump)) vel.y = flySpeed;
      else if (wd.keys.has(keyBindings.sneak)) vel.y = -flySpeed;
      else vel.y = 0;
    }
    
    // Jump
    if (wd.keys.has(keyBindings.jump) && !wd.flying && isOnGround()) {
      vel.y = 8;
    }
    
    // Apply velocity with collision
    const newX = pos.x + vel.x * dt;
    const newY = pos.y + vel.y * dt;
    const newZ = pos.z + vel.z * dt;
    
    // Simple collision
    if (!isBlockSolid(newX, pos.y, pos.z) && !isBlockSolid(newX, pos.y + 1, pos.z)) {
      pos.x = newX;
    } else { vel.x = 0; }
    
    if (!isBlockSolid(pos.x, newY, pos.z) && !isBlockSolid(pos.x, newY + 1, pos.z)) {
      pos.y = newY;
    } else {
      if (vel.y < 0) {
        // Fall damage
        if (!wd.flying && settings.gamemode === 0 && vel.y < -15) {
          setHealth(h => Math.max(0, h + Math.floor(vel.y / 3)));
        }
      }
      vel.y = 0;
    }
    
    if (!isBlockSolid(pos.x, pos.y, newZ) && !isBlockSolid(pos.x, pos.y + 1, newZ)) {
      pos.z = newZ;
    } else { vel.z = 0; }
    
    // Death check
    if (pos.y < -10) {
      setHealth(0);
    }
    
    // Update mobs
    updateMobs(dt);
  }, [keyBindings, settings.gamemode]);

  const isBlockSolid = (x: number, y: number, z: number): boolean => {
    const wd = worldDataRef.current;
    const bx = Math.floor(x), by = Math.floor(y), bz = Math.floor(z);
    const cx = Math.floor(bx / CHUNK_SIZE);
    const cz = Math.floor(bz / CHUNK_SIZE);
    const key = `${cx},${cz}`;
    const chunk = wd.chunks.get(key);
    if (!chunk) return false;
    const lx = ((bx % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    const lz = ((bz % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    if (by < 0 || by >= CHUNK_HEIGHT) return false;
    const blockId = chunk.blocks[getBlockIndex(lx, by, lz)];
    const block = BLOCKS[blockId];
    return block ? block.solid : false;
  };

  const isOnGround = (): boolean => {
    const wd = worldDataRef.current;
    return isBlockSolid(wd.playerPos.x, wd.playerPos.y - 0.1, wd.playerPos.z);
  };

  // ============= BLOCK INTERACTION =============
  const getTargetBlock = (): { x: number; y: number; z: number; nx: number; ny: number; nz: number } | null => {
    const wd = worldDataRef.current;
    const dir = [
      -Math.sin(wd.playerRot.y) * Math.cos(wd.playerRot.x),
      Math.sin(wd.playerRot.x),
      -Math.cos(wd.playerRot.y) * Math.cos(wd.playerRot.x)
    ];
    
    const eye = [wd.playerPos.x, wd.playerPos.y + 1.6, wd.playerPos.z];
    const step = 0.1;
    const maxDist = 6;
    
    let px = eye[0], py = eye[1], pz = eye[2];
    let prevX = Math.floor(px), prevY = Math.floor(py), prevZ = Math.floor(pz);
    
    for (let d = 0; d < maxDist; d += step) {
      px = eye[0] + dir[0] * d;
      py = eye[1] + dir[1] * d;
      pz = eye[2] + dir[2] * d;
      
      const bx = Math.floor(px), by = Math.floor(py), bz = Math.floor(pz);
      
      if (bx !== prevX || by !== prevY || bz !== prevZ) {
        if (isBlockSolid(bx + 0.5, by + 0.5, bz + 0.5)) {
          return { x: bx, y: by, z: bz, nx: prevX, ny: prevY, nz: prevZ };
        }
        prevX = bx; prevY = by; prevZ = bz;
      }
    }
    return null;
  };

  const breakBlock = () => {
    const target = getTargetBlock();
    if (!target) return;
    const wd = worldDataRef.current;
    const { x, y, z } = target;
    const cx = Math.floor(x / CHUNK_SIZE);
    const cz = Math.floor(z / CHUNK_SIZE);
    const key = `${cx},${cz}`;
    const chunk = wd.chunks.get(key);
    if (!chunk) return;
    const lx = ((x % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    const lz = ((z % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    const idx = getBlockIndex(lx, y, lz);
    const blockId = chunk.blocks[idx];
    if (blockId === 0 || blockId === 14) return; // Can't break air or bedrock
    
    chunk.blocks[idx] = 0;
    chunk.dirty = true;
    playSound('break');
    
    // Stats
    setStats(s => ({ ...s, blocksBroken: s.blocksBroken + 1 }));
    if (blockId === 13) unlockAchievement('Алмазы!');
    if (blockId === 29) unlockAchievement('Изумрудная лихорадка');
    
    // Add to inventory
    if (settings.gamemode === 1) return; // Creative doesn't need drops
    addToInventory(blockId, 1);
  };

  const placeBlock = () => {
    const target = getTargetBlock();
    if (!target) return;
    const wd = worldDataRef.current;
    const { nx, ny, nz } = target;
    
    // Don't place inside player
    const px = Math.floor(wd.playerPos.x);
    const py = Math.floor(wd.playerPos.y);
    const pz = Math.floor(wd.playerPos.z);
    if ((nx === px && (ny === py || ny === py + 1) && nz === pz)) return;
    
    const item = hotbar[hotbarSlot];
    if (!item || item.count <= 0) return;
    
    const cx = Math.floor(nx / CHUNK_SIZE);
    const cz = Math.floor(nz / CHUNK_SIZE);
    const key = `${cx},${cz}`;
    const chunk = wd.chunks.get(key);
    if (!chunk) return;
    const lx = ((nx % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    const lz = ((nz % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
    if (ny < 0 || ny >= CHUNK_HEIGHT) return;
    
    const idx = getBlockIndex(lx, ny, lz);
    if (chunk.blocks[idx] !== 0) return;
    
    chunk.blocks[idx] = item.id;
    chunk.dirty = true;
    playSound('place');
    setStats(s => ({ ...s, blocksPlaced: s.blocksPlaced + 1 }));
    if (item.id === 18) unlockAchievement('Да будет свет!');
    
    if (settings.gamemode !== 1) {
      const newHotbar = [...hotbar];
      newHotbar[hotbarSlot] = { ...item, count: item.count - 1 };
      if (newHotbar[hotbarSlot]!.count <= 0) newHotbar[hotbarSlot] = null;
      setHotbar(newHotbar);
    }
  };

  const addToInventory = (blockId: number, count: number) => {
    setHotbar(prev => {
      const newHotbar = [...prev];
      // Try to stack
      for (let i = 0; i < newHotbar.length; i++) {
        if (newHotbar[i] && newHotbar[i]!.id === blockId && newHotbar[i]!.count < 64) {
          const add = Math.min(count, 64 - newHotbar[i]!.count);
          newHotbar[i] = { id: blockId, count: newHotbar[i]!.count + add };
          count -= add;
          if (count <= 0) return newHotbar;
        }
      }
      // Find empty slot
      for (let i = 0; i < newHotbar.length; i++) {
        if (!newHotbar[i]) {
          newHotbar[i] = { id: blockId, count };
          return newHotbar;
        }
      }
      return newHotbar;
    });
  };

  // ============= MOB AI =============
  const updateMobs = (dt: number) => {
    const wd = worldDataRef.current;
    
    // Spawn mobs
    if (wd.mobs.length < 20 && Math.random() < 0.01) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 10 + Math.random() * 20;
      const mx = wd.playerPos.x + Math.cos(angle) * dist;
      const mz = wd.playerPos.z + Math.sin(angle) * dist;
      const my = 80;
      
      const hostile = Math.random() > 0.5;
      const mobTypes = hostile ? 
        [{ type: 'zombie', color: [50, 120, 50] as [number,number,number], health: 20 },
         { type: 'skeleton', color: [200, 200, 200] as [number,number,number], health: 20 },
         { type: 'creeper', color: [50, 180, 50] as [number,number,number], health: 20 }] :
        [{ type: 'cow', color: [140, 90, 50] as [number,number,number], health: 10 },
         { type: 'pig', color: [220, 160, 140] as [number,number,number], health: 10 },
         { type: 'sheep', color: [230, 230, 230] as [number,number,number], health: 8 }];
      
      const mob = mobTypes[Math.floor(Math.random() * mobTypes.length)];
      wd.mobs.push({
        id: Date.now() + Math.random(),
        type: mob.type,
        x: mx, y: my, z: mz,
        vx: 0, vy: 0, vz: 0,
        health: mob.health,
        maxHealth: mob.health,
        hostile,
        color: mob.color,
        size: 0.8,
        aiTimer: 0,
        targetX: mx,
        targetZ: mz,
      });
    }
    
    // Update mob AI
    for (let i = wd.mobs.length - 1; i >= 0; i--) {
      const mob = wd.mobs[i];
      mob.aiTimer -= dt;
      
      const dx = wd.playerPos.x - mob.x;
      const dz = wd.playerPos.z - mob.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      
      if (mob.hostile && dist < 16) {
        // Chase player
        mob.vx = (dx / dist) * 2.5;
        mob.vz = (dz / dist) * 2.5;
        
        // Attack
        if (dist < 1.5) {
          if (settings.gamemode === 0 && mob.aiTimer <= 0) {
            setHealth(h => Math.max(0, h - 3));
            mob.aiTimer = 1;
          }
        }
      } else {
        // Wander
        if (mob.aiTimer <= 0) {
          mob.targetX = mob.x + (Math.random() - 0.5) * 10;
          mob.targetZ = mob.z + (Math.random() - 0.5) * 10;
          mob.aiTimer = 2 + Math.random() * 3;
        }
        const tdx = mob.targetX - mob.x;
        const tdz = mob.targetZ - mob.z;
        const tdist = Math.sqrt(tdx * tdx + tdz * tdz);
        if (tdist > 0.5) {
          mob.vx = (tdx / tdist) * 1.5;
          mob.vz = (tdz / tdist) * 1.5;
        } else {
          mob.vx = 0;
          mob.vz = 0;
        }
      }
      
      mob.x += mob.vx * dt;
      mob.z += mob.vz * dt;
      
      // Remove far mobs
      if (dist > 60) {
        wd.mobs.splice(i, 1);
      }
    }
  };

  // ============= RENDER =============
  const screenRef = useRef(screen);
  screenRef.current = screen;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const showInventoryRef = useRef(showInventory);
  showInventoryRef.current = showInventory;
  const showChatRef = useRef(showChat);
  showChatRef.current = showChat;
  
  const gameLoop = useCallback((time: number) => {
    const gl = glRef.current;
    const wd = worldDataRef.current;
    const game = gameRef.current;
    if (!gl || !game || screenRef.current !== 'game') return;
    if (showInventoryRef.current || showChatRef.current) return;
    
    const dt = 1 / 60;
    wd.time += dt;
    
    // Update physics
    updatePhysics(dt);
    
    // Update chunks
    updateChunks();
    
    // Render
    const canvas = canvasRef.current!;
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    // Sky color based on time
    const dayTime = (wd.time % 1200) / 1200; // 20 min cycle
    const sunAngle = dayTime * Math.PI * 2;
    const sunY = Math.sin(sunAngle);
    const isNight = sunY < 0;
    const skyR = isNight ? 0.05 : 0.4 + sunY * 0.2;
    const skyG = isNight ? 0.05 : 0.6 + sunY * 0.1;
    const skyB = isNight ? 0.15 : 0.9;
    
    gl.clearColor(skyR, skyG, skyB, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    // Camera
    const eye = [wd.playerPos.x, wd.playerPos.y + 1.6, wd.playerPos.z];
    const lookDir = [
      -Math.sin(wd.playerRot.y) * Math.cos(wd.playerRot.x),
      Math.sin(wd.playerRot.x),
      -Math.cos(wd.playerRot.y) * Math.cos(wd.playerRot.x)
    ];
    const center = [eye[0] + lookDir[0], eye[1] + lookDir[1], eye[2] + lookDir[2]];
    
    const proj = perspective(settings.fov * Math.PI / 180, canvas.width / canvas.height, 0.1, 300);
    const view = lookAt(eye, center, [0, 1, 0]);
    
    gl.uniformMatrix4fv(game.uProjection, false, proj);
    gl.uniformMatrix4fv(game.uView, false, view);
    gl.uniform1f(game.uTime, wd.time);
    gl.uniform1f(game.uBrightness, settingsRef.current.brightness);
    gl.uniform3f(game.uSunDir, Math.cos(sunAngle) * 0.5, Math.max(0.2, sunY), 0.3);
    gl.uniform3f(game.uSkyColor, skyR, skyG, skyB);
    gl.uniform1f(game.uFogDist, settingsRef.current.renderDistance * 16);
    gl.uniform1i(game.uFogEnabled, settingsRef.current.fog ? 1 : 0);
    
    // Draw chunks
    const stride = 10 * 4;
    for (const [, chunk] of wd.chunks) {
      if (!chunk.mesh || chunk.vertexCount === 0) continue;
      
      // Frustum culling (simple distance check)
      const cdx = (chunk.x + 0.5) * CHUNK_SIZE - eye[0];
      const cdz = (chunk.z + 0.5) * CHUNK_SIZE - eye[2];
      if (cdx * cdx + cdz * cdz > (settingsRef.current.renderDistance * 16 + 16) ** 2) continue;
      
      gl.bindBuffer(gl.ARRAY_BUFFER, chunk.mesh);
      gl.enableVertexAttribArray(game.aPos);
      gl.vertexAttribPointer(game.aPos, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(game.aColor);
      gl.vertexAttribPointer(game.aColor, 3, gl.FLOAT, false, stride, 12);
      gl.enableVertexAttribArray(game.aNormal);
      gl.vertexAttribPointer(game.aNormal, 3, gl.FLOAT, false, stride, 24);
      gl.enableVertexAttribArray(game.aAlpha);
      gl.vertexAttribPointer(game.aAlpha, 1, gl.FLOAT, false, stride, 36);
      
      gl.drawArrays(gl.TRIANGLES, 0, chunk.vertexCount);
    }
    
    // Draw mobs as simple cubes
    for (const mob of wd.mobs) {
      const dx = mob.x - eye[0], dz = mob.z - eye[2];
      if (dx * dx + dz * dz > 40 * 40) continue;
      drawMobCube(gl, game, mob);
    }
    
    // Update UI state
    try {
      setCoords({ x: Math.floor(wd.playerPos.x), y: Math.floor(wd.playerPos.y), z: Math.floor(wd.playerPos.z) });
      setFps(Math.round(1 / dt));
    } catch (e) {
      console.error('UI update error:', e);
    }
  }, [updateChunks, updatePhysics]);

  // Draw mob as colored cube
  const drawMobCube = (gl: WebGL2RenderingContext, game: any, mob: Mob) => {
    const s = mob.size;
    const verts = new Float32Array([
      // Simple box
      mob.x-s, mob.y, mob.z-s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      mob.x+s, mob.y, mob.z-s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      mob.x+s, mob.y, mob.z+s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      mob.x-s, mob.y, mob.z-s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      mob.x+s, mob.y, mob.z+s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      mob.x-s, mob.y, mob.z+s, mob.color[0]/255, mob.color[1]/255, mob.color[2]/255, 0,1,0, 1,
      // Top
      mob.x-s, mob.y+s*2, mob.z-s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
      mob.x+s, mob.y+s*2, mob.z-s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
      mob.x+s, mob.y+s*2, mob.z+s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
      mob.x-s, mob.y+s*2, mob.z-s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
      mob.x+s, mob.y+s*2, mob.z+s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
      mob.x-s, mob.y+s*2, mob.z+s, mob.color[0]/255*0.8, mob.color[1]/255*0.8, mob.color[2]/255*0.8, 0,1,0, 1,
    ]);
    
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    
    const stride = 10 * 4;
    gl.enableVertexAttribArray(game.aPos);
    gl.vertexAttribPointer(game.aPos, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(game.aColor);
    gl.vertexAttribPointer(game.aColor, 3, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(game.aNormal);
    gl.vertexAttribPointer(game.aNormal, 3, gl.FLOAT, false, stride, 24);
    gl.enableVertexAttribArray(game.aAlpha);
    gl.vertexAttribPointer(game.aAlpha, 1, gl.FLOAT, false, stride, 36);
    
    gl.drawArrays(gl.TRIANGLES, 0, 12);
    gl.deleteBuffer(buf);
  };

  // ============= INPUT HANDLING =============
  useEffect(() => {
    if (screen !== 'game') return;
    
    const wd = worldDataRef.current;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (showChat) return;
      
      wd.keys.add(e.code);
      
      if (e.code === 'Escape') {
        if (showInventory) { setShowInventory(false); }
        else { setScreen('pause'); }
        document.exitPointerLock();
      }
      if (e.code === 'KeyE') {
        setShowInventory(!showInventory);
        if (!showInventory) document.exitPointerLock();
        else canvasRef.current?.requestPointerLock();
      }
      if (e.code === 'F3') {
        e.preventDefault();
        setShowDebug(!showDebug);
      }
      if (e.code === 'KeyT') {
        e.preventDefault();
        setShowChat(true);
        document.exitPointerLock();
      }
      if (e.code === 'KeyF') {
        // Pick block to right hand
        const target = getTargetBlock();
        if (target) {
          const wd = worldDataRef.current;
          const cx = Math.floor(target.x / CHUNK_SIZE);
          const cz = Math.floor(target.z / CHUNK_SIZE);
          const chunk = wd.chunks.get(`${cx},${cz}`);
          if (chunk) {
            const lx = ((target.x % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
            const lz = ((target.z % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
            const blockId = chunk.blocks[getBlockIndex(lx, target.y, lz)];
            if (blockId > 0) {
              wd.rightHandItem = { id: blockId, count: 1 };
              setChatMessages(prev => [...prev.slice(-10), `§7Взял ${BLOCKS[blockId]?.name || 'блок'} в правую руку`]);
            }
          }
        }
      }
      if (e.code === 'KeyQ') {
        // Drop item
        wd.rightHandItem = null;
      }
      // Hotbar selection 1-9
      if (e.code >= 'Digit1' && e.code <= 'Digit9') {
        setHotbarSlot(parseInt(e.code.replace('Digit', '')) - 1);
      }
      // Fly toggle in creative
      if (e.code === 'Space' && settings.gamemode === 1) {
        // Double tap space for fly
      }
    };
    
    const handleKeyUp = (e: KeyboardEvent) => {
      wd.keys.delete(e.code);
    };
    
    const handleMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement === canvasRef.current) {
        wd.playerRot.y += e.movementX * 0.002 * settings.mouseSens;
        wd.playerRot.x -= e.movementY * 0.002 * settings.mouseSens;
        wd.playerRot.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, wd.playerRot.x));
      }
    };
    
    const handleMouseDown = (e: MouseEvent) => {
      if (document.pointerLockElement !== canvasRef.current) {
        canvasRef.current?.requestPointerLock();
        return;
      }
      if (e.button === 0) breakBlock();
      if (e.button === 2) placeBlock();
    };
    
    const handleContextMenu = (e: Event) => e.preventDefault();
    
    const handleWheel = (e: WheelEvent) => {
      setHotbarSlot(prev => {
        let next = prev + (e.deltaY > 0 ? 1 : -1);
        if (next < 0) next = 8;
        if (next > 8) next = 0;
        return next;
      });
    };
    
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      glRef.current?.viewport(0, 0, canvas.width, canvas.height);
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('contextmenu', handleContextMenu);
    window.addEventListener('wheel', handleWheel);
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('contextmenu', handleContextMenu);
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('resize', handleResize);
    };
  }, [screen, showInventory, showDebug, showChat, settings, hotbarSlot]);

  // ============= COMMANDS =============
  const executeCommand = (cmd: string) => {
    const parts = cmd.trim().split(' ');
    const command = parts[0].toLowerCase();
    const wd = worldDataRef.current;
    
    switch (command) {
      case '/give': {
        const id = parseInt(parts[1]);
        const count = parseInt(parts[2]) || 64;
        if (BLOCKS[id]) addToInventory(id, count);
        break;
      }
      case '/tp': {
        const x = parseFloat(parts[1]) || wd.playerPos.x;
        const y = parseFloat(parts[2]) || wd.playerPos.y;
        const z = parseFloat(parts[3]) || wd.playerPos.z;
        wd.playerPos = { x, y, z };
        break;
      }
      case '/gamemode': {
        const modes: Record<string, number> = { '0': 0, 'survival': 0, '1': 1, 'creative': 1, '2': 2, 'adventure': 2, '3': 3, 'spectator': 3 };
        const mode = modes[parts[1]?.toLowerCase() || '1'];
        if (mode !== undefined) {
          setSettings(s => ({ ...s, gamemode: mode }));
          wd.flying = mode === 1 || mode === 3;
        }
        break;
      }
      case '/time': {
        if (parts[1] === 'set') {
          const t = parseInt(parts[2]) || 0;
          wd.time = t;
        }
        break;
      }
      case '/kill': {
        setHealth(0);
        break;
      }
      case '/seed': {
        setChatMessages(prev => [...prev, `Сид: ${wd.seed}`]);
        break;
      }
      case '/weather': {
        const weathers = ['clear', 'rain', 'snow', 'storm'];
        const w = weathers.indexOf(parts[1]) >= 0 ? parts[1] : 'clear';
        wd.weather = w as any;
        break;
      }
      case '/spawn': {
        wd.playerPos = { x: 8, y: 80, z: 8 };
        wd.playerVel = { x: 0, y: 0, z: 0 };
        break;
      }
      case '/clear': {
        setHotbar(Array(9).fill(null));
        setInventory(Array(36).fill(null));
        break;
      }
      case '/fly': {
        wd.flying = !wd.flying;
        setChatMessages(prev => [...prev, `Полёт: ${wd.flying ? 'вкл' : 'выкл'}`]);
        break;
      }
      case '/summon': {
        const mobType = parts[1] || 'zombie';
        const hostile = ['zombie', 'skeleton', 'creeper', 'spider'].includes(mobType);
        wd.mobs.push({
          id: Date.now() + Math.random(),
          type: mobType,
          x: wd.playerPos.x + 3,
          y: wd.playerPos.y,
          z: wd.playerPos.z + 3,
          vx: 0, vy: 0, vz: 0,
          health: hostile ? 20 : 10,
          maxHealth: hostile ? 20 : 10,
          hostile,
          color: hostile ? [50, 150, 50] as [number,number,number] : [200, 180, 150] as [number,number,number],
          size: 0.8,
          aiTimer: 0,
          targetX: wd.playerPos.x + 3,
          targetZ: wd.playerPos.z + 3,
        });
        setChatMessages(prev => [...prev, `Призван ${mobType}`]);
        break;
      }
      case '/effect': {
        setChatMessages(prev => [...prev, 'Эффекты пока не реализованы']);
        break;
      }
      case '/enchant': {
        setChatMessages(prev => [...prev, 'Зачарование пока не реализовано']);
        break;
      }
      case '/help': {
        setChatMessages(prev => [
          ...prev,
          '§6=== Команды ===',
          '/give <id> [count] - дать предмет',
          '/tp <x> <y> <z> - телепортация',
          '/gamemode <0-3> - сменить режим',
          '/time set <ticks> - установить время',
          '/weather <clear|rain|snow|storm>',
          '/kill - убить себя',
          '/seed - показать сид',
          '/spawn - вернуться на спавн',
          '/clear - очистить инвентарь',
          '/fly - переключить полёт',
          '/summon <mob> - призвать моба',
          '/help - эта справка',
        ]);
        break;
      }
      default:
        setChatMessages(prev => [...prev, `§cНеизвестная команда: ${command}. Введите /help`]);
    }
  };

  // ============= MOD SYSTEM =============
  const loadMod = (jsonStr: string) => {
    try {
      const mod = JSON.parse(jsonStr);
      if (!mod.id || !mod.name) { alert('Мод должен иметь id и name'); return; }
      
      // Register mod blocks
      if (mod.blocks) {
        for (const block of mod.blocks) {
          const id = 100 + Object.keys(BLOCKS).length;
          BLOCKS[id] = {
            id,
            name: block.name,
            color: block.color || [128, 128, 128],
            transparent: block.transparent || false,
            solid: block.solid !== false,
            light: block.lightLevel || 0,
            hardness: block.hardness || 1,
          };
        }
      }
      
      const newMods = [...mods, mod];
      setMods(newMods);
      localStorage.setItem('minecraft_mods_cache', JSON.stringify(newMods));
      setChatMessages(prev => [...prev, `§aМод "${mod.name}" загружен!`]);
    } catch (e) {
      alert('Ошибка загрузки мода: ' + (e as Error).message);
    }
  };

  const loadTexturePack = (jsonStr: string) => {
    try {
      const pack = JSON.parse(jsonStr);
      if (!pack.name) { alert('Пак должен иметь name'); return; }
      const newPacks = [...texturePacks, pack];
      setTexturePacks(newPacks);
      localStorage.setItem('mc_texture_packs', JSON.stringify(newPacks));
    } catch (e) {
      alert('Ошибка загрузки текстур-пака: ' + (e as Error).message);
    }
  };

  const loadShaderPack = (jsonStr: string) => {
    try {
      const pack = JSON.parse(jsonStr);
      if (!pack.name) { alert('Шейдер должен иметь name'); return; }
      const newPacks = [...shaderPacks, pack];
      setShaderPacks(newPacks);
      localStorage.setItem('mc_shader_packs', JSON.stringify(newPacks));
    } catch (e) {
      alert('Ошибка загрузки шейдера: ' + (e as Error).message);
    }
  };

  // ============= ACHIEVEMENTS POPUP =============
  const [showAchievement, setShowAchievement] = useState<string | null>(null);
  useEffect(() => {
    if (achievements.length > 0) {
      const latest = achievements[achievements.length - 1];
      setShowAchievement(latest);
      setTimeout(() => setShowAchievement(null), 3000);
    }
  }, [achievements.length]);

  // ============= RENDER SCREENS =============
  if (screen === 'loading') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: '#2d1b00' }}>
        <h2 className="text-xl text-white mb-4">Генерация мира...</h2>
        <div className="w-80 h-6 bg-gray-800 border-2 border-gray-600 rounded overflow-hidden">
          <div className="h-full transition-all duration-200" style={{ width: `${loadingProgress}%`, background: 'linear-gradient(90deg, #4a4, #6c6)' }}></div>
        </div>
        <p className="text-gray-400 mt-2 text-sm">{loadingProgress}%</p>
        <p className="text-gray-500 mt-4 text-xs">Совет: Нажмите F для того чтобы взять блок в правую руку</p>
      </div>
    );
  }

  if (screen === 'menu') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'linear-gradient(180deg, #1a0a2e 0%, #16213e 50%, #0f3460 100%)' }}>
        <div className="mb-8">
          <h1 className="text-5xl font-bold text-yellow-400" style={{ textShadow: '3px 3px #000, -1px -1px #553' }}>
            MINECRAFT
          </h1>
          <p className="text-center text-gray-400 mt-2">WebGL2 Clone v1.0</p>
        </div>
        <div className="flex flex-col gap-3">
          <button className="mc-btn" onClick={() => setScreen('worlds')}>Одиночная игра</button>
          <button className="mc-btn" onClick={() => setScreen('servers')}>Сетевая игра</button>
          <button className="mc-btn" onClick={() => setScreen('settings')}>Настройки</button>
        </div>
        <p className="mt-8 text-gray-500 text-xs">Copyright Mojang AB. Fan-made clone.</p>
      </div>
    );
  }

  if (screen === 'worlds') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'linear-gradient(180deg, #1a0a2e 0%, #16213e 100%)' }}>
        <h2 className="text-2xl text-white mb-6">Выбор мира</h2>
        <div className="mc-panel w-96 max-h-64 overflow-y-auto mb-4">
          {worlds.length === 0 && <p className="text-gray-600 text-center p-4">Нет сохранённых миров</p>}
          {worlds.map((w, i) => (
            <div key={i} className="p-2 border-b border-gray-500 flex justify-between items-center">
              <div>
                <div className="text-black font-bold">{w.name}</div>
                <div className="text-gray-600 text-xs">Сид: {w.seed}</div>
              </div>
              <div className="flex gap-1">
                <button className="mc-btn text-xs min-w-0 px-2" onClick={() => startGame(w.name, w.seed)}>Играть</button>
                <button className="mc-btn text-xs min-w-0 px-2" onClick={() => {
                  const nw = worlds.filter((_, idx) => idx !== i);
                  setWorlds(nw);
                  localStorage.setItem('mc_worlds', JSON.stringify(nw));
                }}>✕</button>
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-3">
          <button className="mc-btn" onClick={() => setScreen('create')}>Создать мир</button>
          <button className="mc-btn" onClick={() => setScreen('menu')}>Назад</button>
        </div>
      </div>
    );
  }

  if (screen === 'create') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'linear-gradient(180deg, #1a0a2e 0%, #16213e 100%)' }}>
        <h2 className="text-2xl text-white mb-6">Создание мира</h2>
        <div className="mc-panel w-96">
          <div className="mb-3">
            <label className="text-black text-sm">Название мира:</label>
            <input className="mc-text-input w-full mt-1" id="worldName" defaultValue="Мой мир" />
          </div>
          <div className="mb-3">
            <label className="text-black text-sm">Сид (оставьте пустым для случайного):</label>
            <input className="mc-text-input w-full mt-1" id="worldSeed" type="number" placeholder="12345" />
          </div>
          <div className="mb-3">
            <label className="text-black text-sm">Режим игры:</label>
            <select className="mc-text-input w-full mt-1" id="worldGamemode">
              <option value="0">Выживание</option>
              <option value="1">Креатив</option>
              <option value="2">Приключение</option>
              <option value="3">Наблюдатель</option>
            </select>
          </div>
        </div>
        <div className="flex gap-3 mt-4">
          <button className="mc-btn" onClick={() => {
            const name = (document.getElementById('worldName') as HTMLInputElement)?.value || 'Мой мир';
            const seed = parseInt((document.getElementById('worldSeed') as HTMLInputElement)?.value) || undefined;
            const gm = parseInt((document.getElementById('worldGamemode') as HTMLSelectElement)?.value) || 0;
            setSettings(s => ({ ...s, gamemode: gm }));
            startGame(name, seed);
          }}>Создать</button>
          <button className="mc-btn" onClick={() => setScreen('worlds')}>Назад</button>
        </div>
      </div>
    );
  }

  if (screen === 'servers') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'linear-gradient(180deg, #1a0a2e 0%, #16213e 100%)' }}>
        <h2 className="text-2xl text-white mb-6">Сетевая игра</h2>
        <div className="mc-panel w-96 max-h-64 overflow-y-auto mb-4">
          {servers.length === 0 && <p className="text-gray-600 text-center p-4">Нет серверов</p>}
          {servers.map((s, i) => (
            <div key={i} className="p-2 border-b border-gray-500 flex justify-between">
              <div>
                <div className="text-black font-bold">{s.name}</div>
                <div className="text-gray-600 text-xs">{s.address}</div>
              </div>
              <button className="mc-btn text-xs min-w-0 px-2" onClick={() => {
                setChatMessages([`Подключение к ${s.name}...`]);
                setChatMessages(prev => [...prev, 'Сервер недоступен (демо-режим)']);
              }}>⚡</button>
            </div>
          ))}
        </div>
        <div className="flex gap-3">
          <button className="mc-btn" onClick={() => {
            const name = prompt('Имя сервера:');
            const addr = prompt('Адрес сервера:');
            if (name && addr) {
              const ns = [...servers, { name, address: addr }];
              setServers(ns);
              localStorage.setItem('mc_servers', JSON.stringify(ns));
            }
          }}>Добавить</button>
          <button className="mc-btn" onClick={() => setScreen('menu')}>Назад</button>
        </div>
      </div>
    );
  }

  if (screen === 'settings') {
    const tabs = ['Геймплей', 'Управление', 'Графика', 'Ресурсы'];
    return (
      <div className="w-full h-full flex flex-col items-center justify-center overflow-y-auto py-8" style={{ background: 'linear-gradient(180deg, #1a0a2e 0%, #16213e 100%)' }}>
        <h2 className="text-2xl text-white mb-4">Настройки</h2>
        <div className="flex gap-1 mb-4">
          {tabs.map((t, i) => (
            <button key={i} className={`tab-btn ${settingsTab === i ? 'active' : ''}`} onClick={() => setSettingsTab(i)}>{t}</button>
          ))}
        </div>
        <div className="mc-panel w-[500px] max-h-[60vh] overflow-y-auto">
          {settingsTab === 0 && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-black">Сложность:</span>
                <select className="mc-text-input text-sm" value={settings.difficulty} onChange={e => setSettings(s => ({...s, difficulty: parseInt(e.target.value)}))}>
                  <option value={0}>Мирная</option><option value={1}>Лёгкая</option><option value={2}>Нормальная</option><option value={3}>Сложная</option>
                </select>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Режим игры:</span>
                <select className="mc-text-input text-sm" value={settings.gamemode} onChange={e => setSettings(s => ({...s, gamemode: parseInt(e.target.value)}))}>
                  <option value={0}>Выживание</option><option value={1}>Креатив</option><option value={2}>Приключение</option><option value={3}>Наблюдатель</option>
                </select>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Показывать координаты:</span>
                <input type="checkbox" checked={settings.showCoords} onChange={e => setSettings(s => ({...s, showCoords: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Показывать FPS:</span>
                <input type="checkbox" checked={settings.showFps} onChange={e => setSettings(s => ({...s, showFps: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Автосохранение:</span>
                <input type="checkbox" checked={settings.autoSave} onChange={e => setSettings(s => ({...s, autoSave: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Язык:</span>
                <select className="mc-text-input text-sm" value={settings.language} onChange={e => setSettings(s => ({...s, language: e.target.value}))}>
                  <option value="ru">Русский</option><option value="en">English</option>
                </select>
              </div>
              <div>
                <span className="text-black text-sm">Громкость: {Math.round(settings.volume * 100)}%</span>
                <input type="range" className="mc-slider" min="0" max="1" step="0.1" value={settings.volume} onChange={e => setSettings(s => ({...s, volume: parseFloat(e.target.value)}))} />
              </div>
            </div>
          )}
          {settingsTab === 1 && (
            <div className="space-y-2">
              <p className="text-black text-sm mb-2">Назначение клавиш (клик для переназначения):</p>
              {Object.entries(keyBindings).map(([action, key]) => (
                <div key={action} className="flex justify-between items-center">
                  <span className="text-black text-sm">{action}:</span>
                  <button className="mc-btn text-xs min-w-0 px-3 py-1" onClick={() => {
                    const newKey = prompt(`Нажмите клавишу для "${action}":`, key);
                    if (newKey) {
                      const nk = { ...keyBindings, [action]: newKey };
                      setKeyBindings(nk);
                      localStorage.setItem('mc_keybindings', JSON.stringify(nk));
                    }
                  }}>{key}</button>
                </div>
              ))}
              <div className="mt-3">
                <span className="text-black text-sm">Чувствительность мыши: {settings.mouseSens.toFixed(1)}</span>
                <input type="range" className="mc-slider" min="0.1" max="2" step="0.1" value={settings.mouseSens} onChange={e => setSettings(s => ({...s, mouseSens: parseFloat(e.target.value)}))} />
              </div>
              <button className="mc-btn w-full mt-2" onClick={() => {
                const def = { forward: 'KeyW', back: 'KeyS', left: 'KeyA', right: 'KeyD', jump: 'Space', sneak: 'ShiftLeft', sprint: 'ControlLeft', attack: 'Mouse0', use: 'Mouse2', inventory: 'KeyE', drop: 'KeyQ', chat: 'KeyT', command: 'Slash', debug: 'F3', thirdPerson: 'F5', fullscreen: 'F11', pause: 'Escape', pickBlock: 'KeyF' };
                setKeyBindings(def);
                localStorage.setItem('mc_keybindings', JSON.stringify(def));
              }}>Сбросить</button>
            </div>
          )}
          {settingsTab === 2 && (
            <div className="space-y-3">
              <div>
                <span className="text-black text-sm">Дальность прорисовки: {settings.renderDistance} чанков</span>
                <input type="range" className="mc-slider" min="2" max="32" value={settings.renderDistance} onChange={e => setSettings(s => ({...s, renderDistance: parseInt(e.target.value)}))} />
              </div>
              <div>
                <span className="text-black text-sm">FOV: {settings.fov}°</span>
                <input type="range" className="mc-slider" min="30" max="110" value={settings.fov} onChange={e => setSettings(s => ({...s, fov: parseInt(e.target.value)}))} />
              </div>
              <div>
                <span className="text-black text-sm">Яркость: {(settings.brightness * 100).toFixed(0)}%</span>
                <input type="range" className="mc-slider" min="0.2" max="2" step="0.1" value={settings.brightness} onChange={e => setSettings(s => ({...s, brightness: parseFloat(e.target.value)}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Облака:</span>
                <input type="checkbox" checked={settings.clouds} onChange={e => setSettings(s => ({...s, clouds: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Частицы:</span>
                <input type="checkbox" checked={settings.particles} onChange={e => setSettings(s => ({...s, particles: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Плавное освещение:</span>
                <input type="checkbox" checked={settings.smoothLighting} onChange={e => setSettings(s => ({...s, smoothLighting: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Ambient Occlusion:</span>
                <input type="checkbox" checked={settings.ao} onChange={e => setSettings(s => ({...s, ao: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Тени:</span>
                <input type="checkbox" checked={settings.shadows} onChange={e => setSettings(s => ({...s, shadows: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">Туман:</span>
                <input type="checkbox" checked={settings.fog} onChange={e => setSettings(s => ({...s, fog: e.target.checked}))} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-black">VSync:</span>
                <input type="checkbox" checked={settings.vsync} onChange={e => setSettings(s => ({...s, vsync: e.target.checked}))} />
              </div>
              <div>
                <span className="text-black text-sm">Макс. FPS:</span>
                <select className="mc-text-input text-sm ml-2" value={settings.maxFps} onChange={e => setSettings(s => ({...s, maxFps: parseInt(e.target.value)}))}>
                  <option value={30}>30</option><option value={60}>60</option><option value={120}>120</option><option value={144}>144</option><option value={0}>Без лимита</option>
                </select>
              </div>
            </div>
          )}
          {settingsTab === 3 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-black font-bold mb-2">Текстур-паки ({texturePacks.length})</h3>
                <div className="space-y-1 mb-2">
                  {texturePacks.map((p, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-200 p-1">
                      <span className="text-black text-sm">{p.name} v{p.version || '1.0'}</span>
                      <button className="text-red-600 text-xs" onClick={() => {
                        const np = texturePacks.filter((_, idx) => idx !== i);
                        setTexturePacks(np);
                        localStorage.setItem('mc_texture_packs', JSON.stringify(np));
                      }}>Удалить</button>
                    </div>
                  ))}
                </div>
                <button className="mc-btn text-xs" onClick={() => {
                  const json = prompt('Вставьте JSON текстур-пака:');
                  if (json) loadTexturePack(json);
                }}>Загрузить пак</button>
              </div>
              <div>
                <h3 className="text-black font-bold mb-2">Шейдер-паки ({shaderPacks.length})</h3>
                <div className="space-y-1 mb-2">
                  {shaderPacks.map((p, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-200 p-1">
                      <span className="text-black text-sm">{p.name}</span>
                      <button className="text-red-600 text-xs" onClick={() => {
                        const np = shaderPacks.filter((_, idx) => idx !== i);
                        setShaderPacks(np);
                        localStorage.setItem('mc_shader_packs', JSON.stringify(np));
                      }}>Удалить</button>
                    </div>
                  ))}
                </div>
                <button className="mc-btn text-xs" onClick={() => {
                  const json = prompt('Вставьте JSON шейдер-пака:');
                  if (json) loadShaderPack(json);
                }}>Загрузить шейдер</button>
              </div>
              <div>
                <h3 className="text-black font-bold mb-2">Моды ({mods.length})</h3>
                <div className="space-y-1 mb-2">
                  {mods.map((m, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-200 p-1">
                      <span className="text-black text-sm">{m.name} v{m.version}</span>
                      <button className="text-red-600 text-xs" onClick={() => {
                        const nm = mods.filter((_, idx) => idx !== i);
                        setMods(nm);
                        localStorage.setItem('minecraft_mods_cache', JSON.stringify(nm));
                      }}>Удалить</button>
                    </div>
                  ))}
                </div>
                <button className="mc-btn text-xs" onClick={() => {
                  const json = prompt('Вставьте JSON мода:');
                  if (json) loadMod(json);
                }}>Добавить мод</button>
              </div>
            </div>
          )}
        </div>
        <button className="mc-btn mt-4" onClick={() => setScreen('menu')}>Назад</button>
      </div>
    );
  }

  if (screen === 'pause') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'rgba(0,0,0,0.7)' }}>
        <h2 className="text-xl text-white mb-6">Игра приостановлена</h2>
        <div className="flex flex-col gap-3">
          <button className="mc-btn" onClick={() => { setScreen('game'); canvasRef.current?.requestPointerLock(); }}>Продолжить</button>
          <button className="mc-btn" onClick={() => setScreen('settings')}>Настройки</button>
          <button className="mc-btn" onClick={() => {
            // Manual save
            const wd = worldDataRef.current;
            const saveData = {
              playerPos: wd.playerPos,
              playerRot: wd.playerRot,
              health,
              hunger,
              hotbar,
              inventory,
              time: wd.time,
              seed: wd.seed,
              stats,
            };
            localStorage.setItem('mc_autosave', JSON.stringify(saveData));
            setChatMessages(prev => [...prev, '§aМир сохранён!']);
          }}>Сохранить</button>
          <button className="mc-btn" onClick={() => setScreen('menu')}>Выйти в меню</button>
        </div>
        <div className="mc-panel mt-4 p-3 text-xs" style={{ minWidth: '300px' }}>
          <h3 className="text-black font-bold mb-2">Статистика:</h3>
          <div className="text-black">Блоков сломано: {stats.blocksBroken}</div>
          <div className="text-black">Блоков поставлено: {stats.blocksPlaced}</div>
          <div className="text-black">Мобов убито: {stats.mobsKilled}</div>
          <div className="text-black">Прыжков: {stats.jumps}</div>
          <h3 className="text-black font-bold mt-2 mb-1">Достижения ({achievements.length}):</h3>
          {achievements.length === 0 && <div className="text-gray-600">Пока нет достижений</div>}
          {achievements.map((a, i) => <div key={i} className="text-black">🏆 {a}</div>)}
        </div>
      </div>
    );
  }

  if (screen === 'death') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: 'rgba(150,0,0,0.7)' }}>
        <h2 className="text-3xl text-white mb-6">Вы погибли!</h2>
        <div className="flex flex-col gap-3">
          <button className="mc-btn" onClick={() => {
            const wd = worldDataRef.current;
            wd.playerPos = { x: 8, y: 80, z: 8 };
            wd.playerVel = { x: 0, y: 0, z: 0 };
            setHealth(20);
            setScreen('game');
            canvasRef.current?.requestPointerLock();
          }}>Возродиться</button>
          <button className="mc-btn" onClick={() => { document.exitPointerLock(); setScreen('menu'); }}>Выйти в меню</button>
        </div>
      </div>
    );
  }

  // GAME SCREEN
  return (
    <div className="w-full h-full relative">
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* Crosshair */}
      {!showInventory && <div className="crosshair" />}
      
      {/* Hotbar */}
      <div className="hotbar">
        {hotbar.map((item, i) => (
          <div key={i} className={`mc-slot ${i === hotbarSlot ? 'selected' : ''}`} onClick={() => setHotbarSlot(i)}>
            {item && (
              <div className="w-6 h-6 rounded-sm" style={{ backgroundColor: `rgb(${BLOCKS[item.id]?.color?.join(',') || '128,128,128'})` }} title={BLOCKS[item.id]?.name}>
                {item.count > 1 && <span className="absolute bottom-0 right-0 text-white text-xs" style={{ textShadow: '1px 1px #000' }}>{item.count}</span>}
              </div>
            )}
            <span className="absolute top-0 left-0 text-gray-400 text-[8px]">{i + 1}</span>
          </div>
        ))}
      </div>
      
      {/* Health */}
      <div className="hearts">
        {Array(10).fill(0).map((_, i) => (
          <span key={i} style={{ color: i < health / 2 ? '#e00' : '#555', fontSize: '14px' }}>♥</span>
        ))}
      </div>
      
      {/* Hunger */}
      <div className="hunger">
        {Array(10).fill(0).map((_, i) => (
          <span key={i} style={{ color: i < hunger / 2 ? '#a60' : '#555', fontSize: '14px' }}>✦</span>
        ))}
      </div>
      
      {/* Right hand item indicator */}
      {worldDataRef.current.rightHandItem && (
        <div className="fixed top-4 right-4 bg-black/50 p-2 text-white text-xs z-50">
          Правая рука: {BLOCKS[worldDataRef.current.rightHandItem.id]?.name}
        </div>
      )}
      
      {/* Debug overlay */}
      {showDebug && (
        <div className="debug-overlay">
          <div>Minecraft Clone v1.0 (WebGL2)</div>
          <div>FPS: {fps}</div>
          <div>XYZ: {coords.x} / {coords.y} / {coords.z}</div>
          <div>Chunk: {Math.floor(coords.x / 16)}, {Math.floor(coords.z / 16)}</div>
          <div>Chunks loaded: {worldDataRef.current.chunks.size}</div>
          <div>Mobs: {worldDataRef.current.mobs.length}</div>
          <div>Facing: {['south','west','north','east'][Math.floor(((worldDataRef.current.playerRot.y / Math.PI * 180) % 360 + 360) % 360 / 90) % 4]}</div>
          <div>Fly: {worldDataRef.current.flying ? 'ON' : 'OFF'}</div>
          <div>Gamemode: {['Survival','Creative','Adventure','Spectator'][settings.gamemode]}</div>
          <div>Seed: {worldDataRef.current.seed}</div>
        </div>
      )}
      
      {/* Chat */}
      {showChat && (
        <div className="chat-box">
          <div className="chat-messages">
            {chatMessages.slice(-10).map((msg, i) => (
              <div key={i} className="chat-msg">{msg}</div>
            ))}
          </div>
          <input
            className="mc-text-input w-full"
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                if (chatInput.startsWith('/')) {
                  executeCommand(chatInput);
                } else if (chatInput) {
                  setChatMessages(prev => [...prev, `<Player> ${chatInput}`]);
                }
                setChatInput('');
                setShowChat(false);
                canvasRef.current?.requestPointerLock();
              }
              if (e.key === 'Escape') {
                setShowChat(false);
                setChatInput('');
                canvasRef.current?.requestPointerLock();
              }
            }}
            autoFocus
            placeholder="Введите сообщение или /команду..."
          />
        </div>
      )}
      
      {/* Inventory */}
      {showInventory && (
        <div className="inventory-screen" onClick={e => { if (e.target === e.currentTarget) { setShowInventory(false); canvasRef.current?.requestPointerLock(); } }}>
          <div className="mc-panel p-4" style={{ minWidth: '400px' }}>
            <h3 className="text-black font-bold mb-3 text-center">Инвентарь</h3>
            
            {/* Crafting 2x2 */}
            <div className="mb-3">
              <span className="text-black text-xs">Крафт 2×2:</span>
              <div className="grid grid-cols-2 gap-1 mt-1">
                {Array(4).fill(null).map((_, i) => (
                  <div key={i} className="mc-slot" style={{ width: '32px', height: '32px' }}></div>
                ))}
              </div>
            </div>
            
            {/* Main inventory */}
            <div className="mb-3">
              <span className="text-black text-xs">Инвентарь:</span>
              <div className="grid grid-cols-9 gap-1 mt-1">
                {inventory.map((item, i) => (
                  <div key={i} className="mc-slot" style={{ width: '32px', height: '32px' }}
                    onClick={() => {
                      if (item) {
                        // Move to hotbar
                        const newInv = [...inventory];
                        const newHotbar = [...hotbar];
                        const emptySlot = newHotbar.findIndex(h => !h);
                        if (emptySlot >= 0) {
                          newHotbar[emptySlot] = item;
                          newInv[i] = null;
                          setInventory(newInv);
                          setHotbar(newHotbar);
                        }
                      }
                    }}>
                    {item && (
                      <div className="w-5 h-5 rounded-sm" style={{ backgroundColor: `rgb(${BLOCKS[item.id]?.color?.join(',') || '128,128,128'})` }}>
                        {item.count > 1 && <span className="text-white text-[8px]" style={{ textShadow: '1px 1px #000' }}>{item.count}</span>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Hotbar */}
            <div>
              <span className="text-black text-xs">Хотбар:</span>
              <div className="grid grid-cols-9 gap-1 mt-1">
                {hotbar.map((item, i) => (
                  <div key={i} className={`mc-slot ${i === hotbarSlot ? 'selected' : ''}`} style={{ width: '32px', height: '32px' }}
                    onClick={() => {
                      // F: take to right hand
                      if (item) {
                        worldDataRef.current.rightHandItem = { ...item };
                      }
                    }}>
                    {item && (
                      <div className="w-5 h-5 rounded-sm" style={{ backgroundColor: `rgb(${BLOCKS[item.id]?.color?.join(',') || '128,128,128'})` }}>
                        {item.count > 1 && <span className="text-white text-[8px]" style={{ textShadow: '1px 1px #000' }}>{item.count}</span>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Crafting recipes */}
            <div className="mt-3">
              <span className="text-black text-xs">Рецепты крафта:</span>
              <div className="max-h-32 overflow-y-auto mt-1">
                {recipes.map((recipe, i) => (
                  <div key={i} className="flex justify-between items-center bg-gray-200 p-1 mb-1 text-xs">
                    <span className="text-black">{recipe.name}</span>
                    <button 
                      className={`mc-btn text-xs min-w-0 px-2 py-0 ${canCraft(recipe) ? '' : 'opacity-50'}`}
                      onClick={() => craft(recipe)}
                      disabled={!canCraft(recipe)}
                    >
                      Крафт
                    </button>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Creative tabs */}
            {settings.gamemode === 1 && (
              <div className="mt-3">
                <span className="text-black text-xs">Креативные вкладки:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {[1,2,3,4,5,6,8,17,18,25,32,33,34,35,36].map(id => (
                    <div key={id} className="mc-slot cursor-pointer" style={{ width: '28px', height: '28px' }}
                      onClick={() => addToInventory(id, 64)}
                      title={BLOCKS[id]?.name}>
                      <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: `rgb(${BLOCKS[id]?.color?.join(',') || '128,128,128'})` }}></div>
                    </div>
                  ))}
                  {/* Mod items */}
                  {mods.map((mod, mi) => mod.blocks?.map((b: any, bi: number) => (
                    <div key={`${mi}-${bi}`} className="mc-slot cursor-pointer" style={{ width: '28px', height: '28px' }}
                      onClick={() => addToInventory(100 + bi, 64)}
                      title={b.name}>
                      <div className="w-4 h-4 rounded-sm" style={{ backgroundColor: `rgb(${(b.color || [128,128,128]).join(',')})` }}></div>
                    </div>
                  )))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Achievement popup */}
      {showAchievement && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 bg-purple-900/90 border-2 border-purple-400 px-4 py-2 rounded z-50 animate-pulse">
          <div className="text-yellow-300 text-sm font-bold">🏆 Достижение!</div>
          <div className="text-white text-xs">{showAchievement}</div>
        </div>
      )}
      
      {/* Mobile controls */}
      <div className="fixed bottom-20 left-4 z-50 md:hidden">
        <div className="grid grid-cols-3 gap-1">
          <div></div>
          <button className="mc-btn text-xs min-w-0 w-10 h-10 p-0" onTouchStart={() => worldDataRef.current.keys.add(keyBindings.forward)} onTouchEnd={() => worldDataRef.current.keys.delete(keyBindings.forward)}>▲</button>
          <div></div>
          <button className="mc-btn text-xs min-w-0 w-10 h-10 p-0" onTouchStart={() => worldDataRef.current.keys.add(keyBindings.left)} onTouchEnd={() => worldDataRef.current.keys.delete(keyBindings.left)}>◄</button>
          <button className="mc-btn text-xs min-w-0 w-10 h-10 p-0" onTouchStart={() => worldDataRef.current.keys.add(keyBindings.jump)} onTouchEnd={() => worldDataRef.current.keys.delete(keyBindings.jump)}>⬆</button>
          <button className="mc-btn text-xs min-w-0 w-10 h-10 p-0" onTouchStart={() => worldDataRef.current.keys.add(keyBindings.right)} onTouchEnd={() => worldDataRef.current.keys.delete(keyBindings.right)}>►</button>
          <div></div>
          <button className="mc-btn text-xs min-w-0 w-10 h-10 p-0" onTouchStart={() => worldDataRef.current.keys.add(keyBindings.back)} onTouchEnd={() => worldDataRef.current.keys.delete(keyBindings.back)}>▼</button>
          <div></div>
        </div>
      </div>
      <div className="fixed bottom-20 right-4 z-50 md:hidden flex gap-2">
        <button className="mc-btn text-xs min-w-0 w-12 h-12 p-0" onTouchStart={() => breakBlock()}>⛏</button>
        <button className="mc-btn text-xs min-w-0 w-12 h-12 p-0" onTouchStart={() => placeBlock()}>📦</button>
      </div>
      
      {/* Death check */}
      {health <= 0 && screen === 'game' && (
        <div className="fixed inset-0 flex flex-col items-center justify-center z-50" style={{ background: 'rgba(150,0,0,0.8)' }}>
          <h2 className="text-3xl text-white mb-6">Вы погибли!</h2>
          <button className="mc-btn" onClick={() => {
            const wd = worldDataRef.current;
            wd.playerPos = { x: 8, y: 80, z: 8 };
            wd.playerVel = { x: 0, y: 0, z: 0 };
            setHealth(20);
            canvasRef.current?.requestPointerLock();
          }}>Возродиться</button>
        </div>
      )}
    </div>
  );
}
