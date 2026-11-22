import type { FeatureColumn } from "./kumo/preprocess.ts";

type Column = (row: number) => number;

export function designMatrix(features: FeatureColumn[], trainCount: number, rowCount: number) {
  const columns: Column[] = [() => 1];
  for (const feature of features) {
    if (feature.kind === "numerical") {
      const values = Array.from(feature.values).slice(0, trainCount);
      const mean = values.reduce((sum, value) => sum + value, 0) / trainCount;
      const scale = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / trainCount);
      if (scale > 0) columns.push((row) => (feature.values[row] - mean) / scale);
    } else {
      const levels = [...new Set(feature.values.slice(0, trainCount))].sort();
      for (const level of levels.slice(1)) columns.push((row) => Number(feature.values[row] === level));
    }
  }
  return Array.from({ length: rowCount }, (_, row) => Float64Array.from(columns, (column) => column(row)));
}

function dot(vector: Float64Array, weights: Float64Array) {
  let value = 0;
  for (let index = 0; index < weights.length; index++) value += vector[index] * weights[index];
  return value;
}

// Solve a symmetric positive-definite system with Cholesky factorization.
function solve(matrix: Float64Array[], target: Float64Array) {
  const size = target.length;
  const out = target.slice();
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let value = matrix[i][j];
      for (let k = 0; k < j; k++) value -= matrix[i][k] * matrix[j][k];
      matrix[i][j] = i === j ? Math.sqrt(value) : value / matrix[j][j];
    }
    for (let j = 0; j < i; j++) out[i] -= matrix[i][j] * out[j];
    out[i] /= matrix[i][i];
  }
  for (let i = size - 1; i >= 0; i--) {
    for (let j = i + 1; j < size; j++) out[i] -= matrix[j][i] * out[j];
    out[i] /= matrix[i][i];
  }
  return out;
}

function normalSystem(
  matrix: Float64Array[],
  target: ArrayLike<number>,
  ridge: number,
  terms: (actual: number, vector: Float64Array) => [residual: number, weight: number],
) {
  const size = matrix[0].length;
  const system = Array.from({ length: size }, () => new Float64Array(size));
  const right = new Float64Array(size);
  for (let row = 0; row < target.length; row++) {
    const vector = matrix[row];
    const [residual, weight] = terms(target[row], vector);
    for (let i = 0; i < size; i++) {
      right[i] += vector[i] * residual;
      const weighted = vector[i] * weight;
      for (let j = 0; j <= i; j++) system[i][j] += weighted * vector[j];
    }
  }
  for (let i = 1; i < size; i++) system[i][i] += ridge;
  return { system, right };
}

export function linearRegression(matrix: Float64Array[], target: ArrayLike<number>) {
  const { system, right } = normalSystem(matrix, target, 1e-8 * target.length, (value) => [value, 1]);
  const weights = solve(system, right);
  return Float64Array.from(matrix.slice(target.length), (row) => dot(row, weights));
}

const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

function binaryLogistic(matrix: Float64Array[], target: Float64Array) {
  const weights = new Float64Array(matrix[0].length);
  const ridge = 1e-4 * target.length;
  for (let iteration = 0; iteration < 35; iteration++) {
    const { system, right } = normalSystem(matrix, target, ridge, (actual, vector) => {
      const p = sigmoid(dot(vector, weights));
      return [actual - p, p * (1 - p)];
    });
    for (let i = 1; i < weights.length; i++) right[i] -= weights[i] * ridge;
    const step = solve(system, right);
    let maxChange = 0;
    for (let i = 0; i < weights.length; i++) {
      weights[i] += step[i];
      maxChange = Math.max(maxChange, Math.abs(step[i]));
    }
    if (maxChange < 1e-7) break;
  }
  return Float64Array.from(matrix.slice(target.length), (row) => sigmoid(dot(row, weights)));
}

export function logisticRegression(matrix: Float64Array[], codes: Int32Array, classCount: number) {
  const out = new Float64Array((matrix.length - codes.length) * classCount);
  if (classCount === 2) {
    const positive = binaryLogistic(matrix, Float64Array.from(codes));
    positive.forEach((value, row) => {
      out[2 * row] = 1 - value;
      out[2 * row + 1] = value;
    });
    return out;
  }
  const estimates = Array.from({ length: classCount }, (_, code) =>
    binaryLogistic(matrix, Float64Array.from(codes, (label) => Number(label === code))));
  for (let row = 0; row < matrix.length - codes.length; row++) {
    const total = estimates.reduce((sum, column) => sum + column[row], 0);
    for (let code = 0; code < classCount; code++) out[row * classCount + code] = estimates[code][row] / total;
  }
  return out;
}
