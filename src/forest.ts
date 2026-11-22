import { randomSource } from "./data.ts";

const treeCount = 200;
const maxDepth = 12;
const minLeafSize = 3;

type Tree = { prediction: number } | { column: number; threshold: number; left: Tree; right: Tree };
type NodeStats = {
  prediction: number;
  pure: boolean;
  scorePrefix: () => (value: number, leftCount: number) => number;
};

function classStats(target: ArrayLike<number>, rows: number[], classCount: number): NodeStats {
  const counts = Array<number>(classCount).fill(0);
  for (const row of rows) counts[target[row]]++;
  const prediction = counts.indexOf(Math.max(...counts));
  const squares = counts.reduce((sum, count) => sum + count ** 2, 0);
  const parentScore = squares / rows.length;
  return {
    prediction,
    pure: counts[prediction] === rows.length,
    scorePrefix() {
      const left = Array<number>(classCount).fill(0);
      const right = counts.slice();
      let leftSquares = 0;
      let rightSquares = squares;
      return (code, size) => {
        leftSquares += 2 * left[code] + 1;
        rightSquares -= 2 * right[code] - 1;
        left[code]++;
        right[code]--;
        // Weighted Gini reduction multiplied by the number of rows in this node.
        return leftSquares / size + rightSquares / (rows.length - size) - parentScore;
      };
    },
  };
}

function regressionStats(target: ArrayLike<number>, rows: number[]): NodeStats {
  const prediction = rows.reduce((sum, row) => sum + target[row], 0) / rows.length;
  return {
    prediction,
    pure: rows.every((row) => target[row] === target[rows[0]]),
    scorePrefix() {
      let leftSum = 0;
      return (value, size) => {
        leftSum += value - prediction;
        // The centered right sum is negative leftSum. This is the reduction in SSE.
        return leftSum ** 2 * (1 / size + 1 / (rows.length - size));
      };
    },
  };
}

function fitTree(
  matrix: Float64Array[],
  target: ArrayLike<number>,
  rows: number[],
  columns: number[],
  depth: number,
  classCount: number,
): Tree {
  const stats = classCount ? classStats(target, rows, classCount) : regressionStats(target, rows);
  if (depth >= maxDepth || rows.length < 2 * minLeafSize || stats.pure) return { prediction: stats.prediction };

  let bestGain = 0;
  let split: { column: number; threshold: number; sorted: number[]; size: number } | null = null;
  for (const column of columns) {
    const sorted = rows.slice().sort((a, b) => matrix[a][column] - matrix[b][column]);
    const gainAt = stats.scorePrefix();
    for (let size = 1; size <= sorted.length - minLeafSize; size++) {
      const gain = gainAt(target[sorted[size - 1]], size);
      const lower = matrix[sorted[size - 1]][column];
      const upper = matrix[sorted[size]][column];
      if (size >= minLeafSize && lower < upper && gain > bestGain) {
        const midpoint = lower + (upper - lower) / 2;
        const threshold = midpoint < upper ? midpoint : lower;
        split = { column, threshold, sorted, size };
        bestGain = gain;
      }
    }
  }
  if (!split) return { prediction: stats.prediction };
  const { column, threshold, sorted, size } = split;
  return {
    column,
    threshold,
    left: fitTree(matrix, target, sorted.slice(0, size), columns, depth + 1, classCount),
    right: fitTree(matrix, target, sorted.slice(size), columns, depth + 1, classCount),
  };
}

function predict(tree: Tree, row: Float64Array): number {
  while ("column" in tree) tree = row[tree.column] <= tree.threshold ? tree.left : tree.right;
  return tree.prediction;
}

function predictForest(matrix: Float64Array[], target: ArrayLike<number>, classCount: number) {
  if (!target.length) throw new Error("Random Forest needs labeled training rows.");
  const random = randomSource(20817);
  const trainCount = target.length;
  const featureCount = matrix[0].length - 1;
  // Column zero is the regression intercept. Trees split on encoded predictors.
  const allColumns = Array.from({ length: featureCount }, (_, index) => index + 1);
  const classificationFeaturesPerTree = Math.ceil(Math.sqrt(featureCount));
  const width = Math.max(1, classCount);
  const result = new Float64Array((matrix.length - trainCount) * width);
  for (let treeIndex = 0; treeIndex < treeCount; treeIndex++) {
    const sample = Array.from({ length: trainCount }, () => Math.floor(random() * trainCount));
    let columns = allColumns;
    if (classCount) {
      columns = allColumns.slice();
      for (let index = 0; index < classificationFeaturesPerTree; index++) {
        const other = index + Math.floor(random() * (featureCount - index));
        [columns[index], columns[other]] = [columns[other], columns[index]];
      }
      columns.length = classificationFeaturesPerTree;
    }
    const root = fitTree(matrix, target, sample, columns, 0, classCount);
    for (let row = trainCount; row < matrix.length; row++) {
      const value = predict(root, matrix[row]);
      const offset = (row - trainCount) * width;
      if (classCount) result[offset + value]++;
      else result[offset] += value;
    }
  }
  return result.map((value) => value / treeCount);
}

export function forestClassification(matrix: Float64Array[], target: Int32Array, classCount: number) {
  return predictForest(matrix, target, classCount);
}

export function forestRegression(matrix: Float64Array[], target: Float64Array) {
  return predictForest(matrix, target, 0);
}
