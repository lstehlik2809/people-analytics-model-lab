import type { Result } from "./analysis.ts";

export type Point = { x: number; y: number };

export function precisionRecall(actual: number[], scores: ArrayLike<number>, positive: number) {
  const ranked = actual.map((label, index) => ({ positive: label === positive, score: scores[index] }))
    .sort((a, b) => b.score - a.score);
  const positives = ranked.filter((row) => row.positive).length;
  const prevalence = positives / ranked.length;
  const points = [{ x: 0, y: 1 }];
  let ap = 0;
  let correct = 0;
  let index = 0;
  while (index < ranked.length) {
    const threshold = ranked[index].score;
    do {
      if (ranked[index].positive) correct++;
      index++;
    } while (index < ranked.length && ranked[index].score === threshold);
    const point = { x: correct / positives, y: correct / index };
    ap += (point.x - points[points.length - 1].x) * point.y;
    points.push(point);
  }
  return { ap, points, prevalence };
}

export function classificationMetrics(actual: number[], scores: ArrayLike<number>, classes: number) {
  const confusion = Array.from({ length: classes }, () => Array<number>(classes).fill(0));
  actual.forEach((code, row) => {
    const offset = row * classes;
    let predicted = 0;
    for (let i = 1; i < classes; i++) if (scores[offset + i] > scores[offset + predicted]) predicted = i;
    confusion[code][predicted]++;
  });
  const accuracy = confusion.reduce((sum, row, code) => sum + row[code], 0) / actual.length;
  const balanced = confusion.reduce((sum, row, code) => sum + row[code] / row.reduce((a, b) => a + b, 0), 0) / classes;
  return { accuracy, balanced, confusion };
}

export function regressionMetrics(actual: number[], predictions: ArrayLike<number>) {
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  let absoluteError = 0;
  let squaredError = 0;
  let targetVariance = 0;
  actual.forEach((target, row) => {
    const error = predictions[row] - target;
    absoluteError += Math.abs(error);
    squaredError += error ** 2;
    targetVariance += (target - mean) ** 2;
  });
  return {
    mae: absoluteError / actual.length,
    rmse: Math.sqrt(squaredError / actual.length),
    r2: targetVariance > 0 ? 1 - squaredError / targetVariance : null,
  };
}

export function classScores(scores: ArrayLike<number>, count: number, code: number): Float64Array {
  return Float64Array.from({ length: scores.length / count }, (_, row) => scores[row * count + code]);
}

export const modelKeys = ["kumo", "baseline", "forest"] as const;

/** One split's metrics, each with a value per model in modelKeys order. Average precision is keyed by class value. */
export type SplitScores = Record<string, number[]>;

export function splitScores(result: Result): SplitScores {
  const predictions = modelKeys.map((key) => result[key]);
  if (result.task === "regression") {
    const metrics = predictions.map((values) => regressionMetrics(result.actual, values));
    return {
      mae: metrics.map(({ mae }) => mae),
      rmse: metrics.map(({ rmse }) => rmse),
      r2: metrics.map(({ r2 }) => r2 ?? NaN),
    };
  }
  const { actual, classes } = result;
  const metrics = predictions.map((values) => classificationMetrics(actual, values, classes.length));
  const scores: SplitScores = {
    balanced: metrics.map(({ balanced }) => balanced),
    accuracy: metrics.map(({ accuracy }) => accuracy),
  };
  // Class codes follow each split's training order, so key by the class value.
  classes.forEach((value, code) => {
    scores[`ap:${value}`] = predictions.map((values) =>
      precisionRecall(actual, classScores(values, classes.length, code), code).ap);
  });
  return scores;
}

/** Per model: mean and range of one metric across splits, and the number of splits where it was best. Ties count for each tied model. */
export function acrossSplits(splits: SplitScores[], metric: string, direction: "higher" | "lower") {
  const rows = splits.map((split) => split[metric]);
  const best = rows.map((row) => direction === "higher" ? Math.max(...row) : Math.min(...row));
  return modelKeys.map((_, model) => {
    const values = rows.map((row) => row[model]);
    return {
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      best: values.filter((value, split) => value === best[split]).length,
    };
  });
}
