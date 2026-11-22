import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { prepareAnalysis } from "../src/analysis.ts";
import { designMatrix, linearRegression, logisticRegression } from "../src/baseline.ts";
import { fields, parseData, splitRows, type Field } from "../src/data.ts";
import { forestClassification, forestRegression } from "../src/forest.ts";
import { classProbabilities, regressionOutputs, type FeatureColumn } from "../src/kumo/preprocess.ts";
import { classificationMetrics, precisionRecall, regressionMetrics } from "../src/metrics.ts";

const rows = parseData(readFileSync(new URL("../public/data/ibm-hr-attrition.csv", import.meta.url), "utf8"));
const getField = (key: string): Field => fields.find((field) => field.key === key)!;

test("every eligible outcome retains all employee rows with disjoint, repeatable splits", () => {
  assert.equal(rows.length, 1470);
  assert.equal(new Set(rows.map(({ EmployeeNumber }) => EmployeeNumber)).size, 1470);
  assert.equal(fields.length, 31);
  for (const outcome of fields) {
    const task = outcome.kind === "categorical" ? "classification" : "regression";
    const split = splitRows(rows, outcome, task);
    const { train, test } = split;
    assert.deepEqual(split, splitRows(rows, outcome, task));
    assert.equal(train.length + test.length, 1470);
    assert.equal(new Set([...train, ...test]).size, 1470);
    assert.ok(test.length >= 290 && test.length <= 298);
    if (task === "classification") {
      const all = new Set(rows.map((row) => row[outcome.key]));
      assert.deepEqual(new Set(train.map((index) => rows[index][outcome.key])), all);
      assert.deepEqual(new Set(test.map((index) => rows[index][outcome.key])), all);
    }
  }
});

test("preparation uses all selected predictors and keeps targets out of query tensors", () => {
  const outcome = getField("JobRole");
  const predictors = fields.filter((field) => field.key !== outcome.key);
  const { train, test, prepared, features, targets } = prepareAnalysis(rows, { task: "classification", outcome, predictors });
  assert.equal(features.length, 30);
  assert.equal(prepared.columns, features.length);
  assert.equal(prepared.x.length, rows.length * prepared.columns);
  assert.equal(prepared.classes?.length, 9);
  assert.deepEqual([...prepared.yContext].map((code) => prepared.classes![code]), train.map((index) => rows[index].JobRole));
  assert.equal(prepared.yContext.length + test.length, rows.length);
  assert.equal(targets, null);
  assert.throws(() => prepareAnalysis(rows, { task: "classification", outcome, predictors: [outcome] }), /at least one predictor/);
  assert.throws(() => prepareAnalysis(rows, { task: "classification", outcome, predictors: [] }), /at least one predictor/);
});

test("altering regression test labels has no effect on prepared model inputs", () => {
  const outcome = getField("MonthlyIncome");
  const setup = { task: "regression" as const, outcome, predictors: [getField("Age"), getField("JobRole")] };
  const first = prepareAnalysis(rows, setup);
  const changedRows = [...rows];
  for (const index of first.test) changedRows[index] = { ...changedRows[index], MonthlyIncome: "99999999" };
  const second = prepareAnalysis(changedRows, setup);
  assert.deepEqual(second, first);
});

test("baseline design uses context statistics and fitted regression tolerates collinearity", () => {
  const features: FeatureColumn[] = [
    { name: "value", kind: "numerical", values: [0, 1, 2, 10] },
    { name: "same value", kind: "numerical", values: [0, 1, 2, 10] },
    { name: "category", kind: "categorical", values: ["a", "b", "b", "new"] },
  ];
  const matrix = designMatrix(features, 3, 4);
  assert.ok(Math.abs(matrix[3][1] - 9 / Math.sqrt(2 / 3)) < 1e-9);
  assert.equal(matrix[3][3], 0);
  assert.ok(Math.abs(linearRegression(matrix, [3, 5, 7])[0] - 23) < 1e-5);
});

test("binary and one-vs-rest logistic predict held-out examples with normalized scores", () => {
  const binary = designMatrix([{ name: "x", kind: "numerical", values: [-2, -1, 1, 2, -3, 3] }], 4, 6);
  const binaryScores = logisticRegression(binary, Int32Array.of(0, 0, 1, 1), 2);
  assert.ok(binaryScores[0] > 0.9 && binaryScores[3] > 0.9);
  assert.equal(binaryScores[0] + binaryScores[1], 1);
  const multiclass = designMatrix([{ name: "x", kind: "categorical", values: ["a", "a", "b", "b", "c", "c", "b"] }], 6, 7);
  const scores = logisticRegression(multiclass, Int32Array.of(0, 0, 1, 1, 2, 2), 3);
  assert.ok(scores[1] > 0.9);
  assert.ok(Math.abs(scores.reduce((sum, score) => sum + score, 0) - 1) < 1e-12);
});

test("Random Forest classification returns deterministic held-out vote fractions in class order", () => {
  const targets = Int32Array.from({ length: 60 }, (_, row) => Math.floor(row / 20));
  const train = Array.from(targets, (label, row) => label * 10 + row % 20 / 100);
  const feature: FeatureColumn = { name: "x", kind: "numerical", values: [...train, 20.1, 0.1, 10.1] };
  const matrix = designMatrix([feature], targets.length, feature.values.length);
  const votes = forestClassification(matrix, targets, 3);
  assert.deepEqual(votes, forestClassification(matrix, targets, 3));
  assert.equal(votes.length, 9);
  for (const [row, expectedClass] of [2, 0, 1].entries()) {
    const scores = votes.slice(row * 3, (row + 1) * 3);
    assert.equal(scores.reduce((sum, score) => sum + score, 0), 1);
    assert.ok(scores.every((score) => {
      const votes = score * 200;
      return Math.abs(votes - Math.round(votes)) < 1e-12;
    }));
    assert.ok(scores[expectedClass] > 0.9);
  }
});

test("Random Forest regression uses all selected predictors and learns from training rows only", () => {
  const targets = Float64Array.from({ length: 40 }, (_, row) => row < 20 ? 4 : 20);
  const train = Array.from(targets, (target, row) => target + row % 20 / 100);
  const predictor: FeatureColumn = { name: "x", kind: "numerical", values: [...train, 4.1, 20.1] };
  const noise = [3, 5, 7].map((period): FeatureColumn => ({
    name: `noise-${period}`, kind: "numerical", values: Array.from(predictor.values, (_, row) => row % period),
  }));
  const matrix = designMatrix([predictor, ...noise], targets.length, predictor.values.length);
  const predictions = forestRegression(matrix, targets);
  assert.deepEqual([...predictions], [4, 20]);
  const extended = forestRegression([...matrix, Float64Array.of(1, 99999, 0, 0, 0)], targets);
  assert.deepEqual(extended.slice(0, 2), predictions);
});

test("Random Forest keeps pure targets and stops when children would be too small", () => {
  const matrix = [0, 1, 2, 3, 0, 3].map((value) => Float64Array.of(1, value));
  assert.deepEqual([...forestClassification(matrix, Int32Array.of(1, 1, 1, 1), 3)], [0, 1, 0, 0, 1, 0]);
  assert.deepEqual([...forestRegression(matrix, Float64Array.of(7, 7, 7, 7))], [7, 7]);
  const short = forestRegression(matrix, Float64Array.of(4, 4, 20, 20));
  assert.equal(short[0], short[1]);
  assert.ok(short[0] > 4 && short[0] < 20);
  const constant = Array.from({ length: 10 }, () => Float64Array.of(1, 4));
  const constantScores = forestRegression([...constant, Float64Array.of(1, 0), Float64Array.of(1, 2)],
    Float64Array.from({ length: constant.length }, (_, row) => row));
  assert.equal(constantScores[0], constantScores[1]);
  assert.ok(constantScores[0] > 0 && constantScores[0] < 9);
});

test("precision–recall handles ties and metrics match hand calculations", () => {
  const curve = precisionRecall([1, 0, 1, 0], [0.9, 0.9, 0.8, 0.1], 1);
  assert.deepEqual(curve.points, [{ x: 0, y: 1 }, { x: 0.5, y: 0.5 }, { x: 1, y: 2 / 3 }, { x: 1, y: 0.5 }]);
  assert.equal(curve.prevalence, 0.5);
  assert.ok(Math.abs(curve.ap - 7 / 12) < 1e-12);
  assert.deepEqual(classificationMetrics([0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], 2),
    { accuracy: 0.75, balanced: 0.5, confusion: [[3, 0], [1, 0]] });
  assert.deepEqual(regressionMetrics([0, 2, 4], [1, 2, 1]), { mae: 4 / 3, rmse: Math.sqrt(10 / 3), r2: -0.25 });
  assert.equal(regressionMetrics([2, 2], [1, 2]).r2, null);
});

test("Kumo decoding respects active class columns and reverses target standardization", () => {
  assert.deepEqual([...classProbabilities(Float32Array.of(3, 3, NaN, 0, 0, NaN), 2, 2)], [0.5, 0.5, 0.5, 0.5]);
  const { point, quantiles } = regressionOutputs(Float32Array.of(2, 0, 4, -2, 2, 0), 2, { mean: 10, scale: 2 });
  assert.deepEqual([...point], [14, 10]);
  assert.deepEqual([...quantiles], [10, 14, 18, 6, 10, 14]);
});
