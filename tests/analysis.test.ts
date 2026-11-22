import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { prepareAnalysis } from "../src/analysis.ts";
import { designMatrix, linearRegression, logisticRegression } from "../src/baseline.ts";
import { fields, parseData, splitRows, type Field } from "../src/data.ts";
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
