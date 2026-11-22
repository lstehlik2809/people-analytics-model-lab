import type { FeatureColumn, Task } from "./kumo/preprocess.ts";

type FieldKind = "numerical" | "categorical";
type FieldGroup = "Personal" | "Role and work" | "Pay and benefits" | "Experience" | "Survey and ratings";

export type Field = {
  key: string;
  name: string;
  kind: FieldKind;
  group: FieldGroup;
  unit?: string;
  labels?: Record<string, string>;
};

const satisfaction = { 1: "Low", 2: "Medium", 3: "High", 4: "Very high" };
const groups = ["Personal", "Role and work", "Pay and benefits", "Experience", "Survey and ratings"] as const;
export { groups };

export const fields = [
  { key: "Age", name: "Age", kind: "numerical", group: "Personal", unit: "years" },
  { key: "DistanceFromHome", name: "Distance from home", kind: "numerical", group: "Personal", unit: "dataset units" },
  { key: "Education", name: "Education level", kind: "categorical", group: "Personal", labels: { 1: "Below college", 2: "College", 3: "Bachelor’s", 4: "Master’s", 5: "Doctorate" } },
  { key: "EducationField", name: "Education field", kind: "categorical", group: "Personal" },
  { key: "Gender", name: "Gender", kind: "categorical", group: "Personal" },
  { key: "MaritalStatus", name: "Marital status", kind: "categorical", group: "Personal" },
  { key: "Attrition", name: "Attrition", kind: "categorical", group: "Role and work" },
  { key: "BusinessTravel", name: "Business travel", kind: "categorical", group: "Role and work", labels: { Non_Travel: "No travel", Travel_Rarely: "Rarely", Travel_Frequently: "Frequently" } },
  { key: "Department", name: "Department", kind: "categorical", group: "Role and work" },
  { key: "JobLevel", name: "Job level", kind: "categorical", group: "Role and work" },
  { key: "JobRole", name: "Job role", kind: "categorical", group: "Role and work" },
  { key: "OverTime", name: "Overtime", kind: "categorical", group: "Role and work" },
  { key: "DailyRate", name: "Daily rate", kind: "numerical", group: "Pay and benefits", unit: "dataset units" },
  { key: "HourlyRate", name: "Hourly rate", kind: "numerical", group: "Pay and benefits", unit: "dataset units" },
  { key: "MonthlyIncome", name: "Monthly income", kind: "numerical", group: "Pay and benefits", unit: "dataset currency units" },
  { key: "MonthlyRate", name: "Monthly rate", kind: "numerical", group: "Pay and benefits", unit: "dataset units" },
  { key: "PercentSalaryHike", name: "Salary hike (%)", kind: "numerical", group: "Pay and benefits", unit: "percentage points" },
  { key: "StockOptionLevel", name: "Stock option level", kind: "categorical", group: "Pay and benefits" },
  { key: "NumCompaniesWorked", name: "Companies worked at", kind: "numerical", group: "Experience", unit: "companies" },
  { key: "TotalWorkingYears", name: "Total working years", kind: "numerical", group: "Experience", unit: "years" },
  { key: "TrainingTimesLastYear", name: "Training times last year", kind: "numerical", group: "Experience", unit: "count" },
  { key: "YearsAtCompany", name: "Years at company", kind: "numerical", group: "Experience", unit: "years" },
  { key: "YearsInCurrentRole", name: "Years in current role", kind: "numerical", group: "Experience", unit: "years" },
  { key: "YearsSinceLastPromotion", name: "Years since last promotion", kind: "numerical", group: "Experience", unit: "years" },
  { key: "YearsWithCurrManager", name: "Years with current manager", kind: "numerical", group: "Experience", unit: "years" },
  { key: "EnvironmentSatisfaction", name: "Environment satisfaction", kind: "categorical", group: "Survey and ratings", labels: satisfaction },
  { key: "JobInvolvement", name: "Job involvement", kind: "categorical", group: "Survey and ratings", labels: satisfaction },
  { key: "JobSatisfaction", name: "Job satisfaction", kind: "categorical", group: "Survey and ratings", labels: satisfaction },
  { key: "PerformanceRating", name: "Performance rating", kind: "categorical", group: "Survey and ratings", labels: { 3: "Excellent", 4: "Outstanding" } },
  { key: "RelationshipSatisfaction", name: "Relationship satisfaction", kind: "categorical", group: "Survey and ratings", labels: satisfaction },
  { key: "WorkLifeBalance", name: "Work-life balance", kind: "categorical", group: "Survey and ratings", labels: { 1: "Bad", 2: "Good", 3: "Better", 4: "Best" } },
] as const satisfies readonly Field[];

export type Row = Record<string, string>;
export type Split = { train: number[]; test: number[] };

// The retained source is an unquoted, comma-delimited CSV with no missing cells.
export function parseData(csv: string): Row[] {
  const [header, ...records] = csv.trim().split(/\r?\n/);
  const keys = header.split(",");
  if (!fields.every(({ key }) => keys.includes(key))) throw new Error("Missing dataset fields.");
  return records.map((record) => {
    const values = record.split(",");
    if (values.length !== keys.length) throw new Error("Malformed dataset row.");
    return Object.fromEntries(keys.map((key, index) => [key, values[index]]));
  });
}

export function randomSource(seed: number) {
  return () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
}

function shuffle(items: number[], random: () => number) {
  for (let cursor = items.length - 1; cursor > 0; cursor--) {
    const other = Math.floor(random() * (cursor + 1));
    [items[cursor], items[other]] = [items[other], items[cursor]];
  }
}

// Repeat 0 is the split behind the displayed results. Later repeats reshuffle the same rows.
export function splitRows(rows: Row[], outcome: Field, task: Task, repeat = 0): Split {
  const random = randomSource(740512 ^ Math.imul(repeat, 0x9e3779b1));
  const strata = new Map<string, number[]>();
  for (const [index, row] of rows.entries()) {
    const key = task === "classification" ? row[outcome.key] : "";
    if (!strata.has(key)) strata.set(key, []);
    strata.get(key)!.push(index);
  }
  const train: number[] = [];
  const test: number[] = [];
  for (const indexes of strata.values()) {
    if (indexes.length < 2) throw new Error("A class needs at least two rows for evaluation.");
    shuffle(indexes, random);
    const testCount = Math.max(1, Math.min(indexes.length - 1, Math.round(indexes.length / 5)));
    test.push(...indexes.slice(0, testCount));
    train.push(...indexes.slice(testCount));
  }
  shuffle(train, random);
  shuffle(test, random);
  return { train, test };
}

export function getFeatures(rows: Row[], indexes: number[], predictors: Field[]): FeatureColumn[] {
  return predictors.map((field) => field.kind === "numerical"
    ? { name: field.key, kind: "numerical", values: Float64Array.from(indexes, (index) => Number(rows[index][field.key])) }
    : { name: field.key, kind: "categorical", values: indexes.map((index) => rows[index][field.key]) });
}
