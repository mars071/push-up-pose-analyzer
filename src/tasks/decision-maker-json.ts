/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * JSON import/export for the Decision Maker text playground.
 *
 * Accepted request shapes (see @mediapipe/tasks-decision decision.d.ts):
 *
 * 1. One question, as passed to evaluateBoolean / evaluateChoice / evaluateScore:
 *      { "type": "boolean" | "choice" | "score", "text": "...", "question": { ... } }
 *    `type` may be omitted (it's inferred from the question fields), `input`
 *    may be used instead of `text`, and the question fields may be inlined.
 *
 * 2. A ClassifierSchema, as passed to evaluate(input, schema):
 *      { "input": "...", "context": "...", "questions": [{ "id", "type", "prompt", "options", "threshold" }] }
 *    The playground shows one question at a time, so the first one is loaded.
 */

export type QuestionKind = 'boolean' | 'choice' | 'score';

export interface Item {
  label: string;
  description: string;
}

/** Everything the playground form holds for one question type. */
export interface Draft {
  input: string;
  condition: string;
  /** Boolean only: what "true" / "false" mean (sent as `options`). Optional. */
  trueDescription: string;
  falseDescription: string;
  /** Optional domain context prepended to the question. */
  context: string;
  /** Boolean only: P(true) needed to answer Yes. */
  threshold: number;
  items: Item[];
  instructions: string;
}

export interface ParsedRequest {
  kind: QuestionKind;
  draft: Draft;
  /** Something the user should know (e.g. only the first of several questions was loaded). */
  note?: string;
}

const KIND_ALIASES: Record<string, QuestionKind> = {
  boolean: 'boolean',
  binary: 'boolean',
  choice: 'choice',
  categorical: 'choice',
  score: 'score',
  ordinal: 'score',
};

const str = (v: unknown) => (typeof v === 'string' ? v : v == null ? '' : String(v));

function kindOf(type: unknown, q: Record<string, unknown>): QuestionKind {
  if (type !== undefined) {
    const kind = KIND_ALIASES[str(type).toLowerCase()];
    if (!kind) throw new Error(`Unknown question type "${str(type)}". Use boolean, choice, or score.`);
    return kind;
  }
  if ('condition' in q) return 'boolean';
  if ('rubric' in q) return 'score';
  if ('criteria' in q || 'options' in q) return 'choice';
  throw new Error('Can’t tell the question type. Add "type": "boolean", "choice", or "score".');
}

function toItems(options: unknown): Item[] {
  if (options === undefined) return [];
  if (!Array.isArray(options)) throw new Error('"options" must be a list of { "label", "description" }.');
  return options.map((o, i) => {
    if (typeof o === 'string') return { label: o, description: '' };
    if (!o || typeof o !== 'object' || !('label' in o)) throw new Error(`options[${i}] needs a "label".`);
    return { label: str((o as Item).label), description: str((o as Item).description) };
  });
}

/** "Satisfied: helpful support" -> { label: "Satisfied", description: "helpful support" } */
function rubricItem(level: unknown): Item {
  const s = str(level);
  const i = s.indexOf(': ');
  return i > 0 ? { label: s.slice(0, i), description: s.slice(i + 2) } : { label: s, description: '' };
}

/** Parses a pasted Decision API request into a playground draft. Throws with a readable message. */
export function parseDecisionRequest(json: string, empty: Omit<Draft, 'input'>): ParsedRequest {
  let root: unknown;
  try {
    root = JSON.parse(json);
  } catch (e: any) {
    throw new Error(`Invalid JSON: ${e?.message ?? e}`);
  }
  if (!root || typeof root !== 'object' || Array.isArray(root)) throw new Error('Expected a JSON object.');
  const obj = root as Record<string, any>;
  const input = str(obj.text ?? obj.input);
  let note: string | undefined;

  // ClassifierSchema: { questions: [...] }, optionally wrapped as { schema: {...} }.
  const schema = obj.questions ? obj : obj.schema?.questions ? obj.schema : undefined;
  if (schema) {
    const questions = schema.questions;
    if (!Array.isArray(questions) || questions.length === 0) throw new Error('"questions" must be a non-empty list.');
    if (questions.length > 1) note = `Loaded question 1 of ${questions.length} (${str(questions[0].id) || 'unnamed'}).`;
    const q = questions[0] as Record<string, unknown>;
    const kind = kindOf(q.type, q);
    const prompt = str(q.prompt);
    const context = str(schema.context ?? obj.context);
    const items = toItems(q.options);
    const draft: Draft = { ...structuredClone(empty), input, context };
    if (kind === 'boolean') {
      draft.condition = prompt;
      draft.threshold = typeof q.threshold === 'number' ? q.threshold : 0.5;
      draft.trueDescription = items.find((it) => it.label === 'true')?.description ?? '';
      draft.falseDescription = items.find((it) => it.label === 'false')?.description ?? '';
    } else {
      draft.instructions = prompt;
      draft.items = items;
    }
    return { kind, draft, note };
  }

  // One question: { type, text, question } or the question fields inlined.
  const q = (obj.question && typeof obj.question === 'object' ? obj.question : obj) as Record<string, unknown>;
  const kind = kindOf(obj.type ?? obj.kind ?? q.type, q);
  const draft: Draft = { ...structuredClone(empty), input, context: str(q.context ?? obj.context) };
  if (kind === 'boolean') {
    if (!str(q.condition)) throw new Error('A boolean question needs a "condition".');
    const items = toItems(q.options);
    draft.condition = str(q.condition);
    draft.threshold = typeof q.threshold === 'number' ? q.threshold : 0.5;
    draft.trueDescription = items.find((it) => it.label === 'true')?.description ?? '';
    draft.falseDescription = items.find((it) => it.label === 'false')?.description ?? '';
  } else if (kind === 'choice') {
    const criteria = q.criteria && typeof q.criteria === 'object' ? (q.criteria as Record<string, unknown>) : {};
    draft.items = [
      ...Object.entries(criteria).map(([label, description]) => ({ label, description: str(description) })),
      ...toItems(q.options),
    ];
    draft.instructions = str(q.instructions);
  } else {
    draft.items = q.options !== undefined ? toItems(q.options) : ((q.rubric as unknown[]) ?? []).map(rubricItem);
    draft.instructions = str(q.instructions);
  }
  if (draft.input === '') note = 'No "text" in the request; enter the input text before evaluating.';
  return { kind, draft, note };
}

/** The request the playground would send for the current form, as pretty JSON. */
export function toRequestJson(kind: QuestionKind, text: string, question: object): string {
  return JSON.stringify({ type: kind, text, question }, null, 2);
}
