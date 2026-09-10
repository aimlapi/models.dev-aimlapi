import { z } from "zod";

import type { SyncProvider } from "../index.js";
import { factorBaseModel, modelMetadata, resolveModelMetadataBaseModel } from "./openrouter.js";

// The public catalog needs no key, and `include` is what turns on the pricing
// and modality blocks this sync depends on.
const API_ENDPOINT = "https://api.aimlapi.com/v1/models?include=pricing,modalities";

// Per-model request schema. It is the only place the API states which reasoning
// controls a model actually accepts, so reasoning_options is read from here
// rather than assumed.
const DOCS_ENDPOINT = "https://api.aimlapi.com/docs-json";

// AI/ML API serves one id under several endpoint types — a model can be both a
// chat model and, say, an image model. Only the chat surface belongs here.
const CHAT_COMPLETIONS_TYPE = "openai/chat-completions";

// Values this schema accepts for an "effort" reasoning control, in the order a
// reader expects to see them. Anything the API documents outside this set is
// dropped rather than coerced.
const EFFORT_VALUES = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"] as const;
const EFFORT_RANK = new Map(EFFORT_VALUES.map((value, index) => [value as string, index]));

/**
 * Models whose documented schema and whose live behaviour disagree, with what
 * the endpoint actually served when asked.
 *
 * `/docs-json` describes a request *shape* per fallback hop, and a hop can
 * advertise a control the vendor behind it still refuses. Reading the schema
 * alone therefore over-states these two, and reading only the first hop
 * under-states others — neither direction is safe to publish unchecked.
 *
 * How these were established, because the method is the evidence: each value
 * was sent to production with `max_tokens` high enough to leave room for an
 * answer, and the **error body** was read rather than the status code. A
 * reasoning model given a small budget returns 400 for an accepted value too
 * ("max_tokens or model output limit was reached"), so a status-only probe
 * reports a rejection that never happened. A real refusal says either
 * "Validation failed" or, for gemini, "Reasoning is mandatory for this
 * endpoint and cannot be disabled".
 *
 * Measured 2026-09-09. Re-probe before trusting these after a vendor change:
 * an entry that has become wrong is worse than no entry.
 */
/**
 * Models that accept any string in `reasoning_effort` — `banana` and `zzz9` both
 * return 200 — so the parameter is not honoured and no ladder can be established
 * by probing. The schema still lists one, inherited from the family template.
 *
 * Publishing it would state a control the caller does not have. Emitting nothing
 * says only that we cannot describe it, which is the truth. Re-probe with a
 * deliberately invalid value before adding a ladder back: a 400 means the field
 * became real.
 *
 * Measured 2026-09-09 across all 87 entries that carry reasoning options; these
 * were the only ones that failed the check. Re-probed 2026-09-10: five still
 * accept `banana` with a 200 and stay.
 *
 * `moonshot/kimi-k3` was removed on 2026-09-10. Re-probed, its control answers
 * 200 and a deliberately invalid effort answers 400 — the field is read and
 * enforced, which is the one thing `[]` asserts is not true. Whether the levels
 * ORDER is still unproven: `low` measured 919 and 1618 reasoning tokens and
 * `high` measured 1943, so the spread within `low` is nearly the distance from
 * `low` to `high`, and `max` timed out at the gateway on both attempts. The id
 * therefore goes back to the schema-derived ladder its card declares, which is
 * the host's own statement about itself, rather than to a claim of ours that
 * the measurement does not support in either direction.
 */
/**
 * Catalogue rows that cannot actually be called: `/v1/models` lists them with a
 * name and prices, inference answers 404 "No endpoints found". Publishing one
 * hands a reader a model id that fails on first use.
 *
 * Measured 2026-09-09; the sibling `-fast` and `-pro` aliases all answered 200,
 * so this is one broken row rather than a broken family. Re-check before adding
 * to this list — a row that starts working should come back.
 *
 * The guard sits on the publish path, so the entry is never created or updated.
 * It does not delete a file that is already on disk: this sync runs with
 * `deleteMissing: false`, so removing a published row still takes an explicit
 * delete, as it did here.
 */
const NOT_CALLABLE: ReadonlySet<string> = new Set([
  "anthropic/claude-opus-4.7-fast",
  // Joined it on 2026-09-09: gone from `/v1/models` (785 rows, neither `-fast`
  // id among them) and `/docs-json` answers 404 for it, so there is nothing
  // left to describe. The host's own deprecation record calls both withdrawn.
  "anthropic/claude-opus-4.8-fast",
]);

/**
 * Ladders the endpoint VALIDATES and then does not act on.
 *
 * Kept apart from `EFFORT_NOT_HONOURED` because the evidence is different and
 * the two must not be confused. There, an invalid value returns 200, which
 * proves the field is never read. Here an invalid value is rejected — the
 * field is read, parsed and enforced — and the levels still fail to order.
 *
 * `moonshotai/kimi-k2-thinking`, measured 2026-09-10 on a prompt hard enough
 * to make a difference visible: `low` produced 6723 and 8207 reasoning tokens
 * across repeats, `high` produced 6508. Unset produced 2919 against `low`'s
 * 2799 on a smaller budget. `low` above `high` is not a small delta in the
 * wrong direction, it is the absence of an ordering, and the lab entry for
 * this model publishes `reasoning_options = []` for the same reason: the
 * model always reasons and the caller cannot steer it.
 *
 * Publishing a ladder here would sell a dial that turns nothing.
 */
/**
 * ## Why this file reads `reasoning_effort` and nothing else
 *
 * The review asks repeatedly for `toggle` and `budget_tokens` options, on the
 * grounds that the labs behind these models expose them and a relay should
 * mirror the control family. The premise does not survive a control.
 *
 * Probed on 2026-09-10 against `deepseek-v4-pro`, `qwen-plus`, `kimi-k3` and
 * `glm-5-turbo` on the OpenAI-compatible path, each alongside a no-field
 * control that answered 200:
 *
 *   enable_thinking: false        200
 *   thinking: {type: 'disabled'}  200
 *   thinking_budget: 128          200
 *   banana_toggle: true           200   <- a field that exists nowhere
 *
 * An invented field is accepted exactly like the real ones, so this surface
 * discards unknown top-level keys rather than forwarding them. A 200 on
 * `enable_thinking` is therefore not evidence that a toggle exists; it is
 * evidence that nothing is listening. Publishing `toggle` on that basis would
 * hand callers a control the gateway throws away.
 *
 * `reasoning_effort` is different in kind: an invalid value is REJECTED on the
 * models that read it, which is how this file tells a live control from a
 * swallowed one, and it is why the two categories below are kept apart.
 */
const EFFORT_VALIDATED_BUT_INERT: ReadonlySet<string> = new Set([
  "moonshotai/kimi-k2-thinking",
]);

const EFFORT_NOT_HONOURED: ReadonlySet<string> = new Set([
  // Re-verified 2026-09-10 by the token-scaling test the review asked for,
  // rather than by status codes alone: `low` measured 3837 and 3841 reasoning
  // tokens, `high` measured 3837 and 3838. Four runs, a spread of four tokens,
  // no movement between the levels at all. Peers publish L/M/H for this id;
  // this host accepts the names and steers nothing.
  "google/gemini-3.1-pro-preview",
  "google/gemma-4-26b-a4b-it",
  "z-ai/glm-5v-turbo",
  "z-ai/glm-5-turbo",
  "alibaba/qwen-plus",
]);

/**
 * Ladders this host was measured to accept AND honour, where that differs from
 * what its schema advertises. `accepted` is not enough on its own: a rung the
 * endpoint takes and then ignores publishes worse advice than one it refuses,
 * because the caller asks for more thinking, is told yes, and is billed for a
 * reply produced with less.
 */
const MEASURED_EFFORTS: Readonly<Record<string, readonly string[]>> = {
  // schema offers none+minimal; both refused, `none` explicitly and by name
  "google/gemini-3.7-flash": ["low", "medium", "high", "max"],
  // schema offers none; refused as a validation error
  "google/gemini-3.6-flash": ["minimal", "low", "medium", "high", "max"],
  // The schema offers `xhigh` on both and the endpoint answers 200 for it, so
  // no status-code probe could catch this. Pinned to the direct hop on a prompt
  // hard enough to show the difference, `xhigh` produced 0 reasoning tokens on
  // o1 and 192 on o3-mini, where `high` produced 10176 and 16000 and even `low`
  // produced ~4500. The rung is accepted and dropped. The review asked for
  // exactly this set on both ids and was right; the host has since stopped
  // advertising `xhigh` here, so this entry becomes a no-op rather than an
  // override once that reaches production.
  "openai/o1": ["low", "medium", "high"],
  "openai/o3-mini": ["low", "medium", "high"],
};

const DOCS_CONCURRENCY = 8;

const PricingUnit = z
  .object({
    name: z.string().nullish(),
    content: z.string().nullish(),
    origin: z.string().nullish(),
    price: z.number().nullish(),
    per: z.number().nullish(),
  })
  .passthrough();

const Info = z
  .object({
    name: z.string().nullish(),
    contextLength: z.number().int().nonnegative().nullish(),
    outputMax: z.number().int().nonnegative().nullish(),
  })
  .passthrough();

export const AimlapiModel = z
  .object({
    id: z.string().min(1),
    type: z.string().nullish(),
    info: Info.nullish(),
    modalities: z
      .object({
        input: z.array(z.string()).nullish(),
        output: z.array(z.string()).nullish(),
      })
      .passthrough()
      .nullish(),
    pricing: z
      .object({
        units: z.array(PricingUnit).nullish(),
      })
      .passthrough()
      .nullish(),
    /** Attached by fetchModels; not part of the upstream payload. */
    reasoningEffort: z.array(z.string()).nullish(),
  })
  .passthrough();

export const AimlapiResponse = z
  .object({
    data: z.array(AimlapiModel).min(1),
  })
  .passthrough();

export type AimlapiModel = z.infer<typeof AimlapiModel>;

type Modality = "text" | "audio" | "image" | "video" | "pdf";

const MODALITIES = new Set<string>(["text", "audio", "image", "video", "pdf"]);

function normalizeModalities(values: readonly string[] | null | undefined): Modality[] {
  const seen = new Set<Modality>();
  for (const value of values ?? []) {
    const normalized = value.toLowerCase();
    if (MODALITIES.has(normalized)) seen.add(normalized as Modality);
  }
  if (seen.size === 0) seen.add("text");
  return [...seen];
}

/**
 * Ids this host also serves on a non-text surface.
 *
 * The catalog lists an id once per endpoint type, and the chat-surface record of
 * an image model claims text output. Measured 2026-09-04:
 * `google/gemini-2.5-flash-image` appears both as `openai/image-generations`
 * with `output: ["image"]` and as `openai/chat-completions` with
 * `output: ["text"]`; the same holds for the `gemini-3-pro-image` and
 * `gemini-3.1-flash-image` families. Judging a record only by its own modalities
 * therefore admits image generators into a chat catalog.
 *
 * An id this host serves as a media model is not a text-only chat model, whatever
 * its chat record claims. Populated from the whole response before any record is
 * judged, because the answer is not in the record itself.
 */
const mediaOutputIDs = new Set<string>();

function indexMediaOutputs(models: readonly AimlapiModel[]): void {
  mediaOutputIDs.clear();
  for (const model of models) {
    const declared = model.modalities?.output ?? [];
    // `normalizeModalities` treats an empty list as text, so an undeclared
    // record must not be read as evidence of anything.
    if (declared.length === 0) continue;
    if (normalizeModalities(declared).some((modality) => modality !== "text")) {
      mediaOutputIDs.add(model.id);
    }
  }
}

/**
 * Ids that are a second spelling of a model already in the catalogue.
 *
 * The host lists a dotted id and a dashed one for the same Anthropic model —
 * `anthropic/claude-opus-4.8` and `anthropic/claude-opus-4-8`, `claude-sonnet-4.6`
 * and `claude-sonnet-4-6` — because the dashed form is an alias its router also
 * answers to. `/v1/models` returns both as full rows, so a sync that trusts the
 * listing publishes the same model twice under two ids.
 *
 * The rule is deliberately narrow, because "looks like an alias" is not enough
 * to delete a row: the dotted twin must be present in the SAME payload and
 * carry the SAME display name. Two genuinely different models that happen to
 * collide on spelling would differ in one of those, and both survive.
 *
 * The dotted form wins because it is what the host's own docs and its
 * `base_model` mapping use; the dashed alias keeps working for callers either
 * way, it just stops being a catalogue entry of its own.
 */
const ALIAS_DUPLICATE_IDS = new Set<string>();

/** `anthropic/claude-opus-4-8` -> `anthropic/claude-opus-4.8`; undefined if not that shape. */
function dottedTwin(id: string): string | undefined {
  const dotted = id.replace(/-(\d+)-(\d+)$/, "-$1.$2");
  return dotted === id ? undefined : dotted;
}

function indexAliasDuplicates(models: readonly AimlapiModel[]): void {
  ALIAS_DUPLICATE_IDS.clear();
  const nameByID = new Map<string, string | undefined>();
  for (const model of models) nameByID.set(model.id, model.info?.name ?? undefined);

  for (const model of models) {
    const twin = dottedTwin(model.id);
    if (twin === undefined || !nameByID.has(twin)) continue;
    if (nameByID.get(twin) !== nameByID.get(model.id)) continue;
    ALIAS_DUPLICATE_IDS.add(model.id);
  }
}

function isChatTextModel(model: AimlapiModel): boolean {
  if (model.type !== CHAT_COMPLETIONS_TYPE) return false;
  // Cross-surface check first: the chat record of a media model does not admit
  // to being one.
  if (mediaOutputIDs.has(model.id)) return false;
  const output = normalizeModalities(model.modalities?.output);
  // A chat model whose output is not purely text is a media model riding the
  // chat protocol, and does not belong in a chat catalog.
  return output.length === 1 && output[0] === "text";
}

/**
 * Lab entry this id is a host for. AI/ML API is an aggregator and authors none
 * of these models, so every entry has to point at the lab file rather than
 * restate it.
 */
function baseModelFor(id: string): string | undefined {
  return resolveModelMetadataBaseModel(id);
}

function baseReasoning(baseModelID: string): boolean {
  try {
    return modelMetadata(baseModelID).reasoning === true;
  } catch {
    return false;
  }
}

/**
 * Prices are quoted as `price` per `per` tokens; models.dev stores dollars per
 * million. The unit discriminator is `origin`, not `measure`: provided is
 * input, generated is output, cached is a cache read. Only text token charges
 * are taken — a model's image or audio units are a different surface.
 */
function perMillion(units: readonly z.infer<typeof PricingUnit>[], origin: string): number | undefined {
  const unit = units.find(
    (candidate) => candidate.name === "token" && candidate.content === "text" && candidate.origin === origin,
  );
  if (!unit || unit.price == null || !unit.per) return undefined;
  // Round before returning. `(0.0000000078 / 1) * 1e6` is
  // 0.0078000000000000005 in IEEE double, and `formatNumber` serializes the
  // residue verbatim — so the catalogue carried a price with fifteen decimals
  // that changes shape whenever the upstream `per` does. Six decimals is finer
  // than any published USD/MTok figure here.
  return Math.round(((unit.price / unit.per) * 1_000_000) * 1e6) / 1e6;
}

function positive(value: number | null | undefined): number | undefined {
  return value != null && value > 0 ? value : undefined;
}

/**
 * Reads the documented `reasoning_effort` values for one model. Returns
 * undefined when the docs do not describe the control, which is treated as
 * "cannot state it" rather than "the model has none".
 *
 * One model's schema can carry the control more than once, because the request
 * body is a union of per-family variants and several of them accept it with
 * different ladders. Taking the first one found means taking whichever variant
 * happens to be earliest in the document, which drops real values: at the time
 * of writing that costs `minimal` on the gpt-5 family, `max` on claude-sonnet-4.6
 * and claude-opus-4.8, and `none` on the gemini flash models. The union across
 * every occurrence is what the endpoint actually accepts.
 */
async function fetchReasoningEffort(id: string): Promise<string[] | undefined> {
  const url = `${DOCS_ENDPOINT}?model=${encodeURIComponent(id)}&endpoint=${encodeURIComponent(CHAT_COMPLETIONS_TYPE)}`;
  let payload: unknown;
  try {
    const response = await fetch(url);
    if (!response.ok) return undefined;
    payload = await response.json();
  } catch {
    return undefined;
  }

  if (EFFORT_NOT_HONOURED.has(id) || EFFORT_VALIDATED_BUT_INERT.has(id)) return undefined;

  const measured = MEASURED_EFFORTS[id];
  if (measured) return [...measured];

  const found = new Set<string>();
  collectReasoningEffortEnums(payload, found);

  const values = [...found]
    .filter((value) => EFFORT_RANK.has(value))
    .sort((a, b) => EFFORT_RANK.get(a)! - EFFORT_RANK.get(b)!);
  return values.length > 0 ? values : undefined;
}

function collectReasoningEffortEnums(node: unknown, into: Set<string>): void {
  if (Array.isArray(node)) {
    for (const item of node) collectReasoningEffortEnums(item, into);
    return;
  }
  if (node === null || typeof node !== "object") return;

  const record = node as Record<string, unknown>;
  const effort = record["reasoning_effort"];
  if (effort !== null && typeof effort === "object") {
    const values = (effort as Record<string, unknown>)["enum"];
    if (Array.isArray(values) && values.every((value) => typeof value === "string")) {
      for (const value of values as string[]) into.add(value);
    }
  }

  // Keep walking either way: the same schema can describe the control again in
  // a sibling variant of the request-body union.
  for (const value of Object.values(record)) collectReasoningEffortEnums(value, into);
}

async function attachReasoningEffort(models: AimlapiModel[]): Promise<void> {
  // Only models whose lab entry says they reason need the control documented,
  // and only those are worth a request.
  const pending = models.filter((model) => {
    if (NOT_CALLABLE.has(model.id)) return false;
    if (ALIAS_DUPLICATE_IDS.has(model.id)) return false;
    if (!isChatTextModel(model)) return false;
    const base = baseModelFor(model.id);
    return base !== undefined && baseReasoning(base);
  });

  let cursor = 0;
  const workers = Array.from({ length: Math.min(DOCS_CONCURRENCY, pending.length) }, async () => {
    while (cursor < pending.length) {
      const model = pending[cursor++];
      if (model === undefined) return;
      model.reasoningEffort = await fetchReasoningEffort(model.id);
    }
  });
  await Promise.all(workers);
}

export const aimlapi = {
  id: "aimlapi",
  name: "AI/ML API",
  modelsDir: "providers/aimlapi/models",
  // The catalog turns over quickly and lists far more than the chat surface, so
  // a local model missing from one response is not proof that it is gone.
  deleteMissing: false,
  sourceID(model) {
    if (NOT_CALLABLE.has(model.id)) return undefined;
    if (ALIAS_DUPLICATE_IDS.has(model.id)) return undefined;
    return isChatTextModel(model) ? model.id : undefined;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `${ids.length} AI/ML API chat models were skipped because this repository has no lab entry to point \`base_model\` at, or because the API does not document the reasoning control a reasoning model requires.`,
      `Skipped remote IDs: ${ids.map((id) => `\`${id}\``).join(", ")}`,
    ];
  },
  missingNotice(paths) {
    if (paths.length === 0) return [];
    return [
      `${paths.length} local AI/ML API models were absent from the catalog and were retained for manual lifecycle review.`,
      `Retained local paths: ${paths.map((item) => `\`${item}\``).join(", ")}`,
    ];
  },
  async fetchModels() {
    const response = await fetch(API_ENDPOINT);
    if (!response.ok) {
      throw new Error(`AI/ML API request failed: ${response.status} ${response.statusText}`);
    }
    const raw = await response.json();
    const parsed = AimlapiResponse.parse(raw);
    indexMediaOutputs(parsed.data);
    indexAliasDuplicates(parsed.data);
    await attachReasoningEffort(parsed.data);
    return parsed;
  },
  parseModels(raw) {
    const models = AimlapiResponse.parse(raw).data;
    // Replays parse a cached payload without going through fetchModels.
    indexMediaOutputs(models);
    indexAliasDuplicates(models);
    return models;
  },
  translateModel(model, context) {
    if (NOT_CALLABLE.has(model.id)) return undefined;
    if (ALIAS_DUPLICATE_IDS.has(model.id)) return undefined;
    if (!isChatTextModel(model)) return undefined;

    const existing = context.existing(model.id);

    // AI/ML API hosts other people's models, so the entry must reference the
    // lab file instead of duplicating it. Without a lab entry to point at there
    // is nothing correct to write: inlining the metadata is what this schema
    // forbids, and authoring the lab file would mean sourcing capability data
    // the catalog does not publish.
    const base = existing?.base_model ?? baseModelFor(model.id);
    if (base === undefined) return undefined;

    // Required whenever the base model reasons. Only the API's own request
    // schema can say which values it takes, so a model whose docs stay silent
    // is skipped rather than given an invented control.
    let reasoningOptions: Array<{ type: "effort"; values: string[] }> | undefined;
    if (baseReasoning(base)) {
      // A model that reasons but honours no caller control gets an empty list:
      // the schema requires the field, and an empty one states the truth —
      // reasoning happens, nothing about it is selectable.
      if (EFFORT_NOT_HONOURED.has(model.id) || EFFORT_VALIDATED_BUT_INERT.has(model.id)) {
        reasoningOptions = [];
      } else {
        const values = model.reasoningEffort ?? undefined;
        if (values === undefined || values.length === 0) return undefined;
        reasoningOptions = [{ type: "effort", values }];
      }
    }

    const units = model.pricing?.units ?? [];
    const info = model.info ?? {};
    const contextLimit = positive(info.contextLength);
    const outputLimit = positive(info.outputMax);
    // Only what the catalog actually publishes. It reports a context window and
    // an output cap but no input cap, and equating the input cap with the whole
    // context would overwrite the lab's correct split (e.g. 272k in + 128k out
    // within a 400k window) with a wrong number.
    const limit =
      contextLimit === undefined && outputLimit === undefined
        ? undefined
        : { context: contextLimit, output: outputLimit };

    // Everything else — the capability flags, description, dates, modalities —
    // is the lab's to state and is inherited. factorBaseModel drops whatever
    // matches the base, so the file carries only what is genuinely ours.
    return {
      id: model.id,
      model: factorBaseModel(
        base,
        {
          cost: {
            input: perMillion(units, "provided") ?? existing?.cost?.input,
            output: perMillion(units, "generated") ?? existing?.cost?.output,
            cache_read: perMillion(units, "cached") ?? existing?.cost?.cache_read,
          },
          // Aliases such as `-pro` and `-fast` factor onto the base model and
          // would inherit its display name, so the catalogue would list two rows
          // called "GPT-5.6 Luna". The host names them apart; carry that through
          // and the override drops itself when the names already agree.
          name: info.name ?? undefined,
          reasoning_options: reasoningOptions,
          limit,
        },
        limit,
        existing?.base_model === base ? existing?.base_model_omit : undefined,
      ),
    };
  },
} satisfies SyncProvider<AimlapiModel>;
