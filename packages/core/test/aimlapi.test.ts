import { expect, test } from "bun:test";

import { labEffortValues, resolveLadder } from "../src/sync/providers/aimlapi.js";
import { providerDirForMetadataLab } from "../src/sync/providers/openrouter.js";

// The lookup that silently kept the host enum for every GLM id: `base_model`
// names the metadata lab, the ladder lives under the provider directory, and
// for these two they differ.
test("maps a metadata lab to its first-party provider directory", () => {
  expect(providerDirForMetadataLab("zhipuai")).toBe("zai");
  expect(providerDirForMetadataLab("meta")).toBe("llama");
  expect(providerDirForMetadataLab("deepseek")).toBe("deepseek");
  expect(providerDirForMetadataLab("no-such-lab")).toBe("no-such-lab");
});

test("reads the lab ladder through the provider directory, not the model record", () => {
  // `models/zhipuai/*.toml` has no reasoning_options; `providers/zai/models`
  // does. A lookup on the wrong one returns undefined and looks like "no lab
  // ladder", which is a different claim.
  expect(labEffortValues("zhipuai/glm-5.3-flash")).toEqual(["low", "high", "max"]);
  expect(labEffortValues("deepseek/deepseek-v4-pro")).toEqual(["high", "max"]);
});

test("a dated snapshot borrows the ladder of the model it snapshots", () => {
  // There is no `deepseek-v4-pro-0813.toml` in the lab; without the borrow it
  // fell through to the host enum and came out wider than its own base.
  expect(labEffortValues("deepseek/deepseek-v4-pro-0813")).toEqual(labEffortValues("deepseek/deepseek-v4-pro"));
});

test("a lab entry with toggle only and no effort ladder yields undefined", () => {
  expect(labEffortValues("zhipuai/glm-4.7-flash")).toBeUndefined();
});

const HOST = ["none", "low", "medium", "high"] as const;

test("intersects the host enum with the lab ladder and drops the rest", () => {
  // lab high/max, host none..high: `low` and `medium` go although the host
  // accepts them — that is the doctrine, and this test pins its cost.
  expect(resolveLadder("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro", HOST)).toEqual(["none", "high"]);
});

test("`none` survives only where measured as a real off switch", () => {
  // Same lab, same host enum; only the measurement differs.
  expect(resolveLadder("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro", HOST)).toContain("none");
  expect(resolveLadder("alibaba/qwen3.7-max", "alibaba/qwen3.7-max", HOST)).not.toContain("none");
});

test("`none` is kept where the lab itself lists it", () => {
  const host = ["none", "low", "medium", "high"];
  expect(resolveLadder("openai/gpt-5.6-luna", "openai/gpt-5.6-luna", host)).toContain("none");
});

test("an empty intersection keeps the host enum rather than claiming no control", () => {
  // Every lab rung refused here: publishing `[]` would say the field does not
  // exist, which is the one thing the host's enum rules out.
  const host = ["low", "medium"];
  expect(resolveLadder("some/model", "deepseek/deepseek-v4-pro", host)).toEqual(["low", "medium"]);
});

test("a measurement narrows but never widens past the lab", () => {
  // gpt-5.4-pro: lab medium/high/xhigh, host low/medium/high, measured medium/high.
  expect(resolveLadder("openai/gpt-5.4-pro", "openai/gpt-5.4-pro", ["low", "medium", "high"])).toEqual(["medium", "high"]);
});
