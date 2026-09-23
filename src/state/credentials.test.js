import assert from "node:assert/strict";
import test from "node:test";
import { checkHfCredentialStatus, loadAppCredentialState, saveAppCredentialState } from "./credentials.js";

test("disabled API auth loads without a credential request", async () => {
  const calls = [];
  const state = await loadAppCredentialState(async (command) => {
    calls.push(command);
    if (command === "load_app_settings") return { server: { authEnabled: false } };
    throw new Error("unexpected credential request");
  });
  assert.deepEqual(calls, ["load_app_settings"]);
  assert.equal(state.token, null);
});

test("enabled API auth preserves a credential error", async () => {
  const calls = [];
  const state = await loadAppCredentialState(async (command) => {
    calls.push(command);
    if (command === "load_app_settings") return { server: { authEnabled: true } };
    throw new Error("credential denied");
  });
  assert.deepEqual(calls, ["load_app_settings", "api_auth_token"]);
  assert.match(state.error, /credential denied/);
  assert.equal(state.token, null);
});

test("enabled API auth reports a missing cached token", async () => {
  const state = await loadAppCredentialState(async (command) =>
    command === "load_app_settings" ? { server: { authEnabled: true } } : null);
  assert.match(state.error, /unavailable/);
  assert.equal(state.token, null);
});

test("settings save uses native returned token without another read", async () => {
  const calls = [];
  const state = await saveAppCredentialState(async (command, args) => {
    calls.push(command);
    assert.equal(args.apiAuthToken, "secret");
    return [{ server: { authEnabled: true } }, { running: true, auth_required: true }, "secret"];
  }, { server: { authEnabled: true } }, "secret");
  assert.deepEqual(calls, ["save_app_settings"]);
  assert.equal(state.token, "secret");
  assert.equal(state.status.auth_required, true);
});

test("HuggingFace status is requested only on an explicit check and preserves read errors", async () => {
  const calls = [];
  const invoke = async (command) => {
    calls.push(command);
    throw new Error("credential denied");
  };
  assert.deepEqual(calls, []);
  await assert.rejects(checkHfCredentialStatus(invoke), /credential denied/);
  assert.deepEqual(calls, ["hf_token_status"]);
});
