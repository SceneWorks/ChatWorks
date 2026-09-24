import assert from "node:assert/strict";
import test from "node:test";
import { sampleVideoAttachment } from "../src/media/video.js";

const tick = () => new Promise((resolve) => setImmediate(resolve));

function browserVideo({ duration = 3, frameCallback = true } = {}) {
  const originalDocument = globalThis.document;
  const originalReader = globalThis.FileReader;
  const originalCreate = URL.createObjectURL;
  const originalRevoke = URL.revokeObjectURL;
  const listeners = new Map();
  const callbacks = new Map();
  const drawn = [];
  const revoked = [];
  let nextId = 0;
  let plays = 0;
  const video = {
    duration, videoWidth: 2, videoHeight: 2, paused: true, presentedColor: "black",
    pause() { this.paused = true; },
    play() { plays++; this.paused = false; return Promise.resolve(); },
    removeAttribute() {}, load() {},
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
    set currentTime(value) { this.time = value; queueMicrotask(() => listeners.get("seeked")?.()); },
    get currentTime() { return this.time ?? 0; },
    cancelVideoFrameCallback(id) { callbacks.delete(id); },
  };
  if (frameCallback) {
    video.requestVideoFrameCallback = (callback) => {
      const id = ++nextId;
      callbacks.set(id, callback);
      return id;
    };
  }
  const canvas = {
    getContext: () => ({ drawImage: () => drawn.push(video.presentedColor) }),
    toBlob: (callback) => callback(new Blob([video.presentedColor], { type: "image/jpeg" })),
  };
  globalThis.document = { createElement: (kind) => kind === "video" ? video : canvas };
  globalThis.FileReader = class {
    readAsDataURL(blob) {
      blob.text().then((value) => {
        this.result = `data:image/jpeg;base64,${Buffer.from(value).toString("base64")}`;
        this.onload();
      });
    }
  };
  URL.createObjectURL = () => "blob:local-video";
  URL.revokeObjectURL = (value) => revoked.push(value);
  return {
    video, callbacks, drawn, revoked,
    get plays() { return plays; },
    begin(signal) {
      const result = sampleVideoAttachment({ name: "fixture.mp4" }, signal);
      video.onloadedmetadata();
      return result;
    },
    async frame(mediaTime, color) {
      const entry = callbacks.entries().next().value;
      assert.ok(entry, "a presented-frame callback is pending");
      const [id, callback] = entry;
      callbacks.delete(id);
      video.presentedColor = color;
      callback(0, { mediaTime });
      await tick();
    },
    restore() {
      globalThis.document = originalDocument;
      globalThis.FileReader = originalReader;
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    },
  };
}

test("video capture uses actual presented timestamps and never paints on seeked alone", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    await tick();
    assert.deepEqual(fixture.drawn, []);
    const times = [0.16, 0.56, 0.92, 1.28, 1.68, 2.04, 2.4, 2.8];
    const colors = ["red", "red", "red", "green", "green", "blue", "blue", "blue"];
    for (let index = 0; index < times.length; index++) await fixture.frame(times[index], colors[index]);
    const result = await work;
    assert.deepEqual(fixture.drawn, colors);
    assert.deepEqual(result.timestamps, times);
    assert.deepEqual(result.frames.map((url) => Buffer.from(url.split(",")[1], "base64").toString()), colors);
    assert.equal(fixture.callbacks.size, 0);
  } finally { fixture.restore(); }
});

test("an older presented frame is retried, not mislabeled as the current seek", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    await tick();
    await fixture.frame(0.16, "red");
    await fixture.frame(0.1, "black"); // Stale compositor frame from before the second seek.
    assert.deepEqual(fixture.drawn, ["red"]);
    await fixture.frame(0.56, "red");
    for (const [time, color] of [[0.92, "red"], [1.28, "green"], [1.68, "green"], [2.04, "blue"], [2.4, "blue"], [2.8, "blue"]]) {
      await fixture.frame(time, color);
    }
    const result = await work;
    assert.equal(result.frames.length, 8);
    assert.deepEqual(result.timestamps.slice(0, 2), [0.16, 0.56]);
    assert.ok(!fixture.drawn.includes("black"));
  } finally { fixture.restore(); }
});

test("paused decoder nudge rejects a frame advanced beyond the sample bin", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    const rejected = assert.rejects(work, /outside the sampled time range/);
    await tick();
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(fixture.plays, 1);
    await fixture.frame(0.5, "red"); // First bin ends at 0.375 seconds.
    await rejected;
    assert.deepEqual(fixture.drawn, []);
    assert.equal(fixture.video.paused, true);
    assert.equal(fixture.callbacks.size, 0);
  } finally { fixture.restore(); }
});

test("paused decoder nudge accepts an in-bin frame at its actual media time", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    await tick();
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(fixture.plays, 1);
    await fixture.frame(0.2, "red");
    for (const [time, color] of [[0.56, "red"], [0.92, "red"], [1.28, "green"], [1.68, "green"], [2.04, "blue"], [2.4, "blue"], [2.8, "blue"]]) {
      await fixture.frame(time, color);
    }
    const result = await work;
    assert.equal(result.timestamps[0], 0.2);
    assert.equal(fixture.drawn[0], "red");
  } finally { fixture.restore(); }
});

test("one-fps source yields ordered unique frames without fabricated timestamps", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    await tick();
    const samples = [[0, "red"], [0, "red"], [0, "red"], [1, "green"], [1, "green"], [2, "blue"], [2, "blue"], [2, "blue"]];
    for (let index = 0; index < samples.length; index++) {
      const [time, color] = samples[index];
      await fixture.frame(time, color);
      if ([1, 2, 4, 6, 7].includes(index)) await new Promise((resolve) => setTimeout(resolve, 510));
    }
    const result = await work;
    assert.deepEqual(result.timestamps, [0, 1, 2]);
    assert.deepEqual(fixture.drawn, ["red", "green", "blue"]);
    assert.equal(fixture.callbacks.size, 0);
  } finally { fixture.restore(); }
});

test("one-fps duplicate followed by the next frame outside its bin skips only that sample", async () => {
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    await tick();
    await fixture.frame(0, "red");
    await fixture.frame(0, "red"); // Seek at 0.562 s still presents the first frame.
    await fixture.frame(1, "green"); // Playback advances before the 500 ms retry expires; bin ends at 0.75.
    assert.deepEqual(fixture.drawn, ["red"]);
    await fixture.frame(0, "red"); // Next seek at 0.938 s can still present PTS 0.
    await fixture.frame(1, "green"); // PTS 1 is inside this later bin (end 1.125).
    await fixture.frame(1, "green");
    await fixture.frame(2, "blue"); // Skip the duplicate's out-of-bin advance at seek 1.313.
    await fixture.frame(1, "green");
    await fixture.frame(2, "blue"); // Same at seek 1.688.
    await fixture.frame(2, "blue"); // Seek 2.063 accepts PTS 2.
    for (let index = 0; index < 2; index++) {
      await fixture.frame(2, "blue");
      await new Promise((resolve) => setTimeout(resolve, 510));
    }
    const result = await work;
    assert.deepEqual(result.timestamps, [0, 1, 2]);
    assert.deepEqual(fixture.drawn, ["red", "green", "blue"]);
    assert.equal(fixture.callbacks.size, 0);
  } finally { fixture.restore(); }
});

test("missing presented-frame API fails closed and releases the local video URL", async () => {
  const fixture = browserVideo({ frameCallback: false });
  try {
    await assert.rejects(fixture.begin(), /requestVideoFrameCallback is required/);
    assert.deepEqual(fixture.drawn, []);
    assert.deepEqual(fixture.revoked, ["blob:local-video"]);
  } finally { fixture.restore(); }
});

test("abort during presented-frame wait cancels the callback and capture", async () => {
  const fixture = browserVideo();
  try {
    const controller = new AbortController();
    const work = fixture.begin(controller.signal);
    await tick();
    assert.equal(fixture.callbacks.size, 1);
    controller.abort();
    await assert.rejects(work, { name: "AbortError" });
    assert.deepEqual(fixture.drawn, []);
    assert.equal(fixture.callbacks.size, 0);
    assert.equal(fixture.video.paused, true);
  } finally { fixture.restore(); }
});

test("a decoder that never presents a frame times out without saving black", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const fixture = browserVideo();
  try {
    const work = fixture.begin();
    const rejected = assert.rejects(work, /not ready after seeking/);
    await tick();
    context.mock.timers.tick(5000);
    await rejected;
    assert.deepEqual(fixture.drawn, []);
    assert.equal(fixture.callbacks.size, 0);
    assert.equal(fixture.video.paused, true);
  } finally { fixture.restore(); }
});
