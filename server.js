import express from "express";
import * as ort from "onnxruntime-node";
import { BertWordPieceTokenizer } from "@huggingface/tokenizers";
import fs from "node:fs/promises";

const {
  PORT = 8080,
  MODEL_R2_ONNX = 'https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/model_quantized.onnx',
  VOCAB_URL = 'https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/vocab.txt',           // e.g. https://pub-...r2.dev/vocab.txt
  THRESHOLDS_URL = 'https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/thresholds.json',
  API_KEY,             // callers must send x-api-key if set
  MAX_LEN = "64",
  INTRA_OP_THREADS = "2"
} = process.env;

const app = express();
app.use(express.json({ limit: "1mb" }));

function requireKey(req, res, next) {
  if (!API_KEY) return next();
  if (req.header("x-api-key") === API_KEY) return next();
  return res.status(401).json({ error: "Unauthorized" });
}

async function fetchToTmp(url, name) {
  const path = `/tmp/${name}`;
  try { await fs.access(path); return path; } catch {}
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch failed ${url}: ${r.status} ${r.statusText}`);
  const buf = Buffer.from(await r.arrayBuffer());
  await fs.writeFile(path, buf);
  return path;
}

let session, tokenizer, thresholds;

async function init() {
  if (!MODEL_R2_ONNX || !VOCAB_URL) {
    throw new Error("Missing MODEL_R2_ONNX or VOCAB_URL");
  }

  // Pull artifacts once
  const [onnxPath, vocabPath] = await Promise.all([
    fetchToTmp(MODEL_R2_ONNX, "model.onnx"),
    fetchToTmp(VOCAB_URL, "vocab.txt")
  ]);

  thresholds = THRESHOLDS_URL
    ? await fetch(THRESHOLDS_URL).then(r => r.json())
    : null;

  tokenizer = await BertWordPieceTokenizer.fromOptions({
    vocabFile: vocabPath, lowercase: true, stripAccents: true
  });

  session = await ort.InferenceSession.create(onnxPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
    intraOpNumThreads: Number(INTRA_OP_THREADS)
  });

  // Warm-up
  const L = Number(MAX_LEN);
  const ones  = new BigInt64Array(L).fill(1n);
  const zeros = new BigInt64Array(L).fill(0n);
  await session.run({
    input_ids:      new ort.Tensor("int64", ones,  [1, L]),
    attention_mask: new ort.Tensor("int64", ones,  [1, L]),
    token_type_ids: new ort.Tensor("int64", zeros, [1, L]),
  });

  console.log("✅ Inference ready");
}

app.get("/health", (_req, res) => res.json({ ok: true, ready: !!session }));

app.post("/classify", requireKey, async (req, res) => {
  try {
    if (!session || !tokenizer) return res.status(503).json({ error: "Model not ready" });

    const { text, maxLen } = req.body || {};
    if (!text || typeof text !== "string") return res.status(400).json({ error: "Provide { text }" });
    const L = Math.min(Math.max(Number(maxLen || MAX_LEN), 8), 256);

    const enc = await tokenizer.encode(text);
    const idsArr  = enc.ids.slice(0, L);
    const maskArr = Array(idsArr.length).fill(1);
    while (idsArr.length < L) { idsArr.push(0); maskArr.push(0); }

    const ids   = new BigInt64Array(idsArr.map(BigInt));
    const mask  = new BigInt64Array(maskArr.map(BigInt));
    const types = new BigInt64Array(L).fill(0n);

    const outputs = await session.run({
      input_ids:      new ort.Tensor("int64", ids,   [1, L]),
      attention_mask: new ort.Tensor("int64", mask,  [1, L]),
      token_type_ids: new ort.Tensor("int64", types, [1, L]),
    });

    const tensor = outputs.logits ?? outputs.output ?? outputs[Object.keys(outputs)[0]];
    if (!tensor) return res.status(500).json({ error: "No logits in outputs" });

    const logits = Array.from(tensor.data);
    const m = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - m));
    const sum  = exps.reduce((a,b)=>a+b, 0);
    const probs = exps.map(v => v/sum);

    const labels = (thresholds?.labels) ?? ["CPR_NEEDED","SEVERE_BLEEDING","CHOKING","SEIZURE","ALLERGIC_REACTION"];
    const paired = labels.map((name, i) => ({ name, p: probs[i] ?? 0 })).sort((a,b)=>b.p-a.p);

    const top = paired[0];
    const globalMin = thresholds?.min_confidence ?? 0.50;
    const perClass  = thresholds?.class_thresholds ?? {};
    const marginMin = thresholds?.margin_min ?? 0.0;
    const unsureBand= thresholds?.unsure_band ?? 0.0;

    const classMin  = perClass[top.name] ?? globalMin;
    const separated = (paired[0].p - (paired[1]?.p ?? 0)) >= marginMin;

    const accept = top.p >= classMin && separated && !(Number.isNaN(top.p));
    const prediction = accept ? top.name : "UNSURE";

    res.json({
      ok: true,
      prediction,
      confidence: top.p,
      top_k: paired.slice(0, 3),
      model_version: thresholds?.model_version ?? null,
      thresholds_version: thresholds?.version ?? null
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || "internal error" });
  }
});

init().then(() => {
  app.listen(Number(PORT), () => console.log(`Listening on :${PORT}`));
}).catch(err => {
  console.error("❌ Init failed:", err);
  process.exit(1);
});
