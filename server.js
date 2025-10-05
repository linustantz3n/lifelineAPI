import express from "express";
import * as ort from "onnxruntime-node";
import fs from "node:fs/promises";

/* ===== Config ===== */
const {
  PORT = 8080,
  MODEL_R2_ONNX = "https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/model_quantized.onnx",
  VOCAB_URL     = "https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/vocab.txt",
  THRESHOLDS_URL= "https://pub-f924fe6854dd4aae99dc7bff4b06ae5d.r2.dev/thresholds.json",      // optional
  TEMPERATURE = "6.0",   // higher = less confident predictions (reduce overconfidence)
  INTRA_OP_THREADS = "2"
} = process.env;

// LOCK to your model’s spec
const INT_DTYPE = "int64";
const SEQ_LEN   = Number(process.env.MAX_LEN ?? 96);
 // BigInt64Array, per your model

const app = express();
app.use(express.json({ limit: "1mb" }));

/* ===== Utils ===== */
async function fetchToTmp(url, name) {
  const path = `/tmp/${name}`;
  try { await fs.access(path); return path; } catch {}
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch failed ${url}: ${r.status} ${r.statusText}`);
  const buf = Buffer.from(await r.arrayBuffer());
  await fs.writeFile(path, buf);
  return path;
}

/* ===== Minimal WordPiece (pure JS) ===== */
function normalize(text) {
  return text
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ").trim();
}
function whitespaceTokenize(text) { return text ? text.split(" ") : []; }

function loadVocabFromText(vocabText) {
  const map = new Map();
  const lines = vocabText.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const tok = lines[i].trim();
    if (tok) map.set(tok, i);
  }
  return map;
}
function wordpiece(tokens, vocab, unk = "[UNK]", maxCharsPerWord = 100) {
  const out = [];
  for (const token of tokens) {
    if (token.length > maxCharsPerWord) { out.push(unk); continue; }
    let start = 0;
    const sub = [];
    while (start < token.length) {
      let end = token.length;
      let cur = null;
      while (start < end) {
        let s = token.slice(start, end);
        if (start > 0) s = "##" + s;
        if (vocab.has(s)) { cur = s; break; }
        end -= 1;
      }
      if (cur == null) { sub.length = 0; sub.push(unk); break; }
      sub.push(cur);
      start = end;
    }
    out.push(...sub);
  }
  return out;
}
function encodeToIds(text, vocab, L) {
  const CLS="[CLS]", SEP="[SEP]", PAD="[PAD]", UNK="[UNK]";
  const n = normalize(text);
  const basic = whitespaceTokenize(n);
  const wp = wordpiece(basic, vocab, UNK);
  const pieces = [CLS, ...wp.slice(0, L - 2), SEP];

  const unkId = vocab.get(UNK), padId = vocab.get(PAD);
  const ids = pieces.map(t => vocab.get(t) ?? unkId);
  const mask = Array(ids.length).fill(1);
  while (ids.length < L) { ids.push(padId); mask.push(0); }
  return { ids, mask };
}

/* ===== Globals ===== */
let session, vocabMap, thresholds;

/* ===== Init ===== */
async function init() {
  if (!MODEL_R2_ONNX || !VOCAB_URL) {
    throw new Error("Missing MODEL_R2_ONNX or VOCAB_URL env vars");
  }

  const [onnxPath, vocabPath] = await Promise.all([
    fetchToTmp(MODEL_R2_ONNX, "model.onnx"),
    fetchToTmp(VOCAB_URL, "vocab.txt"),
  ]);

  thresholds = THRESHOLDS_URL
    ? await fetch(THRESHOLDS_URL).then(r => r.json()).catch(err => { console.error("Failed to load thresholds:", err); return null; })
    : null;
  console.log("Loaded thresholds:", JSON.stringify(thresholds, null, 2));

  const vocabText = await fs.readFile(vocabPath, "utf8");
  vocabMap = loadVocabFromText(vocabText);

  session = await ort.InferenceSession.create(onnxPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
    intraOpNumThreads: Number(INTRA_OP_THREADS),
    interOpNumThreads: 1,
  });

  // Warm-up EXACT spec: int64 [1,96] for all three inputs
  const L = SEQ_LEN;
  const ones  = new BigInt64Array(L).fill(1n);
  const zeros = new BigInt64Array(L).fill(0n);
  await session.run({
    input_ids:      new ort.Tensor(INT_DTYPE, ones,  [1, L]),
    attention_mask: new ort.Tensor(INT_DTYPE, ones,  [1, L]),
    token_type_ids: new ort.Tensor(INT_DTYPE, zeros, [1, L]),
  });

  console.log("✅ Inference ready (locked)", { SEQ_LEN, INT_DTYPE, inputs: session.inputNames, outputs: session.outputNames });
}

/* ===== Routes ===== */
app.get("/health", (_req, res) => res.json({ ok: true, ready: !!session }));

// optional: debug how text is encoded
app.post("/debug-encode", async (req, res) => {
  const { text } = req.body || {};
  if (!text || typeof text !== "string") return res.status(400).json({ error: "Provide { text }" });
  const L = SEQ_LEN;
  const { ids, mask } = encodeToIds(text, vocabMap, L);
  return res.json({
    L,
    first_16_ids: ids.slice(0,16),
    first_16_mask: mask.slice(0,16),
    last_16_ids: ids.slice(-16),
    last_16_mask: mask.slice(-16),
  });
});

app.post("/classify", async (req, res) => {
  try {
    if (!session || !vocabMap) return res.status(503).json({ error: "Model not ready" });

    const { text } = req.body || {};
    if (!text || typeof text !== "string") return res.status(400).json({ error: "Provide { text }" });

    const L = SEQ_LEN;

    // Build EXACT inputs your model expects
    const { ids: idsArr0, mask: maskArr0 } = encodeToIds(text, vocabMap, L);
    const idsArr  = idsArr0.slice(0, L);
    const maskArr = maskArr0.slice(0, L);
    while (idsArr.length  < L) idsArr.push(vocabMap.get("[PAD]"));
    while (maskArr.length < L) maskArr.push(0);

    const idsT   = new BigInt64Array(idsArr.map(BigInt));
    const maskT  = new BigInt64Array(maskArr.map(BigInt));
    const typesT = new BigInt64Array(L).fill(0n);

    const outputs = await session.run({
      input_ids:      new ort.Tensor(INT_DTYPE, idsT,   [1, L]),
      attention_mask: new ort.Tensor(INT_DTYPE, maskT,  [1, L]),
      token_type_ids: new ort.Tensor(INT_DTYPE, typesT, [1, L]),
    });

    const tensor = outputs.logits ?? outputs.output ?? outputs[Object.keys(outputs)[0]];
    if (!tensor) return res.status(500).json({ error: "No logits in outputs" });

    const logits = Array.from(tensor.data);

    // Apply temperature scaling to reduce overconfidence
    const temp = Number(TEMPERATURE);
    const scaledLogits = logits.map(x => x / temp);

    // Apply sigmoid for multi-label classification
    const sigmoid = x => 1 / (1 + Math.exp(-x));
    const probs = scaledLogits.map(sigmoid);

    const labels = ["CPR_NEEDED","SEVERE_BLEEDING","CHOKING","SEIZURE","BURN"];
    const paired = labels.map((name,i)=>({ name, p: probs[i] ?? 0 })).sort((a,b)=>b.p-a.p);

    const top = paired[0];
    // thresholds JSON is { "CPR_NEEDED": 0.65, "SEIZURE": 0.7, ... }
    const classMin = thresholds?.[top.name] ?? 0.50;
    const accept = top.p >= classMin && !(Number.isNaN(top.p));
    const prediction = accept ? top.name : "UNSURE";

    res.json({ ok: true, prediction, confidence: top.p, top_k: paired.slice(0,3) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || "internal error" });
  }
});

/* ===== Boot ===== */
init().then(() => {
  app.listen(Number(PORT), () => console.log(`Listening on :${PORT}`));
}).catch(err => {
  console.error("❌ Init failed:", err);
  process.exit(1);
});
