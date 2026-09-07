import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { getDocumentProxy, extractText } from "unpdf";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
 
dotenv.config();
 
const __dirname = path.dirname(fileURLToPath(import.meta.url));
 
/**
 * Extracts text page-by-page from a PDF using unpdf, instead of
 * LangChain's PDFLoader (wraps `pdf-parse`) or raw pdfjs-dist.
 *
 * Why not pdf-parse: it has a packaging bug that throws
 * ERR_PACKAGE_PATH_NOT_EXPORTED specifically under serverless bundlers
 * like Vercel's.
 *
 * Why not raw pdfjs-dist: it normally offloads parsing to a separate
 * "worker" file. Vercel's bundler only includes files it can see via
 * static imports — it doesn't know to include that worker file, since
 * pdfjs-dist loads it dynamically at runtime. Result: "Setting up fake
 * worker failed" in production, despite working fine locally.
 *
 * unpdf avoids both problems — it's built specifically for edge/
 * serverless runtimes, with no separate worker file to lose track of.
 */
async function loadPdfPages(filePath) {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await getDocumentProxy(data);
  const { text: pagesText, totalPages } = await extractText(pdf, { mergePages: false });
 
  const documents = [];
  for (let i = 0; i < totalPages; i++) {
    documents.push(
      new Document({
        pageContent: pagesText[i],
        metadata: { source: filePath, loc: { pageNumber: i + 1 } },
      })
    );
  }
  return documents;
}
 
const app = express();
app.use(cors());
app.use(express.json());
 
/**
 * Google's free-tier Gemini API has a fairly low per-minute rate limit,
 * especially for the embedding model. Rapid testing (or real concurrent
 * users later) can trip a 429 even though nothing is actually wrong.
 *
 * This wraps any async call (embedding or chat) with automatic retries
 * using exponential backoff — wait 1s, then 2s, then 4s — before giving
 * up. Smooths out normal free-tier bursts without you needing to
 * manually wait and retry every time.
 */
async function retryWithBackoff(fn, { retries = 5, baseDelayMs = 3000 } = {}) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const isRateLimit = err?.status === 429 || err?.message?.includes("429");
      const isLastAttempt = attempt === retries;
 
      if (!isRateLimit || isLastAttempt) throw err;
 
      const delay = baseDelayMs * 2 ** attempt;
      console.warn(`Rate limited (attempt ${attempt + 1}/${retries + 1}) — retrying in ${delay}ms…`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}
 
const PORT = process.env.PORT || 3000;
 
if (!process.env.GEMINI_API_KEY) {
  console.error("Missing GEMINI_API_KEY in .env — copy .env.example to .env and add your key.");
  process.exit(1);
}
 
// --- Day 2 chain (unchanged) — plain chat, no document grounding ---
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.5-flash",
  temperature: 0.3,
});
const plainPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant. Answer clearly and concisely."],
  ["human", "{question}"],
]);
const plainChain = plainPromptTemplate.pipe(model).pipe(new StringOutputParser());
 
app.post("/chat", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Request body must include a 'question' string." });
    }
    const answer = await retryWithBackoff(() => plainChain.invoke({ question }));
    res.json({ question, answer });
  } catch (err) {
    console.error("Error running chain:", err);
    res.status(500).json({ error: "Something went wrong running the chain." });
  }
});
 
// --- Day 3 upload plumbing ---
// Uses os.tmpdir() instead of a local ./uploads folder — Vercel's
// filesystem is read-only except for /tmp, so this path works both on
// Vercel and locally (os.tmpdir() resolves to a normal temp folder on
// Windows/Mac/Linux too).
const uploadsDir = path.join(os.tmpdir(), "askmydocs-uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
 
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});
const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    // MIME type detection for PDFs is unreliable across OS/browsers —
    // checking the extension as a fallback is more dependable.
    const isPdfExtension = file.originalname.toLowerCase().endsWith(".pdf");
    const isPdfMimetype = file.mimetype === "application/pdf";
    if (!isPdfExtension && !isPdfMimetype) {
      return cb(new Error("Only PDF files are allowed."));
    }
    cb(null, true);
  },
  limits: { fileSize: 20 * 1024 * 1024 },
});
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
 
// --- Day 4 embeddings model (unchanged) ---
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
  outputDimensionality: 768,
});
 
// --- Day 5 vector store (unchanged) ---
let vectorStore = null;
 
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded. Field name must be 'file'." });
    }
 
    const pageDocuments = await loadPdfPages(req.file.path);
    const chunks = await textSplitter.splitDocuments(pageDocuments);
 
    if (!vectorStore) {
      vectorStore = await retryWithBackoff(() => MemoryVectorStore.fromDocuments(chunks, embeddings));
    } else {
      await retryWithBackoff(() => vectorStore.addDocuments(chunks));
    }
 
    res.json({
      filename: req.file.originalname,
      totalPages: pageDocuments.length,
      totalChunks: chunks.length,
      message: "Chunks embedded and added to the vector store. Try POST /ask to query them.",
    });
  } catch (err) {
    console.error("Error processing PDF:", err);
    res.status(500).json({ error: "Failed to process PDF.", details: err.message });
  }
});
 
// Kept from Day 5 — still useful for inspecting raw retrieval without
// involving the LLM, e.g. while debugging chunkSize/chunkOverlap choices.
app.post("/search", async (req, res) => {
  try {
    const { question, k } = req.body;
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Request body must include a 'question' string." });
    }
    if (!vectorStore) {
      return res.status(400).json({ error: "No documents uploaded yet. POST /upload a PDF first." });
    }
    const topK = typeof k === "number" && k > 0 ? k : 3;
    const results = await retryWithBackoff(() => vectorStore.similaritySearchWithScore(question, topK));
    res.json({
      question,
      k: topK,
      results: results.map(([doc, score]) => ({ score, content: doc.pageContent, metadata: doc.metadata })),
    });
  } catch (err) {
    console.error("Error searching vector store:", err);
    res.status(500).json({ error: "Search failed.", details: err.message });
  }
});
 
// --- Day 6: The full RAG chain ---
 
/**
 * RAG prompt template — two placeholders instead of Day 2's one.
 *
 * The system message is doing the real work here: it's the instruction
 * that keeps the model grounded in the retrieved context instead of
 * falling back on its own training knowledge. This is what your original
 * roadmap called "RAG evaluation" — without this line, ask it something
 * the PDF doesn't cover and Gemini will happily make up a plausible
 * answer anyway.
 */
const ragPromptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful assistant that answers questions using ONLY the provided context from an uploaded document.
 
Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say clearly: "The document doesn't cover this." Do not guess or make up an answer.
- Keep answers concise and directly responsive to the question.
- Use markdown formatting where it helps readability: bullet points for lists of items (like skills, features, steps), numbered lists for sequences, and **bold** for key terms. Don't force structure onto answers that are naturally a sentence or two.
- Use the conversation history to resolve references like "it", "that", or "give me an example" — these refer back to what was just discussed. Still answer only from the context provided, not from memory of what you said before.
 
Context:
{context}`,
  ],
  new MessagesPlaceholder("history"),
  ["human", "{question}"],
]);
 
const ragChain = ragPromptTemplate.pipe(model).pipe(new StringOutputParser());
 
/**
 * Formats retrieved chunks into a single context string for the prompt.
 * Includes page numbers inline so the model CAN mention them if useful,
 * and so you can see in the raw prompt exactly what it was given.
 */
function formatContext(retrievedDocs) {
  return retrievedDocs
    .map(([doc, score], i) => {
      const page = doc.metadata?.loc?.pageNumber ?? "unknown";
      return `[Excerpt ${i + 1} — page ${page}]\n${doc.pageContent}`;
    })
    .join("\n\n");
}
 
/**
 * Converts the plain { role, content } history the frontend sends into
 * real LangChain message objects the prompt template can slot in via
 * MessagesPlaceholder. Capped to the last 6 messages (3 back-and-forth
 * turns) — enough for "give me an example" to resolve "it" correctly,
 * without letting the prompt grow unbounded as a conversation gets long.
 */
function formatHistory(rawHistory) {
  if (!Array.isArray(rawHistory)) return [];
  return rawHistory.slice(-6).map((m) =>
    m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
  );
}
 
/**
 * POST /ask
 * The full pipeline in one endpoint:
 *   1. Retrieve top-k chunks for the question (Day 5's retriever)
 *   2. Format them into {context}
 *   3. Run {context, history, question} through the RAG chain
 *   4. Return the answer AND the sources it was grounded in
 *
 * This replaces what the frontend was previously stitching together by
 * calling /search then /chat separately — that logic now lives here,
 * server-side, where it belongs.
 */
app.post("/ask", async (req, res) => {
  try {
    const { question, k, history } = req.body;
 
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Request body must include a 'question' string." });
    }
 
    if (!vectorStore) {
      return res.status(400).json({ error: "No documents uploaded yet. POST /upload a PDF first." });
    }
 
    const topK = typeof k === "number" && k > 0 ? k : 4;
 
    // 1. Retrieve
    const retrieved = await retryWithBackoff(() => vectorStore.similaritySearchWithScore(question, topK));
 
    // 2. Format context + history
    const context = formatContext(retrieved);
    const formattedHistory = formatHistory(history);
 
    // 3. Generate, grounded in context AND aware of prior turns
    const answer = await retryWithBackoff(() =>
      ragChain.invoke({ context, question, history: formattedHistory })
    );
 
    // 4. Return answer + sources together
    res.json({
      question,
      answer,
      sources: retrieved.map(([doc, score]) => ({
        score,
        content: doc.pageContent,
        metadata: doc.metadata,
      })),
    });
  } catch (err) {
    console.error("Error running RAG chain:", err);
    res.status(500).json({ error: "Failed to answer question.", details: err.message });
  }
});
 
app.get("/", (req, res) => {
  res.send("AskMyDocs Day 7 server is running. POST /upload, POST /ask, POST /search, or POST /chat.");
});
 
// Vercel's Node runtime imports this file and calls the exported app
// directly per-request — it does NOT run app.listen(). Locally (and on
// Render/Railway/etc), there's no VERCEL env var, so this still starts
// a normal persistent server exactly as before.
if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`Day 7 server listening on http://localhost:${PORT}`);
  });
}
 
export default app;
 