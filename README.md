# AskMyDocs — Day 1: Basic Gemini Chatbot 
 Live Demo :  https://askmy-docs-frontend.vercel.app/
 
No RAG yet — just wiring up Node.js → Gemini → Answer. This is the foundation
you'll build the retrieval pipeline on top of in later days.
 
```
User Question
      ↓
Node.js
      ↓
Gemini
      ↓
Answer
```
 
## Setup
 
1. Install dependencies:
```bash
   npm install
```
 
2. Get a Gemini API key from https://aistudio.google.com/apikey
3. Copy the env file and add your key:
```bash
   cp .env.example .env
   # then edit .env and paste your key
```
 
4. Start the server:
```bash
   npm start
```
   You should see: `Day 1 server listening on http://localhost:3000`
 
## Test it
 
### With curl
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a JavaScript closure?"}'
```
 
### With Postman
- Method: `POST`
- URL: `http://localhost:3000/chat`
- Body → raw → JSON:
```json
  { "question": "What is a JavaScript closure?" }
```
 
Expected response:
```json
{
  "question": "What is a JavaScript closure?",
  "answer": "A closure is..."
}
```
 
## What to notice / experiment with
 
- Change the prompt and see how Gemini responds with zero context — this is
  your baseline. Once you add RAG later, compare answers with and without
  document context to *see* retrieval actually changing the output.
- Try asking something Gemini can't know (e.g. "What's on page 12 of my PDF?")
  — it'll hallucinate or admit it doesn't have the file. That's the gap RAG
  is going to fill.
- Look at `result.response.text()` — later you'll swap `question` for a
  prompt template that also injects retrieved chunks.


## Day 2: LangChain.js — Prompt Templates & Chains
 
Same `/chat` endpoint as Day 1, same Gemini model underneath — but now
routed through LangChain's `PromptTemplate` and `.pipe()` chain instead of
a raw SDK call. The behavior looks identical from Postman; what changed is
the *structure* underneath, which is what you'll extend with retrieval
starting Day 3–5.
 
## Setup
 
```bash
npm install
cp .env.example .env
# paste your GEMINI_API_KEY into .env
npm start
```
 
## Test it (Postman)
- POST `http://localhost:3000/chat`
- Body → raw → JSON:
```json
  { "question": "What is a JavaScript closure?" }
```
 
Should return the same shape as Day 1:
```json
{
  "question": "What is a JavaScript closure?",
  "answer": "..."
}
```
 
## What actually changed vs Day 1
 
| Day 1 | Day 2 |
|---|---|
| `genAI.getGenerativeModel(...)` (raw SDK) | `new ChatGoogleGenerativeAI(...)` (LangChain wrapper) |
| `model.generateContent(question)` | `chain.invoke({ question })` |
| Hardcoded prompt string | `ChatPromptTemplate` with `{question}` placeholder |
| Manually call `.text()` on the response | `StringOutputParser` does it inside the chain |
 
## Things to experiment with
 
1. **Change the system message.** Try:
```js
   "You are a sarcastic assistant who answers in one sentence."
```
   Restart the server and re-test — notice the *chain code* (prompt → model
   → parser) never changed, only the template content did. That
   separation is the whole point of prompt templates.
 
2. **Add a second placeholder.** Try adding a `{tone}` variable:
```js
   ["system", "Answer in a {tone} tone."],
   ["human", "{question}"],
```
   and pass `chain.invoke({ question, tone: "playful" })`. This is exactly
   the mechanism you'll use later to inject `{context}` from retrieved
   document chunks.
 
3. **Log the intermediate steps.** Temporarily add:
```js
   const formattedPrompt = await promptTemplate.invoke({ question });
   console.log(formattedPrompt);
```
   before calling the chain, to see exactly what message array gets sent
   to Gemini. Understanding this shape now will make debugging the RAG
   prompt much easier later.
 
4. **Try `temperature: 0` vs `temperature: 1`** and compare answers to the
   same question — see how it affects consistency vs creativity.

# Day 3: PDF Loading & Document Splitting
 
Adds `POST /upload` alongside the existing `POST /chat`. Upload a PDF and
see exactly how it gets loaded and chunked — no embeddings or storage yet
(that's Day 4/5). This is purely about understanding the load → split step.
 
```
PDF
 ↓
Load (per-page text extraction)
 ↓
Split (small overlapping chunks)
```
 
## Setup
 
```bash
npm install
cp .env.example .env
# paste your GEMINI_API_KEY into .env
npm start
```
 
## Test it (Postman)
 
- Method: `POST`
- URL: `http://localhost:3000/upload`
- Body → **form-data** (not raw/JSON this time — file uploads use multipart form-data)
  - Key: `file` (change the type dropdown from "Text" to **"File"**)
  - Value: pick a PDF from your computer
You should get back something like:
```json
{
  "filename": "javascript_notes.pdf",
  "totalPages": 8,
  "totalChunks": 23,
  "sampleChunks": [
    {
      "content": "Closures in JavaScript...",
      "metadata": { "source": "...", "loc": { "pageNumber": 1 } },
      "length": 987
    }
  ]
}
```
 
`/chat` still works exactly like Day 2 — untouched.
 
## What to look at
 
1. **`totalPages` vs `totalChunks`** — a page's worth of text usually
   splits into multiple chunks. Compare these numbers for a dense
   technical PDF vs a sparse slide-style PDF.
2. **`sampleChunks[].metadata`** — notice `loc.pageNumber` survives the
   split. This is exactly what you'll use in Day 7 to cite "Page 12" in
   an answer.
3. **`length`** — should hover near `chunkSize` (1000) but rarely hit it
   exactly, since `RecursiveCharacterTextSplitter` prefers breaking at
   paragraph/sentence boundaries over cutting mid-word.
## Experiment: tune chunkSize and chunkOverlap
 
In `server.js`:
```js
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
```
 
Try re-uploading the same PDF with different values and compare
`totalChunks` and the sample content each time:
 
| chunkSize | chunkOverlap | Effect |
|---|---|---|
| 300 | 50 | Many small chunks — precise retrieval later, but each chunk has less surrounding context |
| 1000 | 200 | Balanced default — good starting point for most documents |
| 3000 | 300 | Fewer, larger chunks — more context per chunk, but retrieval gets less selective |
 
There's no universally "correct" setting — it depends on document
structure and how granular your questions will be. This is genuinely one
of the highest-leverage things to tune in a real RAG system.
 
## A note on file uploads
 
Uploaded PDFs are saved to `uploads/` on disk (via Multer) before being
read by `PDFLoader`. In a real production app you'd usually delete the
file after processing, or use temp storage — for this learning project,
keeping them on disk is fine and lets you inspect what was uploaded.
`.gitignore` already excludes the PDFs themselves from version control.

# Day 4: Embeddings
 
Two things added on top of Day 3:
1. `/upload` now embeds a couple of sample chunks so you can see real vectors
2. A new standalone `/embed-compare` endpoint to directly experiment with similarity
```
Text
 ↓
Embedding Model (text-embedding-004)
 ↓
Vector (768 numbers)
```
 
## Setup
 
```bash
npm install
cp .env.example .env
# paste your GEMINI_API_KEY into .env
npm start
```
 
## Test 1: `/upload` (same as Day 3, now with embeddings preview)
 
- POST `http://localhost:3000/upload`
- Body → form-data → key `file` (type: File) → pick a PDF
Response now includes:
```json
{
  "filename": "notes.pdf",
  "totalPages": 5,
  "totalChunks": 14,
  "embeddingPreview": [
    {
      "content": "Closures in JavaScript allow...",
      "metadata": {...},
      "vectorDimensions": 768,
      "vectorPreview": [0.021, -0.145, 0.782, 0.003, ...]
    }
  ]
}
```
 
`vectorDimensions: 768` confirms the embedding model turned that chunk of
text into a 768-number vector. We only show the first 8 numbers
(`vectorPreview`) — printing all 768 wouldn't be useful to look at.
 
## Test 2: `/embed-compare` — the important one
 
This is where similarity actually clicks. POST to
`http://localhost:3000/embed-compare` with raw JSON:
 
```json
{
  "textA": "JavaScript closures allow functions to remember their scope",
  "textB": "A closure lets a function access variables from its outer scope"
}
```
 
Expect a **high** similarity score (likely 0.85+) — different words,
same meaning.
 
Now try:
```json
{
  "textA": "JavaScript closures allow functions to remember their scope",
  "textB": "The weather today is sunny with a chance of rain"
}
```
 
Expect a **much lower** score — unrelated meaning.
 
## Experiments to really understand this
 
1. **Synonyms vs unrelated**: Compare `"car"` vs `"automobile"` (should
   be high), then `"car"` vs `"banana"` (should be low).
2. **Same words, different order**: Compare `"the cat chased the dog"`
   vs `"the dog chased the cat"` — these have opposite meanings but share
   every word. See how close/far apart the score is; it tells you
   something about how much embeddings capture word order vs just topic.
3. **Question vs answer**: Compare a question like `"What is a closure?"`
   against an actual definition sentence. This is literally the mechanic
   retrieval will use in Day 5 — embedding the user's *question* and
   finding chunks whose embeddings are closest to it.
4. **Length sensitivity**: Compare a short phrase against a long
   paragraph that's topically related but much longer. Notice similarity
   still works reasonably well — embeddings aren't thrown off by length
   the way raw keyword matching might be.
## Why we're not storing every chunk's embedding yet
 
You could technically embed every chunk from the PDF right now, but
there's nowhere sensible to put hundreds of vectors yet — that's exactly
what a **vector database** is for, and it's Day 5. Today was about
understanding embeddings in isolation before adding storage and search
on top.

 # Day 5: Vector Store + Retriever
 
Uploaded PDFs are now stored (embeddings + text + metadata) in a
`MemoryVectorStore` that persists for the life of the server. A new
`POST /search` endpoint lets you query it directly — this is the
retriever, tested in isolation before Day 6 connects it to an LLM.
 
```
PDF → Chunks → Embeddings → Vector Store
                                 ↑
User Question → embed → Similarity Search → Relevant Chunks
```
 
## Setup
 
```bash
npm install
cp .env.example .env
# paste your GEMINI_API_KEY into .env
npm start
```
 
## Test it
 
### 1. Upload a PDF
Same as Day 3/4 — POST `http://localhost:3000/upload`, form-data, key
`file`, type File.
 
Response now says the chunks were added to the vector store:
```json
{
  "filename": "javascript_notes.pdf",
  "totalPages": 8,
  "totalChunks": 23,
  "message": "Chunks embedded and added to the vector store. Try POST /search to query them."
}
```
 
### 2. Search it
POST `http://localhost:3000/search`
Body → raw → JSON:
```json
{ "question": "What is the difference between map and forEach?" }
```
 
Response:
```json
{
  "question": "What is the difference between map and forEach?",
  "k": 3,
  "results": [
    {
      "score": 0.83,
      "content": "map() creates a new array by applying...",
      "metadata": { "source": "...", "loc": { "pageNumber": 4 } }
    },
    { "score": 0.71, "content": "...", "metadata": {...} },
    { "score": 0.65, "content": "...", "metadata": {...} }
  ]
}
```
 
Results are sorted highest score first — these are your candidate chunks
for answering the question.
 
### 3. Try `k`
```json
{ "question": "What is a closure?", "k": 5 }
```
Returns the top 5 instead of the default 3.
 
## What to notice
 
1. **Score patterns.** Ask a question the PDF clearly answers — top result
   should score noticeably higher than the rest. Ask something unrelated
   to the PDF's content entirely — scores should all be mediocre/low, with
   no clear winner. This gap (or lack of one) is exactly what Day 6's
   "avoid hallucinating when the answer isn't in the doc" logic will lean on.
2. **Wording mismatch still works.** Ask using different words than the
   PDF uses (e.g. if the PDF says "higher-order function" and you ask
   "what's a function that takes another function as input") — retrieval
   should still surface the relevant chunk. That's the embeddings payoff
   from Day 4, now visible in a real search.
3. **Multiple uploads share one store.** Upload a second, unrelated PDF
   and search again — you may get chunks from *either* document. Read the
   comment above `vectorStore` in `server.js` for why, and how a real app
   would scope this per-document or per-user.

# Day 6: Connect Retrieval to the LLM (Full RAG Chain)
 
The full pipeline finally connects in one endpoint:
 
```
                  ┌──────────────┐
                  │    PDF       │
                  └──────┬───────┘
                         ↓
                    Split Text
                         ↓
                    Embeddings
                         ↓
                   Vector Store
                         ↑
                         │
User Question → Retriever
                         ↓
                  Relevant Context
                         ↓
                       LLM
                         ↓
                     Answer
```
 
`POST /ask` retrieves relevant chunks, builds a grounded prompt, calls
Gemini, and returns both the answer and the sources it used — all
server-side. `/search` and `/chat` are both still there for debugging
retrieval or testing plain chat in isolation.
 
## Setup
 
```bash
npm install
cp .env.example .env
# paste your GEMINI_API_KEY into .env
npm start
```
 
CORS is now enabled by default (`app.use(cors())`) so the React frontend
can talk to this server directly — no extra setup needed if you're using
the frontend from before.
 
## Test it
 
### 1. Upload a PDF
Same as before — `POST /upload`, form-data, key `file`.
 
### 2. Ask a grounded question
`POST http://localhost:3000/ask`
```json
{ "question": "What is the difference between map() and forEach()?" }
```
 
Response:
```json
{
  "question": "What is the difference between map() and forEach()?",
  "answer": "map() creates a new array by applying a function to each element, while forEach() just runs a function on each element without returning anything...",
  "sources": [
    { "score": 0.84, "content": "...", "metadata": { "loc": { "pageNumber": 4 } } },
    { "score": 0.79, "content": "...", "metadata": {...} }
  ]
}
```
 
### 3. Try the "avoid hallucination" test
Ask something the PDF clearly doesn't cover:
```json
{ "question": "Who invented JavaScript?" }
```
(unless your PDF happens to cover JS history). The answer should say
something like *"The document doesn't cover this"* instead of confidently
answering from Gemini's general knowledge. This is the exact system
prompt instruction doing its job — try removing it temporarily and asking
again to see the difference.
 
## What actually changed vs Day 5
 
| Day 5 | Day 6 |
|---|---|
| `/search` returns raw chunks only | `/ask` returns a generated answer + the chunks it used |
| Client had to build the prompt itself | `ragPromptTemplate` with `{context}` + `{question}` lives in the backend |
| No hallucination guardrail | System prompt explicitly instructs "don't guess" |
| Frontend called 2 endpoints per question | Frontend calls 1 endpoint (`/ask`) |
 
## The prompt template — the actual RAG "trick"
 
```js
const ragPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", `You are a helpful assistant that answers questions using ONLY
the provided context from an uploaded document.
 
Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say clearly:
  "The document doesn't cover this." Do not guess or make up an answer.
 
Context:
{context}`],
  ["human", "{question}"],
]);
```
 
Two placeholders now instead of Day 2's one. `{context}` gets filled with
the formatted retrieved chunks; `{question}` gets the user's actual
question. This is the entire mechanism that turns a generic chatbot into
a document-grounded one — no exotic library magic, just a carefully
written prompt plus retrieval feeding it real content.
 
## Experiment
 
1. **Loosen the system prompt.** Remove the "do not use outside knowledge"
   line and re-ask an out-of-scope question — watch the model start
   confidently hallucinating again.
2. **Change `k` (how many chunks get retrieved).** Try `k: 1` vs `k: 8` in
   your request body and compare answer quality — too few chunks can miss
   necessary context, too many can dilute the prompt with irrelevant text.
3. **Log the actual formatted context** sent to the model (temporarily
   `console.log(context)` inside `/ask`) to see exactly what Gemini
   received — useful for debugging weird or incomplete answers.
## Frontend
 
The frontend from before has been updated to call `/ask` directly instead
of manually stitching `/search` + `/chat` — one function (`askDocument`)
replaces the old `searchDocument` + `buildGroundedPrompt` + `askChat`
combination. No UI changes were needed since the response shape
(`{ answer, sources }`) was already what the UI expected.
 
## Next: Day 7
Conversation history (so "give me an example" understands "it" refers to
the previous answer) and cited sources are already partly here — next is
tightening both, plus a proper eval pass: systematically testing questions
that ARE and AREN'T in the document to confirm the guardrail holds up.

# Day 7: Conversation History + RAG Evaluation
 
Two things finish off the original roadmap today: the model now remembers
the conversation, and here's how to systematically test whether your RAG
pipeline is actually trustworthy — not just working on the happy path.
 
## What changed
 
`/ask` now accepts an optional `history` array alongside `question`:
```json
{
  "question": "Give me an example",
  "history": [
    { "role": "user", "content": "What is a JavaScript closure?" },
    { "role": "assistant", "content": "A closure is a function that remembers..." }
  ],
  "k": 4
}
```
 
The backend converts this into real `HumanMessage`/`AIMessage` objects and
slots them into the prompt via `MessagesPlaceholder("history")` — capped
to the last 6 messages (3 turns) so the prompt doesn't grow unbounded as
a conversation gets long.
 
The frontend already tracks all messages in React state — it just now
sends the relevant slice along with each new question. No new UI needed;
the "give me an example" pattern from your original roadmap now works
automatically.
 
## Setup
 
```bash
npm install
cp .env.example .env
npm start
```
 
Frontend: no new install needed, just pull the updated `App.jsx`.
 
## Test conversation history
 
1. Upload a PDF
2. Ask something with a clear follow-up potential:
```
   What is a JavaScript closure?
```
3. Then ask, without re-explaining the subject:
```
   Give me an example
```
   It should correctly answer about closures specifically, not ask "an
   example of what?" — that's history resolution working.
 
## RAG evaluation — testing the guardrail systematically
 
Your original roadmap called this out specifically: ask questions where
the answer exists in the document, and questions where it doesn't, and
confirm the bot doesn't confidently hallucinate on the second kind.
 
Do this as an actual pass, not just one or two ad-hoc tries:
 
**In-scope questions (answer should be grounded and correct):**
- Pick 3-5 questions you know the PDF directly answers
- Check the `sources` returned — do the retrieved chunks actually contain
  the answer, or did it get lucky with an unrelated chunk that happened
  to score reasonably?
**Out-of-scope questions (answer should be a clear refusal):**
- Ask something plausible-sounding but absent from the document (e.g. if
  it's a resume, ask about a skill/company not listed)
- Ask something wildly unrelated (weather, a different topic entirely)
- In both cases, confirm you get "The document doesn't cover this" (or
  similar) — not a confident, invented answer
**Borderline questions (the interesting middle ground):**
- Ask something the document only partially covers
- Good behavior: answer what's there, and note what's missing, rather
  than filling the gap with invented detail
If you find a case where it hallucinates despite the guardrail, that's
useful signal — it usually means either the system prompt needs
tightening, or `k` (how many chunks get retrieved) is too low/high for
that kind of question. This is genuinely what "evaluation" means in a
real RAG project — not a single test, but a deliberate sweep across these
three categories.
 
## Where this project stands now
 
You've built, end to end:
- PDF upload → text extraction → chunking → embeddings → vector storage
- A retriever that finds relevant chunks by meaning, not just keywords
- A full RAG chain: retrieval + prompt grounding + generation
- Source citations with page numbers and relevance scores
- Conversation memory for natural follow-ups
- A real React frontend with your own visual design
That's the complete original roadmap. From here, "finishing" is really
about polish and deployment, not new RAG concepts:
 
- **Deploy the backend** (Render, Railway, Fly.io are common free/cheap
  options for a Node/Express app)
- **Deploy the frontend** (Vercel or Netlify — both have simple Vite
  support)
- **Swap `MemoryVectorStore` for a persistent one** (Chroma, or a hosted
  vector DB) if you want uploads to survive a server restart
- **Add basic auth/session scoping** if more than one person will use it,
  so uploads don't collide in the shared in-memory store
None of those are required to call this project "done" — the RAG
fundamentals you set out to learn are all genuinely in place.
