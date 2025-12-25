// Use browser-ready bundles to avoid __version__ export issues
import { WebPDFLoader } from "https://esm.sh/@langchain/community@0.3.20/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/@langchain/textsplitters@0.1.0";
import { MemoryVectorStore } from "https://esm.sh/langchain@0.3.8/vectorstores/memory";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.46";

/**
 * Custom Embedding class using the CDN ESM bundle directly (no global window dependency).
 * This avoids the env.js/__version__ export error seen with some bundlers.
 */
class SimpleEmbeddings {
    constructor() {
        this.pipePromise = null;
    }
    async getPipe() {
        if (!this.pipePromise) {
            this.pipePromise = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
        }
        return this.pipePromise;
    }
    async embedDocuments(texts) {
        return Promise.all(texts.map((t) => this.embedQuery(t)));
    }
    async embedQuery(text) {
        const pipe = await this.getPipe();
        const result = await pipe(text, { pooling: "mean", normalize: true });
        return Array.from(result.data);
    }
}

const logWin = document.getElementById("event-log");
const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const aiStatus = document.getElementById("ai-status");
const aiBar = document.getElementById("ai-bar");

const modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const embeddings = new SimpleEmbeddings();
let vectorStore, engine;

function log(msg) {
    const div = document.createElement("div");
    div.innerText = `> ${msg}`;
    logWin.appendChild(div);
    logWin.scrollTop = logWin.scrollHeight;
}

function appendMsg(role, text) {
    if (chatWindow.innerText.includes("Upload a PDF")) chatWindow.innerHTML = "";
    const div = document.createElement("div");
    div.className = role === "User" ? "user-msg p-2 mb-2 bg-light rounded" : "ai-msg p-2 mb-2 border rounded";
    div.innerHTML = `<strong>${role}:</strong> <p class="mb-0">${text}</p>`;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.querySelector("p");
}

function checkReady() {
    if (vectorStore && engine) {
        userInput.disabled = false;
        sendBtn.disabled = false;
        log("System ready.");
    }
}

document.getElementById("pdf-upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    log("Reading PDF...");
    try {
        const loader = new WebPDFLoader(file);
        const docs = await loader.load();
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
        const splitDocs = await splitter.splitDocuments(docs);
        vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
        log("ƒo. PDF Indexed.");
        checkReady();
    } catch (err) {
        log(`ƒ?O PDF Error: ${err.message}`);
    }
};

document.getElementById("load-ai").onclick = async () => {
    if (!window.crossOriginIsolated) {
        alert("Please refresh the page twice.");
        return;
    }
    document.querySelector(".progress").style.display = "flex";
    log("Initializing AI Engine...");
    try {
        engine = await webllm.CreateWebWorkerMLCEngine(
            new Worker(new URL("./worker.js", import.meta.url), { type: "module" }),
            modelId,
            {
                initProgressCallback: (p) => {
                    aiBar.style.width = p.progress * 100 + "%";
                    aiStatus.innerText = p.text;
                },
            }
        );
        log("ƒo. AI Engine Active.");
        checkReady();
    } catch (err) {
        log(`ƒ?O AI Error: ${err.message}`);
    }
};

sendBtn.onclick = async () => {
    const question = userInput.value;
    if (!question.trim()) return;
    if (!vectorStore || !engine) {
        log("System not ready. Load a PDF and initialize AI first.");
        return;
    }
    userInput.value = "";
    appendMsg("User", question);
    log("Searching context...");
    const relatedDocs = await vectorStore.similaritySearch(question, 2);
    const context = relatedDocs.map((d) => d.pageContent).join("\n---\n");
    const messages = [
        { role: "system", content: "Answer strictly based on context." },
        { role: "user", content: `Context: ${context}\n\nQuestion: ${question}` },
    ];
    const aiPara = appendMsg("AI", "...");
    const chunks = await engine.chat.completions.create({ messages, stream: true });
    let fullText = "";
    for await (const chunk of chunks) {
        fullText += chunk.choices[0]?.delta.content || "";
        aiPara.innerText = fullText;
    }
    log("Response finished.");
};
