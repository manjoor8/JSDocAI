import { WebPDFLoader } from "https://esm.sh/@langchain/community@0.3.20/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/@langchain/textsplitters@0.1.0";
import { MemoryVectorStore } from "https://esm.sh/langchain@0.3.8/vectorstores/memory";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.46";

// Configure Transformers.js to work in browser environments correctly
env.allowLocalModels = false;
env.useBrowserCache = true;

/**
 * Custom Embedding class to bypass LangChain's env.js bug.
 * MemoryVectorStore requires an object with these two methods.
 */
class SimpleEmbeddings {
    constructor() {
        this.pipe = null;
    }
    async getPipe() {
        if (!this.pipe) {
            this.pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        }
        return this.pipe;
    }
    async embedDocuments(texts) {
        const pipe = await this.getPipe();
        return Promise.all(texts.map(t => this.embedQuery(t)));
    }
    async embedQuery(text) {
        const pipe = await this.getPipe();
        const result = await pipe(text, { pooling: 'mean', normalize: true });
        return Array.from(result.data);
    }
}

// UI Elements
const logWin = document.getElementById('event-log');
const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const aiStatus = document.getElementById('ai-status');
const aiBar = document.getElementById('ai-bar');

const modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const embeddings = new SimpleEmbeddings();
let vectorStore, engine;

// Helper: UI Logging
function log(msg) {
    const div = document.createElement('div');
    div.innerText = `> ${msg}`;
    logWin.appendChild(div);
    logWin.scrollTop = logWin.scrollHeight;
    console.log(msg);
}

// Helper: Chat Messages
function appendMsg(role, text) {
    if(chatWindow.innerText.includes("Upload a PDF")) chatWindow.innerHTML = "";
    const div = document.createElement('div');
    div.className = role === 'User' ? 'user-msg p-2 mb-2 bg-light rounded' : 'ai-msg p-2 mb-2 border rounded';
    div.innerHTML = `<strong>${role}:</strong> <p class="mb-0">${text}</p>`;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.querySelector('p');
}

function checkReady() {
    if (vectorStore && engine) {
        userInput.disabled = false;
        sendBtn.disabled = false;
        log("System ready. You can now chat with your PDF.");
    }
}

// --- LOGIC ---

// 1. PDF Loading & Indexing
document.getElementById('pdf-upload').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    log("File detected. Reading PDF...");
    try {
        const loader = new WebPDFLoader(file);
        const docs = await loader.load();
        log(`Loaded ${docs.length} pages.`);

        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
        const splitDocs = await splitter.splitDocuments(docs);
        log(`Created ${splitDocs.length} chunks. Generating vectors...`);

        vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
        log("✅ PDF Indexed in local memory.");
        checkReady();
    } catch (err) {
        log(`❌ PDF Error: ${err.message}`);
        alert(`PDF Error: ${err.message}`);
    }
};

// 2. AI Initialization
document.getElementById('load-ai').onclick = async () => {
    if (!window.crossOriginIsolated) {
        alert("Security Error: Please refresh the page twice to enable Cross-Origin Isolation.");
        return;
    }

    document.querySelector('.progress').style.display = 'flex';
    log("Initializing AI Engine (this may take 1-2 mins on first run)...");

    try {
        engine = await webllm.CreateWebWorkerMLCEngine(
            new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }),
            modelId,
            { initProgressCallback: (p) => {
                aiBar.style.width = (p.progress * 100) + "%";
                aiStatus.innerText = p.text;
                if(p.text.includes("Finish")) log("✅ AI Model cached.");
            }}
        );
        log("✅ AI Engine Active.");
        checkReady();
    } catch (err) {
        log(`❌ AI Error: ${err.message}`);
        alert(`WebGPU Error: ${err.message}. Check if your browser supports WebGPU.`);
    }
};

// 3. RAG Chat Logic
sendBtn.onclick = async () => {
    const question = userInput.value.trim();
    if (!question) return;

    userInput.value = "";
    appendMsg("User", question);

    log("Searching document context...");
    try {
        const relatedDocs = await vectorStore.similaritySearch(question, 2);
        const context = relatedDocs.map(d => d.pageContent).join("\n---\n");
        log(`Found ${relatedDocs.length} relevant sections.`);

        const messages = [
            { role: "system", content: "You are a helpful assistant. Answer based ONLY on the context below." },
            { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` }
        ];

        const aiPara = appendMsg("AI", "Thinking...");
        const chunks = await engine.chat.completions.create({ messages, stream: true });
        
        let fullText = "";
        for await (const chunk of chunks) {
            fullText += chunk.choices[0]?.delta.content || "";
            aiPara.innerText = fullText;
        }
        log("AI response finished.");
    } catch (err) {
        log(`❌ Chat Error: ${err.message}`);
    }
};