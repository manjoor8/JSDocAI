import { WebPDFLoader } from "https://esm.sh/@langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/langchain/text_splitter";
import { MemoryVectorStore } from "https://esm.sh/langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "https://esm.sh/@langchain/community/embeddings/hf_transformers";
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// Immediate Check: Is the security patch working?
if (!window.crossOriginIsolated) {
    console.warn("Security patch not active yet. Please refresh the page twice.");
}

const logWin = document.getElementById('event-log');
const modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
let vectorStore, engine;

function log(msg) {
    const div = document.createElement('div');
    div.innerText = `> ${msg}`;
    logWin.appendChild(div);
    logWin.scrollTop = logWin.scrollHeight;
}

const embeddings = new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" });

// 1. PDF LOADING
document.getElementById('pdf-upload').onchange = async (e) => {
    log("File selection detected...");
    try {
        const file = e.target.files[0];
        if (!file) return;
        
        const loader = new WebPDFLoader(file);
        const docs = await loader.load();
        log("PDF parsed successfully.");

        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
        const splitDocs = await splitter.splitDocuments(docs);
        vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
        log("✅ Knowledge Base Indexed.");
        checkReady();
    } catch (err) {
        alert("CRITICAL ERROR: " + err.message);
        log("❌ Error: " + err.message);
    }
};

// 2. AI LOADING
document.getElementById('load-ai').onclick = async () => {
    log("Checking for WebGPU...");
    if (!navigator.gpu) {
        alert("Your browser does not support WebGPU. Use Chrome or Edge.");
        return;
    }

    log("Starting Worker...");
    try {
        // This creates a worker from the file 'worker.js' in your repo
        const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
        
        engine = await webllm.CreateWebWorkerMLCEngine(worker, modelId, {
            initProgressCallback: (p) => {
                document.getElementById('ai-bar').parentElement.style.display = 'block';
                document.getElementById('ai-bar').style.width = (p.progress * 100) + "%";
                document.getElementById('ai-status').innerText = p.text;
            }
        });
        log("✅ AI Engine Active.");
        checkReady();
    } catch (err) {
        alert("AI INITIALIZATION FAILED: " + err.message);
        log("❌ AI Error: " + err.message);
    }
};

function checkReady() {
    if (vectorStore && engine) {
        document.getElementById('user-input').disabled = false;
        document.getElementById('send-btn').disabled = false;
        log("System Ready. You can now ask questions!");
    }
}