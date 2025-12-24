import { WebPDFLoader } from "https://esm.sh/@langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/langchain/text_splitter";
import { MemoryVectorStore } from "https://esm.sh/langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "https://esm.sh/@langchain/community/embeddings/hf_transformers";
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// DOM Elements
const pdfUpload = document.getElementById('pdf-upload');
const indexStatus = document.getElementById('index-status');
const btnLoadLlm = document.getElementById('btn-load-llm');
const llmStatus = document.getElementById('llm-status');
const llmProgress = document.getElementById('llm-progress');
const progressContainer = document.getElementById('progress-container');
const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');

let vectorStore = null;
let engine = null;
const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

// 1. EMBEDDINGS & PDF LOGIC
const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2", // Small 45MB model for browser
});

pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    indexStatus.innerText = "⏳ Reading PDF & Creating Chunks...";
    try {
        const loader = new WebPDFLoader(file);
        const docs = await loader.load();

        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
        const splitDocs = await splitter.splitDocuments(docs);

        indexStatus.innerText = "⏳ Generating Math Vectors...";
        vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
        
        indexStatus.innerText = "✅ PDF Ready in RAM!";
        checkReadyState();
    } catch (err) {
        indexStatus.innerText = "❌ Error: " + err.message;
    }
});

// 2. LLM ENGINE LOGIC
btnLoadLlm.addEventListener('click', async () => {
    btnLoadLlm.disabled = true;
    progressContainer.style.display = 'block';

    const initProgressCallback = (report) => {
        const p = Math.round(report.progress * 100);
        llmProgress.style.width = p + "%";
        llmProgress.innerText = p + "%";
        llmStatus.innerText = report.text;
    };

    try {
        engine = await webllm.CreateMLCEngine(selectedModel, { initProgressCallback });
        llmStatus.innerText = "✅ AI Engine Active!";
        checkReadyState();
    } catch (err) {
        llmStatus.innerText = "❌ WebGPU Error: " + err.message;
    }
});

// 3. RAG EXECUTION
btnSend.addEventListener('click', async () => {
    const question = userInput.value.trim();
    if (!question) return;

    appendMsg("user", question);
    userInput.value = "";
    
    // RETRIEVAL
    const relatedDocs = await vectorStore.similaritySearch(question, 2);
    const context = relatedDocs.map(d => d.pageContent).join("\n---\n");

    // GENERATION
    const messages = [
        { role: "system", content: "You are a helpful AI. Answer the user's question using only the provided Context. If not in context, say you don't know." },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` }
    ];

    const aiMsgDiv = appendMsg("ai", "...");
    
    // Stream response
    const chunks = await engine.chat.completions.create({ messages, stream: true });
    let fullText = "";
    for await (const chunk of chunks) {
        fullText += chunk.choices[0]?.delta.content || "";
        aiMsgDiv.innerText = fullText;
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});

function checkReadyState() {
    if (vectorStore && engine) {
        userInput.disabled = false;
        btnSend.disabled = false;
        chatWindow.innerHTML = '<div class="text-center text-success mt-5">Ready! Ask me anything about your PDF.</div>';
    }
}

function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role === 'user' ? 'user-msg' : 'ai-msg'}`;
    div.innerHTML = `<strong>${role.toUpperCase()}:</strong> <p class="mb-0">${text}</p>`;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.querySelector('p');
}