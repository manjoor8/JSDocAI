import { WebPDFLoader } from "https://esm.sh/@langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/langchain/text_splitter";
import { MemoryVectorStore } from "https://esm.sh/langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "https://esm.sh/@langchain/community/embeddings/hf_transformers";
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const logWin = document.getElementById('event-log');
const modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC"; // Efficient for browsers
let vectorStore, engine;

function log(msg) {
    const div = document.createElement('div');
    div.innerText = `> ${msg}`;
    logWin.appendChild(div);
    logWin.scrollTop = logWin.scrollHeight;
}

const embeddings = new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" });

// 1. PDF Loading
document.getElementById('pdf-upload').onchange = async (e) => {
    log("Reading PDF file...");
    const loader = new WebPDFLoader(e.target.files[0]);
    const docs = await loader.load();
    log(`PDF loaded. Splitting ${docs.length} pages into chunks...`);
    
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
    const splitDocs = await splitter.splitDocuments(docs);
    log(`Generated ${splitDocs.length} text chunks.`);
    
    log("Creating vector index in RAM...");
    vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    log("✅ Knowledge base indexed.");
    checkReady();
};

// 2. AI Initialization
document.getElementById('load-ai').onclick = async () => {
    document.querySelector('.progress').style.display = 'flex';
    log("Connecting to WebWorker and downloading model...");
    
    try {
        engine = await webllm.CreateWebWorkerMLCEngine(
            new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }),
            modelId,
            { initProgressCallback: (p) => {
                document.getElementById('ai-bar').style.width = (p.progress * 100) + "%";
                document.getElementById('ai-status').innerText = p.text;
                if(p.text.includes("Finish")) log("✅ AI Model cached successfully.");
            }}
        );
        log("✅ AI Engine ready.");
        checkReady();
    } catch (err) { log("❌ GPU/Worker Error: " + err.message); }
};

// 3. Chatting
document.getElementById('send-btn').onclick = async () => {
    const input = document.getElementById('user-input');
    const question = input.value;
    input.value = "";
    appendMsg("User", question);

    log("Searching vector store for context...");
    const contextResults = await vectorStore.similaritySearch(question, 2);
    const contextText = contextResults.map(d => d.pageContent).join("\n---\n");
    log(`Found ${contextResults.length} relevant sections.`);

    log("Thinking...");
    const messages = [
        { role: "system", content: "You are a helpful assistant. Use the provided context to answer." },
        { role: "user", content: `Context:\n${contextText}\n\nQuestion: ${question}` }
    ];

    const aiBox = appendMsg("AI", "...");
    const chunks = await engine.chat.completions.create({ messages, stream: true });
    let responseText = "";
    for await (const chunk of chunks) {
        responseText += chunk.choices[0]?.delta.content || "";
        aiBox.innerText = responseText;
    }
    log("AI response generated.");
};

function checkReady() { if(vectorStore && engine) { document.getElementById('user-input').disabled = false; document.getElementById('send-btn').disabled = false; }}
function appendMsg(role, text) {
    const win = document.getElementById('chat-window');
    if(win.innerText.includes("Upload a PDF")) win.innerHTML = "";
    const div = document.createElement('div');
    div.className = role === 'User' ? 'user-msg' : 'ai-msg';
    div.innerHTML = `<strong>${role}:</strong> <p class="mb-0">${text}</p>`;
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
    return div.querySelector('p');
}