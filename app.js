import { WebPDFLoader } from "https://esm.sh/@langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "https://esm.sh/langchain/text_splitter";
import { MemoryVectorStore } from "https://esm.sh/langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "https://esm.sh/@langchain/community/embeddings/hf_transformers";
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let vectorStore, engine;
const modelId = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

const embeddings = new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" });

// Handle PDF Processing
document.getElementById('pdf-upload').onchange = async (e) => {
    const status = document.getElementById('pdf-status');
    status.innerText = "⏳ Indexing...";
    try {
        const loader = new WebPDFLoader(e.target.files[0]);
        const docs = await loader.load();
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
        vectorStore = await MemoryVectorStore.fromDocuments(await splitter.splitDocuments(docs), embeddings);
        status.innerText = "✅ PDF Ready!";
        checkReady();
    } catch (err) { status.innerText = "❌ Error: " + err.message; }
};

// Handle AI Initialization
document.getElementById('load-ai').onclick = async () => {
    document.querySelector('.progress').style.display = 'flex';
    try {
        // We use a Worker to keep the UI smooth
        engine = await webllm.CreateWebWorkerMLCEngine(
            new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }),
            modelId,
            { initProgressCallback: (p) => {
                document.getElementById('ai-bar').style.width = (p.progress * 100) + "%";
                document.getElementById('ai-status').innerText = p.text;
            }}
        );
        document.getElementById('ai-status').innerText = "✅ AI Engine Ready!";
        checkReady();
    } catch (err) { alert("WebGPU Error: " + err.message); }
};

document.getElementById('send-btn').onclick = async () => {
    const input = document.getElementById('user-input');
    const query = input.value;
    input.value = "";
    appendChat("User", query);

    const context = (await vectorStore.similaritySearch(query, 2)).map(d => d.pageContent).join("\n");
    const messages = [
        { role: "system", content: "Use the context to answer strictly." },
        { role: "user", content: `Context: ${context}\n\nQuestion: ${query}` }
    ];

    const aiBox = appendChat("AI", "Thinking...");
    const chunks = await engine.chat.completions.create({ messages, stream: true });
    let fullText = "";
    for await (const chunk of chunks) {
        fullText += chunk.choices[0]?.delta.content || "";
        aiBox.innerText = fullText;
    }
};

function checkReady() { if(vectorStore && engine) { document.getElementById('user-input').disabled = false; document.getElementById('send-btn').disabled = false; }}
function appendChat(role, text) {
    const div = document.createElement('div');
    div.innerHTML = `<strong>${role}:</strong> <p>${text}</p><hr>`;
    document.getElementById('chat-window').appendChild(div);
    return div.querySelector('p');
}