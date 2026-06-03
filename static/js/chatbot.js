async function sendChatMessage(contextStr) {
    const inputEl = document.getElementById('doctorInput');
    const windowEl = document.getElementById('chatWindow');
    const msg = inputEl.value.trim();
    
    if(!msg) return;

    // 1. Echo User Message Frame (Physician Output Layout)
    windowEl.innerHTML += `
        <div class="bg-slate-800/60 backdrop-blur-md border border-slate-700/50 text-slate-100 p-3.5 rounded-2xl rounded-tr-none max-w-[85%] self-end ml-auto shadow-md mb-3.5 animate__animated animate__fadeInUp">
            <span class="text-[9px] text-slate-400 font-bold block mb-1 uppercase tracking-widest">You (Physician)</span>
            <p class="text-xs font-normal leading-relaxed">${msg}</p>
        </div>`;
    
    inputEl.value = '';
    windowEl.scrollTop = windowEl.scrollHeight;

    try {
        // 2. Transmit pipeline stream payload to backend routing endpoint
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, context: contextStr })
        });
        const data = await response.json();
        
        // 🟢 FIXED: Directly accept the pre-formatted backend HTML string to stop double bullets
        let formattedReply = data.reply;

        // 3. Output Bot Narrative Frame (Dr. Alex Medical Persona Alignment)
        windowEl.innerHTML += `
            <div class="bg-slate-900/80 backdrop-blur-md border border-sky-500/20 text-slate-200 p-4 rounded-2xl rounded-tl-none max-w-[85%] mr-auto shadow-lg mb-3.5 src-bot-msg animate__animated animate__fadeInUp">
                <span class="text-sky-400 font-extrabold block mb-1.5 text-[10px] uppercase tracking-wider font-mono-custom">Dr. Alex (AI Consultant)</span>
                <div class="text-xs font-normal leading-relaxed text-slate-300 space-y-1">${formattedReply}</div>
            </div>`;
        
        windowEl.scrollTop = windowEl.scrollHeight;
    } catch (e) {
        // 4. Fallback pipeline error layout state
        windowEl.innerHTML += `
            <div class="text-rose-400 p-3 text-[10px] bg-rose-950/30 border border-rose-900/30 rounded-xl my-2 font-mono-custom">
                ⚠️ Connection sync failure with the medical knowledge module.
            </div>`;
        windowEl.scrollTop = windowEl.scrollHeight;
    }
}