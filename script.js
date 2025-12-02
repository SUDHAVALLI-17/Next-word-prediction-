const predictBtn = document.getElementById('predictBtn');
const voiceBtn = document.getElementById('voiceBtn');
const clearBtn = document.getElementById('clearBtn');
const inputText = document.getElementById('inputText');
const prediction = document.getElementById('prediction');
const languageSelector = document.getElementById('languageSelector');
const historyList = document.getElementById('historyList');
const modelSelector = document.getElementById('modelSelector');



// Fetch predictions (shared function)
async function fetchPredictions(text) {
    const language = languageSelector.value;
    const model = modelSelector.value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language, model })
    });

    const data = await response.json();
    prediction.innerHTML = "";

    if (data.predictions && data.predictions.length > 0) {
        data.predictions.forEach(word => {
            const btn = document.createElement("button");
            btn.innerText = word;
            btn.className = "prediction-button";
            btn.onclick = () => {
                inputText.value += " " + word;
                inputText.focus();
                fetchPredictions(inputText.value); // Refresh predictions
            };
            prediction.appendChild(btn);
        });
    } else {
        prediction.textContent = "🤔 (no suggestion)";
    }
}


// Manual prediction button
predictBtn.onclick = () => {
    const text = inputText.value.trim();
    if (text) fetchPredictions(text);
};

// Live predictions on input
inputText.addEventListener('input', () => {
    const text = inputText.value.trim();
    if (text) fetchPredictions(text);
    else prediction.innerHTML = "";
});

// Voice input
voiceBtn.onclick = () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    const langMap = { "en": "en-US", "hi": "hi-IN", "ta": "ta-IN" };
    recognition.lang = langMap[languageSelector.value];
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const voiceText = event.results[0][0].transcript;
        inputText.value += " " + voiceText;
        inputText.focus();
        fetchPredictions(inputText.value);
    };

    recognition.onerror = (event) => {
        alert("Voice recognition error: " + event.error);
    };

    recognition.start();
};

// Clear input and predictions
clearBtn.onclick = () => {
    if (inputText.value.trim()) {
        saveHistory(inputText.value);
    }
    inputText.value = "";
    prediction.innerHTML = "";
};

// Save history (session-based)
function saveHistory(text) {
    const item = document.createElement("li");
    item.textContent = text;
    historyList.prepend(item); // newest on top
}

 