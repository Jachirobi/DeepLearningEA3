let tokenizer;
let model;
let autoModeActive = false;

// Starte, sobald die Seite vollständig geladen ist
window.addEventListener("load", async () => {
  tf.setBackend("cpu");

  // Modell laden
  model = await tf.loadLayersModel(
    "https://jachirobi.github.io/DeepLearningEA3/model/model.json"
  );
  console.log("✅ Modell geladen.");

  // Tokenizer laden
  const tokenizerRes = await fetch("https://jachirobi.github.io/DeepLearningEA3/model/tokenizer_word_index.json");
  const wordIndexData = await tokenizerRes.json();
  tokenizer = new Tokenizer();
  tokenizer.wordIndex = wordIndexData;
  tokenizer.indexWord = Object.fromEntries(
    Object.entries(wordIndexData).map(([k, v]) => [v, k])
  );
  tokenizer.vocabSize = Object.keys(tokenizer.wordIndex).length + 1;

  // Button-Handler
  document.getElementById("resetBtn").addEventListener("click", () => {
    document.getElementById("userPrompt").value = "";
    document.getElementById("suggestionsList").innerHTML = "";
  });

  document.getElementById("weiterBtn").addEventListener("click", () => {
    continueWithTopPrediction();
  });

  document.getElementById("autoBtn").addEventListener("click", () => {
    if (!autoModeActive) runAutoPrediction(10);
  });

  document.getElementById("stopBtn").addEventListener("click", () => {
    autoModeActive = false;
  });

  document.getElementById("predictBtn").addEventListener("click", async () => {
    const prompt = document.getElementById("userPrompt").value.trim();
    const suggestionsList = document.getElementById("suggestionsList");
    suggestionsList.innerHTML = "";

    const wordCount = prompt.split(/\s+/).length;
    if (wordCount < 3) {
      const li = document.createElement("li");
      li.textContent = "❗ Bitte mindestens 3 Wörter eingeben.";
      li.style.color = "crimson";
      suggestionsList.appendChild(li);
      return;
    }

    const predictions = await predictNextWords(prompt, 5);
    if (!predictions) return;

    predictions.forEach(pred => {
      const li = document.createElement("li");
      li.textContent = `${pred.word} (${(pred.prob * 100).toFixed(1)}%)`;
      li.addEventListener("click", () => {
        document.getElementById("userPrompt").value += " " + pred.word;
        document.getElementById("predictBtn").click();
      });
      suggestionsList.appendChild(li);
    });
  });

  // Dark Mode Toggle
  document.getElementById("darkModeToggle").addEventListener("click", () => {
    const isDark = document.body.classList.toggle("dark");
    document.getElementById("darkModeToggle").textContent =
      isDark ? "☀️ Light Mode aktivieren" : "🌙 Dark Mode aktivieren";

    const screenshotImg = document.getElementById("screenshot-img");
    if (screenshotImg) {
      screenshotImg.src = isDark ? "overfit-details_dark.png" : "overfit-details.png";
    }
  });

  document.querySelectorAll(".collapsible-section .toggle-button").forEach(btn => {
    btn.addEventListener("click", () => {
      const section = btn.closest(".collapsible-section");
      const contentId = btn.getAttribute("aria-controls");
      const expanded = section.classList.toggle("collapsed");

      btn.textContent = section.classList.contains("collapsed") ? "⬆️ Ausklappen" : "⬇️ Einklappen";
      btn.setAttribute("aria-expanded", !expanded);

      if (contentId) {
        const content = document.getElementById(contentId);
        if (content) {
          content.setAttribute("aria-hidden", expanded ? "true" : "false");
        }
      }
    });
  });
});

// Tokenizer-Klasse
class Tokenizer {
  constructor() {
    this.wordIndex = {};
    this.indexWord = {};
    this.vocabSize = 0;
  }

  textsToSequences(text) {
    return text
      .toLowerCase()
      .split(/\s+/)
      .map(word => this.wordIndex[word] || 0);
  }

  sequencesToTexts(seq) {
    return seq.map(index => this.indexWord[index] || "<?>");
  }
}

// Vorhersage-Funktion
async function predictNextWords(promptText, topK = 5) {
  const words = promptText.trim().toLowerCase().split(/\s+/);
  if (words.length < 3) return;

  const lastWords = words.slice(-3);
  const inputSeq = lastWords.map(w => tokenizer.wordIndex[w] || 0);

  const inputTensor = tf.tensor2d([inputSeq], [1, 3]);
  const prediction = model.predict(inputTensor);
  const probs = await prediction.data();

  const topIndices = Array.from(probs)
    .map((p, i) => ({ word: tokenizer.indexWord[i], prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, topK);

  return topIndices;
}

// „Weiter“-Button Vorhersage
async function continueWithTopPrediction() {
  const promptInput = document.getElementById("userPrompt");
  const currentText = promptInput.value.trim();

  if (currentText.split(/\s+/).length < 3) {
    alert("Bitte mindestens 3 Wörter eingeben, um fortzufahren.");
    return;
  }

  const predictions = await predictNextWords(currentText, 1);
  if (!predictions || predictions.length === 0) return;

  const nextWord = predictions[0].word;
  promptInput.value = currentText + " " + nextWord;

  document.getElementById("predictBtn").click();
}

// Automatische Vorhersage (max. 10 Wörter)
async function runAutoPrediction(maxWords = 10) {
  autoModeActive = true;
  const promptInput = document.getElementById("userPrompt");

  for (let i = 0; i < maxWords; i++) {
    document.getElementById("autoBtn").disabled = true;

    if (!autoModeActive) break;

    const prompt = promptInput.value.trim();
    if (prompt.split(/\s+/).length < 3) {
      alert("Bitte mindestens 3 Wörter eingeben.");
      break;
    }

    const predictions = await predictNextWords(prompt, 1);
    if (!predictions || predictions.length === 0) break;

    const nextWord = predictions[0].word;
    promptInput.value += " " + nextWord;

    await new Promise(resolve => setTimeout(resolve, 300));
    await document.getElementById("predictBtn").click();

    document.getElementById("autoBtn").disabled = false;
  }

  autoModeActive = false;
}
