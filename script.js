let tokenizer;
let xs;
let ys;
let model;
let autoModeActive = false;

class Tokenizer {
	
  constructor() {
    this.wordIndex = {};
    this.indexWord = {};
    this.vocabSize = 0;
  }

  fitOnText(text) {
    const words = text.toLowerCase().split(/\s+/);
    const uniqueWords = [...new Set(words)];

    uniqueWords.forEach((word, idx) => {
      this.wordIndex[word] = idx + 1; // 0 für Padding reserviert
      this.indexWord[idx + 1] = word;
    });

    this.vocabSize = uniqueWords.length + 1; // +1 für Padding
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

// Starte, sobald die Seite vollständig geladen ist
window.addEventListener("load", async () => {

	tf.setBackend('cpu');
	
	await loadLeipzigCorpus("https://jachirobi.github.io/DeepLearningEA3/data/deu_news_2024_1M-sentences.txt");
	
	let autoModeActive = false;

	// Tensor vorbereiten
	// const xs => entfernt, global deklariert oben = tf.tensor2d(inputs, [inputs.length, 3]);
	// const ys => entfernt, global deklariert oben = tf.oneHot(labels, tokenizer.vocabSize);

	// const model => entfernt, global deklariert oben = tf.sequential();

	model.add(tf.layers.embedding({
	  inputDim: tokenizer.vocabSize,
	  outputDim: 64,
	  inputLength: 3
	}));

	model.add(tf.layers.lstm({ units: 100, returnSequences: true }));
	model.add(tf.layers.lstm({ units: 100 }));
	model.add(tf.layers.dense({ units: tokenizer.vocabSize, activation: 'softmax' }));

	model.compile({
	  loss: 'categoricalCrossentropy',
	  optimizer: tf.train.adam(0.01),
	  metrics: ['accuracy']
	});

	await model.fit(xs, ys, {
	  epochs: 100,
	  batchSize: 32,
	  callbacks: {
	    onEpochEnd: async (epoch, logs) => {
	      console.log(`Epoche ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc?.toFixed(4)}`);
	    }
	  }
	});

	document.getElementById("resetBtn").addEventListener("click", () => {
	  document.getElementById("userPrompt").value = "";
	  document.getElementById("suggestionsList").innerHTML = "";
	});

	document.getElementById("autoBtn").addEventListener("click", () => {
	  if (!autoModeActive) {
	    runAutoPrediction(10);
	  }
	});

	document.getElementById("stopBtn").addEventListener("click", () => {
	  autoModeActive = false;
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

			// Optional: dynamisch aria-hidden an content setzen
			if (contentId) {
				const content = document.getElementById(contentId);
				if (content) {
					content.setAttribute("aria-hidden", expanded ? "true" : "false");
				}
			}
		});
	});
	
	document.getElementById("weiterBtn").addEventListener("click", () => {
	  continueWithTopPrediction();
	});

	
	function createSequences(tokens, seqLength = 3) {
	  const inputs = [];
	  const labels = [];

	  for (let i = 0; i < tokens.length - seqLength; i++) {
	    const inputSeq = tokens.slice(i, i + seqLength);
	    const label = tokens[i + seqLength];
	    inputs.push(inputSeq);
	    labels.push(label);
	  }

	  return { inputs, labels };
	}

	async function predictNextWords(promptText, topK = 5) {
	  const words = promptText.trim().toLowerCase().split(/\s+/);
	  if (words.length < 3) return;

	  const lastWords = words.slice(-3);
	  const inputSeq = lastWords.map(w => tokenizer.wordIndex[w] || 0);

	  const inputTensor = tf.tensor2d([inputSeq], [1, 3]);
	  const prediction = model.predict(inputTensor);
	  const probs = await prediction.data();

	  // Top-K Wahrscheinlichkeiten sortieren
	  const topIndices = Array.from(probs)
	    .map((p, i) => ({ word: tokenizer.indexWord[i], prob: p }))
	    .sort((a, b) => b.prob - a.prob)
	    .slice(0, topK);

	  return topIndices;
	}
	
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

	  // Automatisch nächste Vorhersage anzeigen
	  document.getElementById("predictBtn").click();
	}

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

	    // Neue Vorschläge anzeigen
	    await new Promise(resolve => setTimeout(resolve, 300));
	    await document.getElementById("predictBtn").click();

		document.getElementById("autoBtn").disabled = false;
	  }

	  autoModeActive = false;
	}


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
	      document.getElementById("predictBtn").click(); // neue Vorhersage
	    });
	    suggestionsList.appendChild(li);
	  });
	});
	
	async function loadLeipzigCorpus(filename) {
 	 tokenizer = new Tokenizer();
	  const res = await fetch(filename);
	  const raw = await res.text();

	  // Zeilen parsen
	  const lines = raw.split('\n').map(line => {
	    // Nummern und Tabs/Leerzeichen löschen
	    return line.replace(/^\s*\d+\s+/, '').trim();
	  });

	  // Saubere Sätze zusammenfügen
	  const cleanText = lines.filter(l => l.length).join(' ');

	  tokenizer.fitOnText(cleanText);
	  const tokens = tokenizer.textsToSequences(cleanText);
	  const { inputs, labels } = createSequences(tokens, 3);
	  xs = tf.tensor2d(inputs, [inputs.length, 3]);
	  ys = tf.oneHot(labels, tokenizer.vocabSize);

	  console.log("✅ Leipzig-Korpus geladen:", inputs.length, "Sequenzen");
	}


});
