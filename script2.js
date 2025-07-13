let tokenizer;
let model;
let autoModeActive = false;
const MAX_VOCAB_SIZE = 20000; 

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

		//		const wordCount = prompt.split(/\s+/).length;
		//		if (wordCount < 1) {
		//			const li = document.createElement("li");
		//			li.textContent = "❗ Bitte mindestens ein Wort eingeben.";
		//			li.style.color = "crimson";
		//			suggestionsList.appendChild(li);
		//			return;
		//		}

		const predictions = await predictNextWords(prompt, 5);
		if (!predictions) return;

		predictions.forEach(pred => {
			const li = document.createElement("li");
			li.textContent = `${pred.word} (${(pred.prob * 100).toFixed(1)}%)`;
			li.addEventListener("click", () => {
				const input = document.getElementById("userPrompt");
				const endsWithSpace = /\s$/.test(input.value);
				const lastChar = input.value.trim().slice(-1);

				const formattedWord = /[.!?]/.test(lastChar)
					? capitalizeFirst(pred.word)
					: pred.word;

				appendToken(input, formattedWord);


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

			if (window.lastEvalResult) {
			  drawPlotlyChart(window.lastEvalResult);
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
		const tokens = text
			.toLowerCase()
			.trim()
			.split(/\s+/);                   // tokenisieren
		return tokens.map(word => this.wordIndex[word] || 0);
	}


	sequencesToTexts(seq) {
		return seq.map(index => this.indexWord[index] || "<?>");
	}
}

function drawPlotlyChart(data) {
  const topk = data.topk_accuracy;
  const isDark = document.body.classList.contains("dark");

  Plotly.newPlot("plotly-chart", [{
    x: Object.keys(topk).map(k => parseInt(k)),
    y: Object.values(topk),
    type: "bar",
    marker: {
      color: isDark ? "lightblue" : "steelblue"
    }
  }], {
    title: "Top-k Trefferquote",
    paper_bgcolor: isDark ? "#222" : "#fff",
    plot_bgcolor: isDark ? "#222" : "#fff",
    font: { color: isDark ? "#eee" : "#000" },
    xaxis: { title: "k", color: isDark ? "#eee" : "#000" },
    yaxis: { title: "Trefferquote (%)", color: isDark ? "#eee" : "#000" }
  });
}


fetch("https://jachirobi.github.io/DeepLearningEA3/stats/evaluation_result.json")
  .then(res => res.json())
  .then(data => {
    window.lastEvalResult = data; // ⬅️ Speichere Daten global
    const topk = data.topk_accuracy;

    for (const k in topk) {
      const span = document.getElementById(`acc${k}`);
      if (span) span.textContent = topk[k];
    }

    document.getElementById("perplexity").textContent = data.perplexity;
    drawPlotlyChart(data); // ⬅️ Verwende neue Funktion
  });




function isPunctuation(token) {
	return /^[.,!?;:]$/.test(token);
}


function capitalizeFirst(word) {
	return word.charAt(0).toUpperCase() + word.slice(1);
}

function appendToken(inputEl, token) {
	const current = inputEl.value.trim();
	const endsWithSpace = /\s$/.test(inputEl.value);
	const isPunct = isPunctuation(token);

	if (isPunct) {
		inputEl.value = inputEl.value.trimEnd() + token + " ";
	} else {
		let formattedToken = token;
		
		// Wenn der Eingabetext leer ist, schreibe erstes Wort groß
		if (current.length === 0) {
			formattedToken = capitalizeFirst(token);
		}

		inputEl.value = inputEl.value + (endsWithSpace || inputEl.value.length === 0 ? "" : " ") + formattedToken;
	}
}


function formatWordForContext(currentText, predictedWord) {
	const lastChar = currentText.trim().slice(-1);
	if (/[.!?]/.test(lastChar)) {
		return capitalizeFirst(predictedWord);
	}
	return predictedWord.toLowerCase();
}

function sampleFromDistribution(probs, temperature = 1.0) {
	const adjusted = probs.map(p => Math.pow(p, 1 / temperature));
	const sum = adjusted.reduce((a, b) => a + b, 0);
	const normalized = adjusted.map(p => p / sum);

	const r = Math.random();
	let cumSum = 0;
	for (let i = 0; i < normalized.length; i++) {
		cumSum += normalized[i];
		if (r < cumSum) return i;
	}
	return normalized.length - 1;
}


// Vorhersage-Funktion
async function predictNextWords(promptText, topK = 5) {
	const tokens = promptText
	  .trim()
	  .split(/\s+/)
	  .filter(Boolean);

	  const inputTokens = tokens.slice(-5); // Nur die letzten 5 Tokens verwenden
	  const padding = Array(5 - inputTokens.length).fill(0); // ggf. links mit 0 auffüllen

	  const inputSeq = padding.concat(
	  	inputTokens.map(w => {
	  		const index = tokenizer.wordIndex[w] || 0;
	  		return (index > 0 && index < MAX_VOCAB_SIZE) ? index : 0;
	  	})
	  );





	const inputTensor = tf.tensor2d([inputSeq], [1, 5]);
	const prediction = model.predict(inputTensor);
	const probs = await prediction.data();

//	const topIndices = Array.from(probs)
//		.map((p, i) => ({ word: tokenizer.indexWord[i], prob: p }))
//		.filter(entry => entry.word && entry.word.toLowerCase() !== "<unk>")
//		.sort((a, b) => b.prob - a.prob)
//		.slice(0, topK);

//	return topIndices;

const wordsWithProbs = Array.from(probs)
	.map((p, i) => ({ word: tokenizer.indexWord[i], prob: p }))
	.filter(entry => entry.word && entry.word.toLowerCase() !== "<unk>");

const sampled = [];
for (let i = 0; i < topK; i++) {
	const index = sampleFromDistribution(wordsWithProbs.map(w => w.prob), 0.8); // temperature = 0.8 z.B.
	sampled.push(wordsWithProbs[index]);
	wordsWithProbs.splice(index, 1); // entferne bereits gezogenes Wort
}

return sampled;

}



// „Weiter“-Button Vorhersage
async function continueWithTopPrediction() {
	const promptInput = document.getElementById("userPrompt");
	const currentText = promptInput.value.trim();

	const predictions = await predictNextWords(currentText, 1);
	if (!predictions || predictions.length === 0) return;

	const nextWord = predictions[0].word;
	const endsWithSpace = /\s$/.test(promptInput.value);
	const lastChar = currentText.slice(-1);
//	const formattedWord = /[.!?]/.test(lastChar)
//		? capitalizeFirst(nextWord)
//		: nextWord;
		
	const formattedWord = formatWordForContext(currentText, nextWord);

	appendToken(promptInput, formattedWord);






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

		const predictions = await predictNextWords(prompt, 1);
		if (!predictions || predictions.length === 0) break;

		const nextWord = predictions[0].word;
		const endsWithSpace = /\s$/.test(prompt);
		const lastChar = prompt.slice(-1);
		const formattedWord = /[.!?]/.test(lastChar)
			? capitalizeFirst(nextWord)
			: nextWord;

		appendToken(promptInput, formattedWord);




		await new Promise(resolve => setTimeout(resolve, 300));
		await document.getElementById("predictBtn").click();

		document.getElementById("autoBtn").disabled = false;
	}

	autoModeActive = false;
}
