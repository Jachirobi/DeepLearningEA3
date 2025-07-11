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
		const tokens = text
			.replace(/([.,!?;:])/g, " $1 ")   // Satzzeichen abtrennen
			.toLowerCase()
			.trim()
			.split(/\s+/);                   // tokenisieren
		return tokens.map(word => this.wordIndex[word] || 0);
	}


	sequencesToTexts(seq) {
		return seq.map(index => this.indexWord[index] || "<?>");
	}
}

function isPunctuation(token) {
	return /^[.,!?;:]$/.test(token);
}


function capitalizeFirst(word) {
	return word.charAt(0).toUpperCase() + word.slice(1);
}

function appendToken(inputEl, token) {
	const current = inputEl.value;
	const endsWithSpace = /\s$/.test(current);
	const isPunct = isPunctuation(token);

	if (isPunct) {
		// Kein Leerzeichen vor Satzzeichen
		inputEl.value = current.trimEnd() + token + " ";
	} else {
		// Leerzeichen vor Wort, falls nicht vorhanden
		inputEl.value = current + (endsWithSpace ? "" : " ") + token;
	}
}

function formatWordForContext(currentText, predictedWord) {
	const lastChar = currentText.trim().slice(-1);
	if (/[.!?]/.test(lastChar)) {
		return capitalizeFirst(predictedWord);
	}
	return predictedWord.toLowerCase();
}


// Vorhersage-Funktion
async function predictNextWords(promptText, topK = 5) {
	const words = promptText.trim().toLowerCase().split(/\s+/).filter(Boolean);
	const lastWords = words.slice(-5);  // egal wie viele da sind
	const inputSeq = Array(5 - lastWords.length).fill(0).concat(
		lastWords.map(w => {
			const index = tokenizer.wordIndex[w] || 0;
			return Math.min(index, tokenizer.vocabSize - 1); // Begrenze auf Modellbereich
		})
	);


	const inputTensor = tf.tensor2d([inputSeq], [1, 5]);
	const prediction = model.predict(inputTensor);
	const probs = await prediction.data();

	const topIndices = Array.from(probs)
		.map((p, i) => ({ word: tokenizer.indexWord[i], prob: p }))
		.filter(entry => entry.word && entry.word.toLowerCase() !== "<unk>")
		.sort((a, b) => b.prob - a.prob)
		.slice(0, topK);

	return topIndices;
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
