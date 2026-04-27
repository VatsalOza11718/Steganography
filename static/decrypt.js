/* ═══════════════════════════════════════════════════════════════
   BIS — Decrypt page logic
   Fully wired to HTML IDs in decrypt.html and Flask API responses
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    /* ── State ─────────────────────────────────────────────────── */
    let method = "text"; // text | audio | video | image

    /* ── DOM refs ──────────────────────────────────────────────── */
    const methodCards = document.querySelectorAll("[data-method]");
    const decryptBtn = document.getElementById("btn-decrypt");
    const resultCard = document.getElementById("decrypt-result");
    const stegoTitle = document.getElementById("stego-title");
    const stegoDesc = document.getElementById("stego-desc");
    const decOutput = document.getElementById("dec-output");
    const decDots = document.getElementById("decrypt-dots");

    const inputSections = {
        text: document.getElementById("dec-text-input"),
        audio: document.getElementById("dec-audio-input"),
        video: document.getElementById("dec-video-input"),
        image: document.getElementById("dec-image-input"),
    };

    /* ── Helpers ───────────────────────────────────────────────── */
    const labels = {
        text: { t: "Provide Stego Text", d: "Paste the text that contains the hidden message." },
        audio: { t: "Upload Stego Audio", d: "Upload the WAV file that contains hidden data." },
        video: { t: "Upload Stego Video", d: "Upload the video file that contains hidden data." },
        image: { t: "Upload Stego Image", d: "Upload the PNG image that contains hidden data." },
    };

    function updateUI() {
        Object.values(inputSections).forEach(s => { if (s) s.classList.add("hidden"); });
        if (inputSections[method]) inputSections[method].classList.remove("hidden");
        if (stegoTitle) stegoTitle.textContent = labels[method].t;
        if (stegoDesc) stegoDesc.textContent = labels[method].d;
        validateForm();
    }

    function validateForm() {
        let hasInput = false;
        if (method === "text") {
            hasInput = document.getElementById("stego-text")?.value.trim().length > 0;
        } else if (method === "audio") {
            hasInput = document.getElementById("dec-audio-file")?.files.length > 0;
        } else if (method === "video") {
            hasInput = document.getElementById("dec-video-file")?.files.length > 0;
        } else if (method === "image") {
            hasInput = document.getElementById("dec-image-file")?.files.length > 0;
        }
        if (decryptBtn) decryptBtn.disabled = !hasInput;
    }

    /* ── Method selection ──────────────────────────────────────── */
    methodCards.forEach(card => {
        card.addEventListener("click", () => {
            methodCards.forEach(c => { c.classList.remove("active"); c.setAttribute("aria-checked", "false"); });
            card.classList.add("active"); card.setAttribute("aria-checked", "true");
            method = card.dataset.method;
            updateUI();
        });
    });

    /* ── Validation listeners ──────────────────────────────────── */
    document.getElementById("stego-text")?.addEventListener("input", validateForm);

    /* ── Drop zones ────────────────────────────────────────────── */
    initDropZone("dec-audio-dropzone", "dec-audio-file", "dec-audio-filename", () => validateForm());
    initDropZone("dec-video-dropzone", "dec-video-file", "dec-video-filename", () => validateForm());
    initDropZone("dec-image-dropzone", "dec-image-file", "dec-image-filename", () => validateForm());

    /* ── Decrypt ───────────────────────────────────────────────── */
    if (decryptBtn) {
        decryptBtn.addEventListener("click", async () => {
            const pw = document.getElementById("dec-password")?.value || "";
            showOverlay("Decrypting...");
            decryptBtn.disabled = true;

            // Show processing dots
            if (decDots) decDots.classList.remove("hidden");

            // Activate scan line animation on the stego content card
            const stegoCard = document.getElementById("step-stego");
            if (stegoCard && typeof startScanLine === "function") startScanLine(stegoCard);

            try {
                let result;
                if (method === "text") {
                    const stego = document.getElementById("stego-text").value;
                    const r = await fetch("/api/decrypt-text", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ stego_text: stego, password: pw })
                    });
                    result = await r.json();
                    if (!r.ok) throw new Error(result.error || "Decryption failed");
                }
                else if (method === "audio") {
                    const fd = new FormData();
                    fd.append("audio", document.getElementById("dec-audio-file").files[0]);
                    fd.append("password", pw);
                    const r = await fetch("/api/decrypt-audio", { method: "POST", body: fd });
                    result = await r.json();
                    if (!r.ok) throw new Error(result.error || "Decryption failed");
                }
                else if (method === "video") {
                    const fd = new FormData();
                    fd.append("video", document.getElementById("dec-video-file").files[0]);
                    fd.append("password", pw);
                    const r = await fetch("/api/decrypt-video", { method: "POST", body: fd });
                    result = await r.json();
                    if (!r.ok) throw new Error(result.error || "Decryption failed");
                }
                else if (method === "image") {
                    const fd = new FormData();
                    fd.append("stego_image", document.getElementById("dec-image-file").files[0]);
                    fd.append("password", pw);
                    const r = await fetch("/api/decrypt-image", { method: "POST", body: fd });
                    result = await r.json();
                    if (!r.ok) throw new Error(result.error || "Decryption failed");
                }

                // Show result - Use the correct field names from Flask API responses
                // The result card has CSS class `decrypt-reveal` which keeps it
                // at opacity:0 until `revealed` class is added.
                resultCard.classList.remove("hidden");
                requestAnimationFrame(() => {
                    resultCard.classList.add("revealed");
                    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
                });

                // Flask API returns:
                //   decrypt-text:  { text: "..." }
                //   decrypt-audio: { text: "..." }
                //   decrypt-video: { text: "..." }
                //   decrypt-image: { text: "...", extracted_text: "..." }
                const extractedText = result.text || result.extracted_text || "(empty)";

                if (decOutput) {
                    // Use typewriter effect for reveal
                    if (typeof typewriterReveal === "function") {
                        typewriterReveal(decOutput, extractedText, 20);
                    } else {
                        decOutput.textContent = extractedText;
                    }
                }

                // Show success burst animation
                if (typeof triggerBurst === "function") triggerBurst(resultCard);

                // Build metrics
                const metricsGrid = document.getElementById("decrypt-metrics");
                if (metricsGrid) {
                    metricsGrid.innerHTML = "";
                    const items = [];
                    items.push(["Method", method.charAt(0).toUpperCase() + method.slice(1)]);
                    items.push(["Characters", String(extractedText.length)]);
                    if (pw) items.push(["AES-256", "Decrypted"]);
                    if (result.encrypted !== undefined) items.push(["Encrypted", result.encrypted ? "Yes" : "No"]);
                    items.forEach(([l, v]) => {
                        metricsGrid.innerHTML += `<div class="metric-item"><span class="metric-label">${l}</span><span class="metric-value">${v}</span></div>`;
                    });
                }

                showToast("Secret message revealed!");
            } catch (err) {
                showToast(err.message || "Decryption failed", "error");
            } finally {
                hideOverlay();
                if (decDots) decDots.classList.add("hidden");
                if (stegoCard && typeof stopScanLine === "function") stopScanLine(stegoCard);
                validateForm();
            }
        });
    }

    /* ── Copy extracted ────────────────────────────────────────── */
    document.getElementById("btn-copy-dec")?.addEventListener("click", () => {
        copyText(document.getElementById("dec-output")?.textContent || "");
    });

    /* ── Duration slider formatters ───────────────────────────── */
    const aiAudioDur = document.getElementById("ai-audio-dur-slider");
    const aiAudioDurVal = document.getElementById("ai-audio-dur-value");
    if (aiAudioDur && aiAudioDurVal) {
        aiAudioDur.addEventListener("input", () => { aiAudioDurVal.textContent = fmtDuration(+aiAudioDur.value); });
    }

    /* ── Init ──────────────────────────────────────────────────── */
    updateUI();

})();
