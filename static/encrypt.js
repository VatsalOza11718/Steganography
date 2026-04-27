/* ═══════════════════════════════════════════════════════════════
   BIS — Encrypt page logic
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    /* ── State ─────────────────────────────────────────────────── */
    let method = "text";        // text | audio | video | image
    let coverSource = "template"; // template | custom
    let selectedTemplate = null;
    let templateBlob = null;       // fetched file for audio/video templates

    /* ── DOM refs ──────────────────────────────────────────────── */
    const methodCards = document.querySelectorAll("[data-method]");
    const sourceButtons = document.querySelectorAll("[data-source]");
    const coverTitle = document.getElementById("cover-title");
    const coverDesc = document.getElementById("cover-desc");
    const encryptBtn = document.getElementById("btn-encrypt");
    const resultCard = document.getElementById("encrypt-result");
    const secretMsg = document.getElementById("secret-message");

    /* All cover sections keyed by visibility logic */
    const coverSections = {
        "tpl-text": document.getElementById("tpl-text-section"),
        "tpl-audio": document.getElementById("tpl-audio-section"),
        "tpl-video": document.getElementById("tpl-video-section"),
        "tpl-image": document.getElementById("tpl-image-section"),
        "cover-text": document.getElementById("cover-text-section"),
        "cover-audio": document.getElementById("cover-audio-section"),
        "cover-video": document.getElementById("cover-video-section"),
        "upload-image": document.getElementById("upload-image-section"),
    };

    /* ── Helpers ───────────────────────────────────────────────── */
    function hideAllCovers() { Object.values(coverSections).forEach(s => { if (s) s.classList.add("hidden"); }); }

    function showSection(key) {
        const el = coverSections[key];
        if (el) el.classList.remove("hidden");
    }

    function updateCoverUI() {
        hideAllCovers();
        const labels = {
            text: { t: "Provide Cover Text", d: "Select a template or upload your own cover." },
            audio: { t: "Provide Cover Audio", d: "Select a template or upload a WAV file." },
            video: { t: "Provide Cover Video", d: "Select a template or upload a video file." },
            image: { t: "Provide Cover Image", d: "Select a template or upload your own image." },
        };
        if (coverTitle) coverTitle.textContent = labels[method].t;
        if (coverDesc) coverDesc.textContent = labels[method].d;

        if (coverSource === "template") {
            showSection("tpl-" + method);
        } else if (coverSource === "custom") {
            if (method === "image") {
                showSection("upload-image");
            } else {
                showSection("cover-" + method);
            }
        }
    }

    function validateForm() {
        const hasMessage = secretMsg && secretMsg.value.trim().length > 0;
        let hasCover = false;

        if (method === "text") {
            if (coverSource === "template") { hasCover = !!selectedTemplate; }
            else { hasCover = document.getElementById("cover-text")?.value.trim().length > 0; }
        } else if (method === "audio") {
            if (coverSource === "template") { hasCover = !!templateBlob; }
            else { hasCover = document.getElementById("enc-audio-file")?.files.length > 0; }
        } else if (method === "video") {
            if (coverSource === "template") { hasCover = !!templateBlob; }
            else if (coverSource === "custom") { hasCover = document.getElementById("enc-video-file")?.files.length > 0; }
        } else if (method === "image") {
            if (coverSource === "template") { hasCover = !!templateBlob; }
            else { hasCover = document.getElementById("enc-image-file")?.files.length > 0; }
        }

        if (encryptBtn) encryptBtn.disabled = !(hasMessage && hasCover);
    }

    /* ── Method Selection ──────────────────────────────────────── */
    methodCards.forEach(card => {
        card.addEventListener("click", () => {
            methodCards.forEach(c => { c.classList.remove("active"); c.setAttribute("aria-checked", "false"); });
            card.classList.add("active"); card.setAttribute("aria-checked", "true");
            method = card.dataset.method;
            selectedTemplate = null; templateBlob = null;
            updateCoverUI(); validateForm();
        });
    });

    /* ── Source Selection ──────────────────────────────────────── */
    sourceButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            sourceButtons.forEach(b => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
            btn.classList.add("active"); btn.setAttribute("aria-selected", "true");
            coverSource = btn.dataset.source;
            selectedTemplate = null; templateBlob = null;
            updateCoverUI(); validateForm();
        });
    });

    /* ── Text Templates ────────────────────────────────────────── */
    const textTplGrid = document.getElementById("text-template-grid");
    const textPreview = document.getElementById("tpl-text-preview");
    if (textTplGrid) {
        textTplGrid.addEventListener("click", e => {
            const card = e.target.closest(".template-card"); if (!card) return;
            textTplGrid.querySelectorAll(".template-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            const tid = card.dataset.tpl;
            selectedTemplate = tid;
            card.classList.add("loading");
            fetch(`/api/templates/text/${tid}`).then(r => r.json()).then(d => {
                if (textPreview) textPreview.value = d.text || "";
                card.classList.remove("loading"); validateForm();
            }).catch(() => { card.classList.remove("loading"); showToast("Failed to load template", "error"); });
        });
    }

    /* ── Audio Templates ───────────────────────────────────────── */
    const audioTplGrid = document.getElementById("audio-template-grid");
    const audioDurSlider = document.getElementById("audio-duration-slider");
    const audioDurValue = document.getElementById("audio-duration-value");
    if (audioDurSlider && audioDurValue) {
        audioDurSlider.addEventListener("input", () => { audioDurValue.textContent = fmtDuration(+audioDurSlider.value); });
    }
    if (audioTplGrid) {
        audioTplGrid.addEventListener("click", e => {
            const card = e.target.closest(".template-card"); if (!card) return;
            audioTplGrid.querySelectorAll(".template-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active"); card.classList.add("loading");
            selectedTemplate = card.dataset.tpl; templateBlob = null;
            const dur = audioDurSlider ? audioDurSlider.value : "120";
            fetch(`/api/templates/audio/${card.dataset.tpl}?duration=${dur}`)
                .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
                .then(b => {
                    templateBlob = b; card.classList.remove("loading");
                    const status = document.getElementById("tpl-audio-status");
                    const name = document.getElementById("tpl-audio-name");
                    if (status) status.classList.remove("hidden");
                    if (name) name.textContent = `✓ ${card.querySelector(".tpl-name").textContent} loaded (${fmtDuration(+dur)})`;
                    validateForm();
                }).catch(() => { card.classList.remove("loading"); showToast("Audio generation failed", "error"); });
        });
    }

    /* ── Video Templates ───────────────────────────────────────── */
    const videoTplGrid = document.getElementById("video-template-grid");
    const videoDurSlider = document.getElementById("video-duration-slider");
    const videoDurValue = document.getElementById("video-duration-value");
    if (videoDurSlider && videoDurValue) {
        videoDurSlider.addEventListener("input", () => { videoDurValue.textContent = fmtDuration(+videoDurSlider.value); });
    }
    if (videoTplGrid) {
        videoTplGrid.addEventListener("click", e => {
            const card = e.target.closest(".template-card"); if (!card) return;
            videoTplGrid.querySelectorAll(".template-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active"); card.classList.add("loading");
            selectedTemplate = card.dataset.tpl; templateBlob = null;
            const dur = videoDurSlider ? videoDurSlider.value : "60";
            fetch(`/api/templates/video/${card.dataset.tpl}?duration=${dur}`)
                .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
                .then(b => {
                    templateBlob = b; card.classList.remove("loading");
                    const status = document.getElementById("tpl-video-status");
                    const name = document.getElementById("tpl-video-name");
                    if (status) status.classList.remove("hidden");
                    if (name) name.textContent = `✓ ${card.querySelector(".tpl-name").textContent} loaded (${fmtDuration(+dur)})`;
                    validateForm();
                }).catch(() => { card.classList.remove("loading"); showToast("Video generation failed", "error"); });
        });
    }

    /* ── Image Templates ──────────────────────────────────────── */
    const imageTplGrid = document.getElementById("image-template-grid");
    if (imageTplGrid) {
        imageTplGrid.addEventListener("click", e => {
            const card = e.target.closest(".template-card"); if (!card) return;
            imageTplGrid.querySelectorAll(".template-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active"); card.classList.add("loading");
            selectedTemplate = card.dataset.tpl; templateBlob = null;
            fetch(`/api/templates/image/${card.dataset.tpl}`)
                .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
                .then(b => {
                    templateBlob = b; card.classList.remove("loading");
                    const status = document.getElementById("tpl-image-status");
                    const name = document.getElementById("tpl-image-name");
                    if (status) status.classList.remove("hidden");
                    if (name) name.textContent = `✓ ${card.querySelector(".tpl-name").textContent} image loaded`;
                    // Clear any manually uploaded file to avoid confusion
                    const fileInput = document.getElementById("enc-image-file");
                    if (fileInput) fileInput.value = "";
                    const badge = document.getElementById("enc-image-filename");
                    if (badge) badge.classList.add("hidden");
                    validateForm();
                }).catch(() => { card.classList.remove("loading"); showToast("Failed to load image template", "error"); });
        });
    }

    /* ── Drop zones ────────────────────────────────────────────── */
    initDropZone("enc-audio-dropzone", "enc-audio-file", "enc-audio-filename", () => validateForm());
    initDropZone("enc-video-dropzone", "enc-video-file", "enc-video-filename", () => validateForm());
    initDropZone("enc-image-dropzone", "enc-image-file", "enc-image-filename", () => {
        // If user uploads a file, clear any template selection
        templateBlob = null; selectedTemplate = null;
        const status = document.getElementById("tpl-image-status");
        if (status) status.classList.add("hidden");
        imageTplGrid?.querySelectorAll(".template-card").forEach(c => c.classList.remove("active"));
        validateForm();
    });

    /* ── Validation on input ───────────────────────────────────── */
    if (secretMsg) secretMsg.addEventListener("input", validateForm);
    const coverText = document.getElementById("cover-text");
    if (coverText) coverText.addEventListener("input", validateForm);

    /* ── Encrypt ───────────────────────────────────────────────── */
    if (encryptBtn) {
        encryptBtn.addEventListener("click", async () => {
            const msg = secretMsg.value.trim();
            const pw = document.getElementById("enc-password")?.value || "";
            if (!msg) return;

            showOverlay("Encrypting...");
            encryptBtn.disabled = true;

            try {
                if (method === "text") {
                    let cover = "";
                    if (coverSource === "template") {
                        const prev = document.getElementById("tpl-text-preview");
                        cover = prev?.value || "";
                    } else { cover = document.getElementById("cover-text")?.value || ""; }
                    const r = await fetch("/api/encrypt-text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ secret_text: msg, cover_text: cover, password: pw }) });
                    const d = await r.json();
                    if (!r.ok) throw new Error(d.error || "Encryption failed");
                    showTextResult(d);
                }
                else if (method === "audio") {
                    const fd = new FormData(); fd.append("text", msg); fd.append("password", pw);
                    if (coverSource === "custom") { fd.append("audio", document.getElementById("enc-audio-file").files[0]); }
                    else if (coverSource === "template" && templateBlob) { fd.append("audio", templateBlob, "template.wav"); }
                    const r = await fetch("/api/encrypt-audio", { method: "POST", body: fd });
                    const d = await r.json();
                    if (!r.ok) throw new Error(d.error || "Encryption failed");
                    showAudioResult(d);
                }
                else if (method === "video") {
                    const fd = new FormData(); fd.append("text", msg); fd.append("password", pw);
                    if (coverSource === "custom") { fd.append("video", document.getElementById("enc-video-file").files[0]); }
                    else if (coverSource === "template" && templateBlob) { fd.append("video", templateBlob, "template.avi"); }

                    const r = await fetch("/api/encrypt-video", { method: "POST", body: fd });
                    const d = await r.json();
                    if (!r.ok) throw new Error(d.error || "Encryption failed");
                    showVideoResult(d);
                }
                else if (method === "image") {
                    const fd = new FormData(); fd.append("secret_text", msg); fd.append("password", pw);
                    if (coverSource === "template" && templateBlob) {
                        fd.append("cover_image", templateBlob, "template.png");
                    } else {
                        fd.append("cover_image", document.getElementById("enc-image-file").files[0]);
                    }
                    const r = await fetch("/api/encrypt-image", { method: "POST", body: fd });
                    const d = await r.json();
                    if (!r.ok) throw new Error(d.error || "Encryption failed");
                    showImageResult(d);
                }
                showToast("Message hidden successfully!");
            } catch (err) { showToast(err.message || "Encryption failed", "error"); }
            finally { hideOverlay(); validateForm(); }
        });
    }

    /* ── Result Display ────────────────────────────────────────── */
    function hideResultSections() {
        ["result-text-section", "result-audio-section", "result-video-section", "result-image-section"].forEach(id => {
            const el = document.getElementById(id); if (el) el.classList.add("hidden");
        });
    }
    function revealResultCard() {
        resultCard.classList.remove("hidden");
        // The result card has CSS class `decrypt-reveal` which keeps it at opacity:0
        // until `revealed` class is added. Use rAF for smooth transition.
        requestAnimationFrame(() => {
            resultCard.classList.add("revealed");
            resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
        // Fire success burst animation
        if (typeof triggerBurst === "function") triggerBurst(resultCard);
    }
    function showTextResult(d) {
        hideResultSections();
        revealResultCard();
        document.getElementById("result-text-section").classList.remove("hidden");
        document.getElementById("stego-output").value = d.stego_text || "";
        showMetrics(d);
    }
    function showAudioResult(d) {
        hideResultSections();
        revealResultCard();
        document.getElementById("result-audio-section").classList.remove("hidden");
        const player = document.getElementById("enc-audio-player");
        const dl = document.getElementById("enc-audio-download");
        if (d.audio_url) {
            player.src = d.audio_url;
            const filename = d.audio_url.split("/").pop();
            dl.href = "/api/download/" + filename;
        }
        showMetrics(d);
    }
    function showVideoResult(d) {
        hideResultSections();
        revealResultCard();
        document.getElementById("result-video-section").classList.remove("hidden");
        const dl = document.getElementById("enc-video-download");
        if (d.video_url) {
            const filename = d.video_url.split("/").pop();
            dl.href = "/api/download/" + filename;
        }
        showMetrics(d);
    }
    function showImageResult(d) {
        hideResultSections();
        revealResultCard();
        document.getElementById("result-image-section").classList.remove("hidden");
        const img = document.getElementById("enc-image-preview");
        const dl = document.getElementById("enc-image-download");
        if (d.stego_url) {
            img.src = d.stego_url;
            img.classList.add("anim-visible"); // trigger lazy reveal animation
        }
        if (d.stego_file) {
            dl.href = "/api/download/" + d.stego_file;
        }
        showMetrics(d);
    }
    function showMetrics(d) {
        const grid = document.getElementById("encrypt-metrics");
        if (!grid) return; grid.innerHTML = "";
        const items = [];
        const meta = d.metadata || {};
        if (meta.method) items.push(["Method", meta.method]);
        if (meta.capacity) items.push(["Capacity", meta.capacity + " B"]);
        if (meta.data_size) items.push(["Data Size", meta.data_size + " B"]);
        if (meta.psnr) items.push(["PSNR", meta.psnr + " dB"]);
        if (meta.ssim) items.push(["SSIM", String(meta.ssim)]);
        if (meta.duration) items.push(["Duration", meta.duration + "s"]);
        if (meta.frames) items.push(["Frames", String(meta.frames)]);
        if (meta.resolution) items.push(["Resolution", meta.resolution]);
        if (meta.sample_rate) items.push(["Sample Rate", meta.sample_rate + " Hz"]);
        if (d.encrypted !== undefined) items.push(["AES-256", d.encrypted ? "Yes" : "No"]);
        if (d.message) items.push(["Status", d.message]);
        items.forEach(([l, v]) => {
            grid.innerHTML += `<div class="metric-item"><span class="metric-label">${l}</span><span class="metric-value">${v}</span></div>`;
        });
    }

    /* ── Copy / Download buttons ───────────────────────────────── */
    document.getElementById("btn-copy-stego")?.addEventListener("click", () => {
        copyText(document.getElementById("stego-output")?.value || "");
    });
    document.getElementById("btn-download-stego-text")?.addEventListener("click", () => {
        const text = document.getElementById("stego-output")?.value || "";
        const blob = new Blob([text], { type: "text/plain" });
        const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
        a.download = "stego.txt"; a.click(); URL.revokeObjectURL(a.href);
    });

    /* ── Init ──────────────────────────────────────────────────── */
    updateCoverUI();
    validateForm();

})();
