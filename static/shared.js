/* ═══════════════════════════════════════════════════════════════
   BIS — Shared UI Logic (navbar, toast, overlay, utilities)
   Loaded on ALL pages via base.html
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    /* ── Toast ─────────────────────────────────────────────────── */
    const toast = document.getElementById("toast");
    let _toastTimer = null;
    window.showToast = function (msg, type = "success", dur = 3500) {
        if (!toast) return;
        clearTimeout(_toastTimer);
        toast.textContent = msg;
        toast.className = "toast " + type;
        requestAnimationFrame(() => { toast.classList.add("show"); });
        _toastTimer = setTimeout(() => { toast.classList.remove("show"); setTimeout(() => toast.classList.add("hidden"), 350); }, dur);
    };

    /* ── Overlay ───────────────────────────────────────────────── */
    const overlay = document.getElementById("loading-overlay");
    window.showOverlay = function (msg) {
        if (!overlay) return;
        const p = overlay.querySelector("p");
        if (p && msg) p.textContent = msg;
        overlay.classList.remove("hidden");
    };
    window.hideOverlay = function () { if (overlay) overlay.classList.add("hidden"); };

    /* ── Navbar: Scroll Effect ─────────────────────────────────── */
    const nav = document.querySelector(".navbar");
    let _ticking = false;
    function onScroll() {
        if (!_ticking) {
            requestAnimationFrame(() => {
                if (nav) {
                    if (window.scrollY > 30) nav.classList.add("scrolled");
                    else nav.classList.remove("scrolled");
                }
                _ticking = false;
            });
            _ticking = true;
        }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    /* ── Navbar: Mobile Toggle ─────────────────────────────────── */
    const toggle = document.getElementById("nav-toggle");
    const links = document.getElementById("nav-links");
    if (toggle && links) {
        toggle.addEventListener("click", () => {
            const open = links.classList.toggle("open");
            toggle.setAttribute("aria-expanded", open);
        });
        // close on outside click
        document.addEventListener("click", e => {
            if (!toggle.contains(e.target) && !links.contains(e.target)) {
                links.classList.remove("open");
                toggle.setAttribute("aria-expanded", "false");
            }
        });
    }

    /* ── Drop-zone helper ──────────────────────────────────────── */
    window.initDropZone = function (zoneId, inputId, badgeId, onFile) {
        const zone = document.getElementById(zoneId);
        const inp = document.getElementById(inputId);
        const badge = document.getElementById(badgeId);
        if (!zone || !inp) return;
        zone.addEventListener("click", () => inp.click());
        zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
        zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
        zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("dragover"); if (e.dataTransfer.files.length) { inp.files = e.dataTransfer.files; handle(); } });
        inp.addEventListener("change", handle);
        function handle() {
            if (inp.files.length) {
                if (badge) { badge.textContent = "✓ " + inp.files[0].name; badge.classList.remove("hidden"); }
                if (onFile) onFile(inp.files[0]);
            }
        }
    };

    /* ── Copy to clipboard ─────────────────────────────────────── */
    window.copyText = function (text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => showToast("Copied to clipboard!")).catch(() => showToast("Copy failed", "error"));
        } else {
            const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); document.body.removeChild(t); showToast("Copied!");
        }
    };

    /* ── Duration formatter ────────────────────────────────────── */
    window.fmtDuration = function (s) {
        const m = Math.floor(s / 60); const sec = s % 60;
        return m + ":" + (sec < 10 ? "0" : "") + sec;
    };

    /* ── Scroll-reveal observer ────────────────────────────────── */
    if (typeof IntersectionObserver !== "undefined") {
        const revealObs = new IntersectionObserver(entries => {
            entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("visible"); revealObs.unobserve(e.target); } });
        }, { threshold: .12 });
        document.querySelectorAll(".reveal, .enter-up, .enter-scale").forEach(el => revealObs.observe(el));
    }

})();
