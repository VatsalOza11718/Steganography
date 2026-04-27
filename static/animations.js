/* ═══════════════════════════════════════════════════════════════
   BIS Animation Engine — 54 Animations · 60fps · GPU-Accelerated
   Site-wide interactive experience controller
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /* ═══════════════════════════════════════════════════════════════
       1. LOADING SCREEN CONTROLLER (Anims 1-5)
       ═══════════════════════════════════════════════════════════════ */
    const loader = document.getElementById("page-loader");
    if (loader && !REDUCED) {
        const status = document.getElementById("loader-status");
        const msgs = ["Initializing modules", "Loading AI models", "Preparing interface", "Almost ready"];
        let mi = 0;
        const statusInterval = setInterval(() => {
            mi++;
            if (status && mi < msgs.length) status.textContent = msgs[mi];
        }, 500);

        function dismissLoader() {
            clearInterval(statusInterval);
            loader.classList.add("loader-exit");
            document.body.classList.add("content-loaded");
            setTimeout(() => {
                loader.classList.add("loader-done");
                initPostLoad();
            }, 700);
        }

        // Dismiss after content loaded or max 2.5s
        window.addEventListener("load", () => setTimeout(dismissLoader, 400));
        setTimeout(dismissLoader, 2500);
    } else if (loader) {
        loader.style.display = "none";
        document.body.classList.add("content-loaded");
        setTimeout(initPostLoad, 10);
    }

    /* ═══════════════════════════════════════════════════════════════
       2. POST-LOAD INITIALIZATION
       ═══════════════════════════════════════════════════════════════ */
    function initPostLoad() {
        initScrollProgress();
        initScrollReveals();
        initButtonRipples();
        initMagneticButtons();
        initTiltCards();
        initSpotlights();
        initFormAnimations();
        initBackToTop();
        initCounters();
        initWordCascades();
        initTimelines();
        initBitStream();
        initTypewriters();
        initSuccessBursts();
        initScanLines();
    }

    /* ═══════════════════════════════════════════════════════════════
       3. SCROLL PROGRESS BAR (Anim 14)
       ═══════════════════════════════════════════════════════════════ */
    function initScrollProgress() {
        const bar = document.getElementById("scroll-progress-bar");
        if (!bar) return;
        let ticking = false;
        window.addEventListener("scroll", () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    const h = document.documentElement.scrollHeight - window.innerHeight;
                    bar.style.width = h > 0 ? (window.scrollY / h * 100) + "%" : "0";
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
    }

    /* ═══════════════════════════════════════════════════════════════
       4. SCROLL REVEAL OBSERVER (Anims 16-18, 41-42, 45-46)
       ═══════════════════════════════════════════════════════════════ */
    function initScrollReveals() {
        if (typeof IntersectionObserver === "undefined") return;
        const obs = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    e.target.classList.add("anim-visible");
                    if (!e.target.dataset.animRepeat) obs.unobserve(e.target);
                }
            });
        }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });

        const selectors = [
            ".anim-reveal-up", ".anim-reveal-scale",
            ".anim-reveal-left", ".anim-reveal-right",
            ".anim-stagger-left", ".anim-stagger-right",
            ".timeline-node", ".arch-block",
            ".anim-img-reveal",
            ".enter-up", ".enter-scale", ".reveal"
        ];
        document.querySelectorAll(selectors.join(",")).forEach(el => obs.observe(el));
    }

    /* ═══════════════════════════════════════════════════════════════
       5. BUTTON CLICK RIPPLE (Anim 20)
       ═══════════════════════════════════════════════════════════════ */
    function initButtonRipples() {
        document.addEventListener("click", e => {
            const btn = e.target.closest(".ripple-container, .btn, .glass-cta, .method-card, .template-card, .showcase-tile");
            if (!btn) return;
            const rect = btn.getBoundingClientRect();
            const ring = document.createElement("span");
            ring.className = "ripple-ring";
            const size = Math.max(rect.width, rect.height);
            ring.style.width = ring.style.height = size + "px";
            ring.style.left = (e.clientX - rect.left - size / 2) + "px";
            ring.style.top = (e.clientY - rect.top - size / 2) + "px";
            btn.style.position = btn.style.position || "relative";
            btn.style.overflow = "hidden";
            btn.appendChild(ring);
            ring.addEventListener("animationend", () => ring.remove());
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       6. MAGNETIC HOVER (Anim 21)
       ═══════════════════════════════════════════════════════════════ */
    function initMagneticButtons() {
        document.querySelectorAll(".anim-magnetic").forEach(el => {
            el.addEventListener("mousemove", e => {
                const rect = el.getBoundingClientRect();
                const x = (e.clientX - rect.left - rect.width / 2) * 0.15;
                const y = (e.clientY - rect.top - rect.height / 2) * 0.15;
                el.style.transform = `translate3d(${x}px, ${y}px, 0)`;
            });
            el.addEventListener("mouseleave", () => {
                el.style.transform = "translate3d(0,0,0)";
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       7. 3D TILT CARDS (Anim 22)
       ═══════════════════════════════════════════════════════════════ */
    function initTiltCards() {
        document.querySelectorAll(".anim-tilt").forEach(card => {
            card.addEventListener("mousemove", e => {
                const rect = card.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                card.style.transform = `perspective(800px) rotateX(${y * -8}deg) rotateY(${x * 8}deg) translateY(-4px)`;
            });
            card.addEventListener("mouseleave", () => {
                card.style.transform = "perspective(800px) rotateX(0) rotateY(0) translateY(0)";
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       8. CURSOR SPOTLIGHT (Anim 24)
       ═══════════════════════════════════════════════════════════════ */
    function initSpotlights() {
        document.querySelectorAll(".anim-spotlight").forEach(section => {
            const glow = document.createElement("div");
            glow.className = "spotlight-glow";
            section.appendChild(glow);
            section.addEventListener("mousemove", e => {
                const rect = section.getBoundingClientRect();
                glow.style.left = (e.clientX - rect.left) + "px";
                glow.style.top = (e.clientY - rect.top) + "px";
            });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       9. FORM ANIMATIONS (Anims 25-29)
       ═══════════════════════════════════════════════════════════════ */
    function initFormAnimations() {
        // Add anim-input to all text inputs and textareas
        document.querySelectorAll("input[type='text'], input[type='password'], input[type='email'], textarea, select").forEach(inp => {
            inp.classList.add("anim-input");
        });

        // Floating labels
        document.querySelectorAll(".anim-float-group input, .anim-float-group textarea").forEach(inp => {
            if (!inp.placeholder) inp.placeholder = " ";
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       10. BACK TO TOP (Anim 53)
       ═══════════════════════════════════════════════════════════════ */
    function initBackToTop() {
        const btn = document.getElementById("back-to-top");
        if (!btn) return;

        let ticking = false;
        window.addEventListener("scroll", () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    if (window.scrollY > 400) btn.classList.add("visible");
                    else btn.classList.remove("visible");
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });

        btn.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       11. COUNTER ANIMATION (Anim 18 variant)
       ═══════════════════════════════════════════════════════════════ */
    function initCounters() {
        if (typeof IntersectionObserver === "undefined") return;
        const obs = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (!e.isIntersecting) return;
                const el = e.target;
                obs.unobserve(el);
                const target = parseInt(el.dataset.countTo || el.dataset.count, 10);
                if (isNaN(target)) return;
                const duration = 1500;
                const start = performance.now();
                const suffix = el.dataset.countSuffix || "";
                function tick(now) {
                    const p = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
                    el.textContent = Math.round(target * eased) + suffix;
                    if (p < 1) requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
            });
        }, { threshold: 0.3 });

        document.querySelectorAll("[data-count-to], [data-count]").forEach(el => obs.observe(el));
    }

    /* ═══════════════════════════════════════════════════════════════
       12. WORD CASCADE (Anim 43)
       ═══════════════════════════════════════════════════════════════ */
    function initWordCascades() {
        document.querySelectorAll(".word-cascade").forEach(el => {
            if (el.dataset.split === "done") return;
            const text = el.textContent;
            el.innerHTML = "";
            text.split(/\s+/).forEach((word, i) => {
                const span = document.createElement("span");
                span.className = "word";
                span.textContent = word;
                span.style.transitionDelay = (i * 0.06) + "s";
                el.appendChild(span);
                el.appendChild(document.createTextNode(" "));
            });
            el.dataset.split = "done";
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       13. TIMELINE SCROLL BUILD (Anim 41)
       ═══════════════════════════════════════════════════════════════ */
    function initTimelines() {
        if (typeof IntersectionObserver === "undefined") return;
        document.querySelectorAll(".timeline").forEach(tl => {
            const obs = new IntersectionObserver(entries => {
                entries.forEach(e => {
                    if (e.isIntersecting) {
                        tl.classList.add("anim-visible");
                        // Stagger children
                        tl.querySelectorAll(".timeline-node").forEach((node, i) => {
                            setTimeout(() => node.classList.add("anim-visible"), 200 + i * 200);
                        });
                        obs.unobserve(tl);
                    }
                });
            }, { threshold: 0.15 });
            obs.observe(tl);
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       14. BIT STREAM CANVAS (Anim 32)
       ═══════════════════════════════════════════════════════════════ */
    function initBitStream() {
        document.querySelectorAll(".bit-stream-canvas").forEach(canvas => {
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            let raf;

            function resize() {
                const p = canvas.parentElement;
                if (!p) return;
                canvas.width = p.offsetWidth;
                canvas.height = p.offsetHeight;
            }
            resize();

            const cols = [];
            function init() {
                const count = Math.floor(canvas.width / 14);
                cols.length = 0;
                for (let i = 0; i < count; i++) cols.push(Math.random() * canvas.height);
            }
            init();

            function draw() {
                ctx.fillStyle = "rgba(6,8,15,.12)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "rgba(99,102,241,.6)";
                ctx.font = "11px monospace";
                for (let i = 0; i < cols.length; i++) {
                    const ch = Math.random() > 0.5 ? "1" : "0";
                    ctx.fillText(ch, i * 14, cols[i]);
                    if (cols[i] > canvas.height && Math.random() > 0.98) cols[i] = 0;
                    cols[i] += 12;
                }
                raf = requestAnimationFrame(draw);
            }

            // Only animate when visible
            if (typeof IntersectionObserver !== "undefined") {
                const obs = new IntersectionObserver(entries => {
                    entries.forEach(e => {
                        if (e.isIntersecting) { resize(); init(); draw(); }
                        else { cancelAnimationFrame(raf); raf = null; }
                    });
                }, { threshold: 0.05 });
                obs.observe(canvas);
            }
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       15. TYPEWRITER EFFECT (Anim 38)
       ═══════════════════════════════════════════════════════════════ */
    function initTypewriters() {
        // Typewriter applied dynamically when decrypt result appears
        window.typewriterReveal = function (el, text, speed) {
            speed = speed || 25;
            el.textContent = "";
            el.classList.add("anim-typewriter");
            let i = 0;
            function type() {
                if (i < text.length) {
                    el.textContent += text.charAt(i);
                    i++;
                    setTimeout(type, speed);
                } else {
                    el.classList.remove("anim-typewriter");
                }
            }
            type();
        };
    }

    /* ═══════════════════════════════════════════════════════════════
       16. SUCCESS BURST PARTICLES (Anim 35)
       ═══════════════════════════════════════════════════════════════ */
    function initSuccessBursts() {
        window.triggerBurst = function (el) {
            const rect = el.getBoundingClientRect();
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const colors = ["#6366f1", "#a855f7", "#06b6d4", "#10b981", "#f59e0b"];
            for (let i = 0; i < 20; i++) {
                const p = document.createElement("div");
                p.className = "burst-particle";
                p.style.left = cx + "px";
                p.style.top = cy + "px";
                p.style.background = colors[i % colors.length];
                p.style.position = "fixed";
                const angle = (Math.PI * 2 / 20) * i;
                const dist = 60 + Math.random() * 80;
                const tx = Math.cos(angle) * dist;
                const ty = Math.sin(angle) * dist;
                p.style.setProperty("--tx", tx + "px");
                p.style.setProperty("--ty", ty + "px");
                p.style.animation = `burstFlyDir .7s ease forwards`;
                document.body.appendChild(p);
                p.addEventListener("animationend", () => p.remove());
            }
        };
    }

    /* Add dynamic keyframe for directional burst */
    (function () {
        const style = document.createElement("style");
        style.textContent = `
    @keyframes burstFlyDir {
      0% { transform:translate(0,0) scale(1); opacity:1 }
      100% { transform:translate(var(--tx),var(--ty)) scale(0); opacity:0 }
    }
  `;
        document.head.appendChild(style);
    })();

    /* ═══════════════════════════════════════════════════════════════
       17. SCAN LINE (Anim 37)
       ═══════════════════════════════════════════════════════════════ */
    function initScanLines() {
        window.startScanLine = function (container) {
            container.classList.add("scanning");
        };
        window.stopScanLine = function (container) {
            container.classList.remove("scanning");
        };
    }

    /* ═══════════════════════════════════════════════════════════════
       18. MODAL SYSTEM (Anim 52)
       ═══════════════════════════════════════════════════════════════ */
    window.openModal = function (id) {
        const m = document.getElementById(id);
        if (m) { m.classList.add("modal-open"); document.body.style.overflow = "hidden"; }
    };
    window.closeModal = function (id) {
        const m = document.getElementById(id);
        if (m) { m.classList.remove("modal-open"); document.body.style.overflow = ""; }
    };
    // Close modal on backdrop click
    document.addEventListener("click", e => {
        if (e.target.classList.contains("modal-backdrop") && e.target.classList.contains("modal-open")) {
            e.target.classList.remove("modal-open");
            document.body.style.overflow = "";
        }
    });
    // Close modal on Escape
    document.addEventListener("keydown", e => {
        if (e.key === "Escape") {
            document.querySelectorAll(".modal-open").forEach(m => {
                m.classList.remove("modal-open");
                document.body.style.overflow = "";
            });
        }
    });

    /* ═══════════════════════════════════════════════════════════════
       19. ENCRYPT PAGE HELPERS (Anims 30-35)
       ═══════════════════════════════════════════════════════════════ */
    window.animateCapacityMeter = function (el, percent) {
        const fill = el.querySelector(".capacity-meter-fill");
        if (fill) {
            requestAnimationFrame(() => { fill.style.width = Math.min(percent, 100) + "%"; });
        }
    };

    window.animateProgressRing = function (el, percent) {
        const fill = el.querySelector(".ring-fill");
        if (fill) {
            const circumference = 220;
            fill.style.strokeDashoffset = circumference - (circumference * percent / 100);
        }
    };

    /* ═══════════════════════════════════════════════════════════════
       20. PERFORMANCE MONITOR (opt-in debug)
       ═══════════════════════════════════════════════════════════════ */
    window.BISAnimPerf = {
        enabled: false,
        frames: [],
        lastTime: 0,
        start: function () {
            this.enabled = true;
            this.lastTime = performance.now();
            this._tick();
        },
        _tick: function () {
            if (!this.enabled) return;
            const now = performance.now();
            const delta = now - this.lastTime;
            this.lastTime = now;
            this.frames.push(1000 / delta);
            if (this.frames.length > 120) this.frames.shift();
            requestAnimationFrame(() => this._tick());
        },
        getAvgFPS: function () {
            if (!this.frames.length) return 0;
            return Math.round(this.frames.reduce((a, b) => a + b) / this.frames.length);
        },
        report: function () {
            const avg = this.getAvgFPS();
            const min = Math.round(Math.min(...this.frames));
            console.log(`[BIS Anim] FPS: avg=${avg}, min=${min}, samples=${this.frames.length}`);
            return { avg, min, samples: this.frames.length };
        }
    };

    /* ═══════════════════════════════════════════════════════════════
       INITIALIZE ON DOM READY
       ═══════════════════════════════════════════════════════════════ */
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            if (!loader || REDUCED) initPostLoad();
        });
    } else {
        if (!loader || REDUCED) initPostLoad();
    }

})();
