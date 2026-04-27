/* ═══════════════════════════════════════════════════════════════
   BIS Landing — 25 Interactive Elements Engine
   Particle Constellation · Wave Text · Glass Tilt · Click Ripple
   GPU-Accelerated · IntersectionObserver-Gated
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    /* ═══════════════════════════════════════════════════════════════
       ELEMENT 14: Particle Constellation (Canvas)
       Interactive — particles react to mouse position
       ═══════════════════════════════════════════════════════════════ */
    const canvas = document.getElementById("particle-canvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        let W, H;
        const PARTICLES = [];
        const MAX = 90;
        const CONNECT = 130;
        let mouseX = -999, mouseY = -999;
        const MOUSE_RADIUS = 160;

        function resize() {
            const dpr = window.devicePixelRatio || 1;
            W = window.innerWidth; H = window.innerHeight;
            canvas.width = W * dpr; canvas.height = H * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        window.addEventListener("resize", resize); resize();

        function Particle() {
            this.x = Math.random() * W;
            this.y = Math.random() * H;
            this.vx = (Math.random() - 0.5) * 0.4;
            this.vy = (Math.random() - 0.5) * 0.4;
            this.r = Math.random() * 2 + 1;
            this.baseAlpha = Math.random() * 0.5 + 0.15;
            this.alpha = this.baseAlpha;
        }
        Particle.prototype.update = function () {
            // Mouse repulsion
            const dx = this.x - mouseX, dy = this.y - mouseY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < MOUSE_RADIUS && dist > 0) {
                const force = (MOUSE_RADIUS - dist) / MOUSE_RADIUS * 0.8;
                this.vx += (dx / dist) * force * 0.3;
                this.vy += (dy / dist) * force * 0.3;
                this.alpha = Math.min(1, this.baseAlpha + force * 0.5);
            } else {
                this.alpha += (this.baseAlpha - this.alpha) * 0.05;
            }
            // Dampen velocity
            this.vx *= 0.99; this.vy *= 0.99;
            this.x += this.vx; this.y += this.vy;
            // Wrap
            if (this.x < 0) this.x = W; if (this.x > W) this.x = 0;
            if (this.y < 0) this.y = H; if (this.y > H) this.y = 0;
        };
        for (let i = 0; i < MAX; i++) PARTICLES.push(new Particle());

        // Track mouse for particle interaction
        document.addEventListener("mousemove", e => {
            mouseX = e.clientX; mouseY = e.clientY;
        }, { passive: true });
        document.addEventListener("mouseleave", () => {
            mouseX = -999; mouseY = -999;
        });

        let raf;
        function draw() {
            ctx.clearRect(0, 0, W, H);
            for (let i = 0; i < PARTICLES.length; i++) {
                const p = PARTICLES[i]; p.update();
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(99,102,241,${p.alpha})`; ctx.fill();
                for (let j = i + 1; j < PARTICLES.length; j++) {
                    const q = PARTICLES[j];
                    const ddx = p.x - q.x, ddy = p.y - q.y;
                    const d = Math.sqrt(ddx * ddx + ddy * ddy);
                    if (d < CONNECT) {
                        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
                        ctx.strokeStyle = `rgba(99,102,241,${0.12 * (1 - d / CONNECT)})`;
                        ctx.lineWidth = 0.6; ctx.stroke();
                    }
                }
            }
            raf = requestAnimationFrame(draw);
        }

        // Gate animation to viewport visibility
        const heroEl = document.getElementById("hero");
        if (heroEl) {
            const obs = new IntersectionObserver(entries => {
                entries.forEach(e => {
                    if (e.isIntersecting) { if (!raf) draw(); }
                    else { cancelAnimationFrame(raf); raf = null; }
                });
            }, { threshold: 0.02 });
            obs.observe(heroEl);
        } else { draw(); }
    }


    /* ═══════════════════════════════════════════════════════════════
       ELEMENT 15: Wave Text (Letter-by-Letter Animation)
       Customizable speed & amplitude via range controls
       ═══════════════════════════════════════════════════════════════ */
    const waveTitle = document.getElementById("wave-title");
    if (waveTitle) {
        const text = waveTitle.dataset.text || waveTitle.textContent;
        waveTitle.innerHTML = "";
        let idx = 0;
        for (const ch of text) {
            if (ch === " ") {
                waveTitle.appendChild(document.createTextNode("\u00A0"));
            } else {
                const span = document.createElement("span");
                span.className = "wave-letter";
                span.textContent = ch;
                span.style.setProperty("--i", idx);
                waveTitle.appendChild(span);
                idx++;
            }
        }

        // Speed control
        const speedCtrl = document.getElementById("wave-speed-ctrl");
        if (speedCtrl) {
            speedCtrl.addEventListener("input", () => {
                document.documentElement.style.setProperty("--wave-speed", speedCtrl.value + "s");
            });
        }
        // Amplitude control
        const ampCtrl = document.getElementById("wave-amp-ctrl");
        if (ampCtrl) {
            ampCtrl.addEventListener("input", () => {
                document.documentElement.style.setProperty("--wave-amplitude", "-" + ampCtrl.value + "px");
            });
        }
    }


    /* ═══════════════════════════════════════════════════════════════
       ELEMENT 16: Glass Hero Panel — 3D Tilt on Hover
       ═══════════════════════════════════════════════════════════════ */
    const heroPanel = document.getElementById("glass-hero");
    if (heroPanel) {
        let hoverActive = false;
        heroPanel.addEventListener("mouseenter", () => { hoverActive = true; });
        heroPanel.addEventListener("mouseleave", () => {
            hoverActive = false;
            heroPanel.style.transform = "perspective(800px) rotateX(0deg) rotateY(0deg)";
        });
        heroPanel.addEventListener("mousemove", e => {
            if (!hoverActive) return;
            const rect = heroPanel.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;  // -0.5 to 0.5
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            const tiltX = y * -6;  // degrees
            const tiltY = x * 6;
            heroPanel.style.transform = `perspective(800px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`;
        });
    }


    /* ═══════════════════════════════════════════════════════════════
       ELEMENTS 17-22: Glass Feature Cards — 3D Tilt on Hover
       ═══════════════════════════════════════════════════════════════ */
    document.querySelectorAll(".glass-feature-card").forEach(card => {
        let active = false;
        card.addEventListener("mouseenter", () => { active = true; });
        card.addEventListener("mouseleave", () => {
            active = false;
            card.style.transform = "perspective(800px) rotateX(0deg) rotateY(0deg) translateY(0)";
        });
        card.addEventListener("mousemove", e => {
            if (!active) return;
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            card.style.transform = `perspective(800px) rotateX(${y * -8}deg) rotateY(${x * 8}deg) translateY(-6px)`;
        });
    });


    /* ═══════════════════════════════════════════════════════════════
       ELEMENTS 23-24: Glass CTA Buttons — Press Animation
       (Handled via CSS, but we add a tactile pulse on click)
       ═══════════════════════════════════════════════════════════════ */
    document.querySelectorAll(".glass-cta").forEach(btn => {
        btn.addEventListener("click", e => {
            btn.style.transform = "scale(0.95)";
            setTimeout(() => { btn.style.transform = ""; }, 150);
        });
    });


    /* ═══════════════════════════════════════════════════════════════
       ELEMENT 25: Click Ripple Effect
       Creates a glass ripple at click position anywhere on page
       ═══════════════════════════════════════════════════════════════ */
    document.addEventListener("click", e => {
        const ripple = document.createElement("div");
        ripple.className = "click-ripple";
        ripple.style.left = e.clientX + "px";
        ripple.style.top = e.clientY + "px";
        document.body.appendChild(ripple);
        ripple.addEventListener("animationend", () => ripple.remove());
    });


    /* ═══════════════════════════════════════════════════════════════
       Scroll Entrance Animations
       Uses IntersectionObserver to trigger .visible class
       ═══════════════════════════════════════════════════════════════ */
    if (typeof IntersectionObserver !== "undefined") {
        const entranceObs = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    e.target.classList.add("visible");
                    entranceObs.unobserve(e.target);
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll(".enter-up, .enter-scale, .reveal").forEach(el => {
            entranceObs.observe(el);
        });
    }


    /* ═══════════════════════════════════════════════════════════════
       Showcase Tiles — Interactive Click Feedback
       ═══════════════════════════════════════════════════════════════ */
    document.querySelectorAll(".showcase-tile").forEach(tile => {
        tile.addEventListener("click", () => {
            tile.style.transition = "transform .1s";
            tile.style.transform = "scale(0.92)";
            setTimeout(() => {
                tile.style.transition = "transform .4s var(--ease)";
                tile.style.transform = "";
            }, 120);
        });
    });


    /* ═══════════════════════════════════════════════════════════════
       Elements 1-13: Background Parallax (subtle mouse tracking)
       Moves glass orbs slightly based on mouse, enhancing depth
       ═══════════════════════════════════════════════════════════════ */
    const orbs = document.querySelectorAll(".glass-orb");
    const shards = document.querySelectorAll(".glass-shard");
    const meshes = document.querySelectorAll(".gradient-mesh");

    if (orbs.length > 0) {
        let mx = 0, my = 0, cx = 0, cy = 0;
        document.addEventListener("mousemove", e => {
            mx = (e.clientX / window.innerWidth - 0.5) * 2;   // -1 to 1
            my = (e.clientY / window.innerHeight - 0.5) * 2;
        }, { passive: true });

        const speeds = [12, 8, 15, 6, 10];
        const shardSpeeds = [5, 7, 4];
        const meshSpeeds = [3, 4];

        function parallaxTick() {
            cx += (mx - cx) * 0.04;
            cy += (my - cy) * 0.04;

            orbs.forEach((orb, i) => {
                const s = speeds[i] || 8;
                orb.style.transform += ""; // keep CSS animation transform
                // We apply a CSS translate via a custom property instead
                orb.style.setProperty("--parallax-x", (cx * s) + "px");
                orb.style.setProperty("--parallax-y", (cy * s) + "px");
            });
            requestAnimationFrame(parallaxTick);
        }

        // Add parallax to orb CSS (via translate that compounds with animation)
        // We use a smarter approach: apply the parallax as a wrapper transform
        // Since orbs use CSS animation on transform, we add a container
        orbs.forEach(orb => {
            const wrapper = document.createElement("div");
            wrapper.style.cssText = "position:absolute;inset:0;pointer-events:none;will-change:transform;";
            orb.parentNode.insertBefore(wrapper, orb);
            // Don't move orb into wrapper — just track independently
        });

        // Simpler approach: just adjust top/left slightly
        function parallaxSimple() {
            cx += (mx - cx) * 0.03;
            cy += (my - cy) * 0.03;

            orbs.forEach((orb, i) => {
                const s = speeds[i] || 8;
                orb.style.marginLeft = (cx * s) + "px";
                orb.style.marginTop = (cy * s) + "px";
            });

            shards.forEach((shard, i) => {
                const s = shardSpeeds[i] || 5;
                shard.style.marginLeft = (cx * s * 0.5) + "px";
                shard.style.marginTop = (cy * s * 0.5) + "px";
            });

            meshes.forEach((mesh, i) => {
                const s = meshSpeeds[i] || 3;
                mesh.style.marginLeft = (cx * s) + "px";
                mesh.style.marginTop = (cy * s) + "px";
            });

            requestAnimationFrame(parallaxSimple);
        }
        parallaxSimple();
    }

})();
