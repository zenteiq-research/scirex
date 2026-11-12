/*
 * SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Home page benchmark gallery: a carousel of one slide per benchmark, with pagination
 * dots between previous and next arrows. The slides sit in a scroll-snap track, so
 * swiping and trackpads work natively; this script adds the arrows, the dots, the arrow
 * keys, autoplay, and plays only the clip of the slide in view. Without it the track still scrolls sideways.
 */
(function () {
  function setup() {
    var root = document.querySelector(".zq-carousel");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";
    var track = root.querySelector(".zq-carousel__track");
    var slides = Array.prototype.slice.call(track.children);
    var current = -1;

    function button(cls, label, text) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = cls;
      b.setAttribute("aria-label", label);
      b.textContent = text;
      return b;
    }

    var prev = button("zq-carousel__arrow zq-carousel__arrow--prev", "Previous benchmark", "‹");
    var next = button("zq-carousel__arrow zq-carousel__arrow--next", "Next benchmark", "›");
    var nav = document.createElement("div");
    nav.className = "zq-carousel__nav";
    var dots = document.createElement("div");
    dots.className = "zq-carousel__dots";
    var dotButtons = slides.map(function (slide, i) {
      var name = (slide.getAttribute("aria-label") || "").replace(/^\d+ of \d+: /, "");
      var d = button("zq-carousel__dot", "Show " + name, "");
      d.addEventListener("click", function () { go(i); });
      dots.appendChild(d);
      return d;
    });
    nav.appendChild(prev);
    nav.appendChild(dots);
    nav.appendChild(next);
    root.appendChild(nav);

    function offset(i) {
      return slides[i].offsetLeft - slides[0].offsetLeft;
    }

    function go(i) {
      i = (i + slides.length) % slides.length;
      track.scrollTo({ left: offset(i), behavior: "smooth" });
      mark(i);
    }

    function mark(i) {
      if (i === current) return;
      current = i;
      dotButtons.forEach(function (d, k) {
        if (k === i) d.setAttribute("aria-current", "true");
        else d.removeAttribute("aria-current");
      });
      slides.forEach(function (slide, k) {
        var v = slide.querySelector("video");
        if (!v) return;
        if (k === i) {
          v.preload = "auto";
          v.play().catch(function () {});
        } else {
          v.pause();
        }
      });
    }

    function nearest() {
      var x = track.scrollLeft;
      var best = 0;
      slides.forEach(function (slide, k) {
        if (Math.abs(offset(k) - x) < Math.abs(offset(best) - x)) best = k;
      });
      return best;
    }

    var timer;
    track.addEventListener("scroll", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { mark(nearest()); }, 150);
    }, { passive: true });
    prev.addEventListener("click", function () { go(current - 1); });
    next.addEventListener("click", function () { go(current + 1); });
    track.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { e.preventDefault(); go(current - 1); }
      if (e.key === "ArrowRight") { e.preventDefault(); go(current + 1); }
    });
    // Autoplay: the next slide when the clip of the slide in view has played once (after a few
    // seconds for a still image), paused while a mouse pointer is over the gallery, while a
    // control has keyboard focus, when the tab is hidden, and never for readers who ask for
    // reduced motion. A tap on a touch screen does not pause it. A click on a dot or an arrow
    // restarts the wait.
    var STILL_MS = 6000;
    var CLIP_MS = 12000;  // used until the length of the clip is known
    var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var hover = false;
    var focus = false;
    var auto = null;
    function stop() { clearTimeout(auto); auto = null; }
    function wait() {
      var v = slides[current] && slides[current].querySelector("video");
      if (!v) return STILL_MS;
      return isFinite(v.duration) && v.duration > 0 ? Math.max(v.duration * 1000, STILL_MS) : CLIP_MS;
    }
    function start() {
      stop();
      if (still || hover || focus || document.hidden) return;
      auto = setTimeout(function () { go(current + 1); start(); }, wait());
    }
    root.addEventListener("pointerenter", function (e) {
      if (e.pointerType !== "mouse") return;
      hover = true;
      stop();
    });
    root.addEventListener("pointerleave", function (e) {
      if (e.pointerType !== "mouse") return;
      hover = false;
      start();
    });
    root.addEventListener("focusin", function (e) {
      if (!e.target.matches(":focus-visible")) return;  // keyboard focus only, not a tap
      focus = true;
      stop();
    });
    root.addEventListener("focusout", function () { focus = false; start(); });
    nav.addEventListener("click", start);
    document.addEventListener("visibilitychange", start);
    slides.forEach(function (slide) {
      var v = slide.querySelector("video");
      if (v) v.addEventListener("loadedmetadata", function () { if (slide === slides[current]) start(); });
    });

    mark(0);
    start();
  }
  if (window.document$) window.document$.subscribe(setup);
  else document.addEventListener("DOMContentLoaded", setup);
})();
