const visitorMap = document.querySelector("[data-visitor-map]");

if (visitorMap) {
  const widgetId = visitorMap.dataset.widgetId.trim();
  const placeholder = visitorMap.querySelector("[data-visitor-placeholder]");
  const status = visitorMap.querySelector("[data-visitor-status]");
  const embed = visitorMap.querySelector("[data-visitor-embed]");
  let siteHost = "";
  try {
    const siteUrl = new URL(visitorMap.dataset.siteUrl);
    if (siteUrl.protocol === "https:") siteHost = siteUrl.hostname;
  } catch {
    // An incomplete local configuration must never activate a tracker.
  }

  // A widget needs its real provider-issued ID. Local previews never add visits.
  if (widgetId && siteHost && window.location.protocol === "https:" && window.location.hostname === siteHost) {
    const source = new URL("https://mapmyvisitors.com/globe.js");
    source.searchParams.set("d", widgetId);
    source.searchParams.set("w", "a");

    let finished = false;
    let timer;
    const finish = (loaded) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      observer.disconnect();
      placeholder.hidden = loaded;
      embed.hidden = !loaded;
      if (!loaded) status.textContent = "The visitor map is temporarily unavailable.";
    };

    let globe;
    let previousWidth = 0;
    const fitGlobe = () => {
      const availableWidth = embed.clientWidth;
      if (!globe || !availableWidth || availableWidth === previousWidth) return;
      previousWidth = availableWidth;
      const scale = Math.min(1, availableWidth / globe.offsetWidth);
      const offset = Math.max(0, (availableWidth - globe.offsetWidth * scale) / 2);
      globe.style.transformOrigin = "top left";
      globe.style.transform = `translateX(${offset}px) scale(${scale})`;
      embed.style.height = `${globe.offsetHeight * scale}px`;
    };

    const observeMap = () => {
      if (globe || !embed.querySelector(".mmvst_outer .dots_svg")) return;
      globe = embed.querySelector(".mmvst_outer");
      const graphic = globe.querySelector(".mmvst_globe");
      if (graphic) {
        graphic.setAttribute("role", "img");
        graphic.setAttribute("aria-label", "Globe showing approximate visitor locations");
      }
      requestAnimationFrame(() => {
        fitGlobe();
        if ("ResizeObserver" in window) {
          new ResizeObserver(fitGlobe).observe(embed);
        } else {
          window.addEventListener("resize", fitGlobe);
        }
        finish(true);
      });
    };

    const observer = new MutationObserver(observeMap);
    const script = document.createElement("script");
    script.id = "mmvst_globe";
    script.src = source.href;
    script.addEventListener("error", () => finish(false), { once: true });

    status.textContent = "Loading visitor map…";
    observer.observe(embed, { childList: true, subtree: true });
    timer = setTimeout(() => finish(false), 15000);
    embed.appendChild(script);
  } else if (widgetId) {
    status.textContent = "The visitor map is available on the live homepage.";
  }
}
