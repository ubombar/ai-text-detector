(() => {
  const $ = (id) => document.getElementById(id);
  const text = $("text"), btn = $("analyze");
  if (!text) return;

  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };
  const countWords = (s) => (s.trim().match(/\S+/g) || []).length;
  const LABEL = { ai: "AI", human: "Human", uncertain: "Uncertain", inconclusive: "Inconclusive" };

  text.addEventListener("input", () => {
    const n = countWords(text.value);
    $("wordcount").textContent = `${n} word${n === 1 ? "" : "s"}`;
  });

  btn.addEventListener("click", async () => {
    $("error").hidden = true;
    $("results").hidden = true;
    $("loading").hidden = false;
    btn.disabled = true;
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ text: text.value }),
      });
      let data = null;
      try { data = await res.json(); } catch (_) {}
      if (!res.ok) {
        const d = data && data.detail;
        throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `Request failed (${res.status})`);
      }
      render(data);
    } catch (e) {
      $("error").textContent = e.message || String(e);
      $("error").hidden = false;
    } finally {
      $("loading").hidden = true;
      btn.disabled = false;
    }
  });

  function render(d) {
    const ov = $("overall");
    ov.className = `verdict v-${d.overall.verdict}`;
    ov.replaceChildren(el("strong", null, LABEL[d.overall.verdict] || d.overall.verdict), el("p", null, d.overall.summary));

    $("warning").hidden = !d.warning;
    $("warning").textContent = d.warning || "";
    $("meta").textContent = `${d.n_words} words · ${d.n_tokens} tokens · ${Number(d.elapsed_s).toFixed(1)} s`;

    $("methods").replaceChildren(...d.methods.map((m) => {
      const c = el("div", "card method");
      const h = el("h3", null, m.name);
      h.append(el("span", `pill v-${m.verdict}`, LABEL[m.verdict] || m.verdict));
      const hint = m.direction === "lower_is_ai" ? "↓ lower = AI" : "↑ higher = AI";
      c.append(h, el("div", "score", Number(m.score).toFixed(3)),
        el("div", "muted small", `threshold ${Number(m.threshold).toFixed(3)} · ${hint}`));
      if (m.detail) c.append(el("div", "small", m.detail));
      return c;
    }));

    const g = d.gltr || { tokens: [], fractions: [0, 0, 0, 0] };
    $("gltr-bar").replaceChildren(...g.fractions.map((f, i) => {
      const s = el("div", `b${i}`);
      s.style.width = `${(f * 100).toFixed(2)}%`;
      s.title = `${(f * 100).toFixed(1)}%`;
      return s;
    }));
    const frag = document.createDocumentFragment();
    for (const t of g.tokens) {
      const s = el("span", `tok b${t.bucket}`, t.text);
      s.title = `rank ${t.rank}`;
      frag.append(s);
    }
    $("gltr-tokens").replaceChildren(frag);

    $("sentences").replaceChildren(...(d.sentences || []).flatMap((s) => {
      const sp = el("span", `s-${s.verdict}`, s.text);
      sp.title = `score ${Number(s.score).toFixed(3)} · ${LABEL[s.verdict] || s.verdict}`;
      return [sp, document.createTextNode(" ")];
    }));

    $("results").hidden = false;
  }
})();
