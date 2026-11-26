/* ==========================================================================
   tutor.js — Instant Zero-Key Public LLM + Optional Session BYOK Study Tutor
   Supports out-of-the-box chat on static GitHub Pages with ZERO API key:
     1. Public Cloud (Free, No Key): GPT-OSS 20B Reasoning via Pollinations OVH
     2. Browser WebGPU (Local, No Key): Qwen2.5-1.5B / Llama-3.2-1B via WebLLM
     3. Optional BYOK (Session-Only): Google Gemini, Anthropic Claude, OpenAI, OpenRouter
   ========================================================================== */
(function () {
  "use strict";

  const PROVIDERS = {
    public_cloud: {
      label: "Public Cloud (Free · No Key)",
      needsKey: false,
      models: [
        { id: "openai-fast", name: "GPT-OSS 20B Reasoning (Instant)" }
      ],
      note: "<strong>✓ Ready to chat immediately:</strong> Connected to the public open-weights <code>GPT-OSS 20B</code> endpoint (no API key required). Switch provider above anytime to use your own Gemini, Claude, or OpenAI key.",
    },
    webgpu: {
      label: "Local WebGPU (In-Browser · No Key)",
      needsKey: false,
      models: [
        { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen 2.5 1.5B (WebGPU)" },
        { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", name: "Llama 3.2 1B (WebGPU)" }
      ],
      note: "<strong>100% On-Device WebGPU:</strong> Downloads open weights once from HuggingFace into browser cache (~900 MB) and runs locally on your GPU with zero external servers.",
    },
    gemini: {
      label: "Google Gemini (BYOK)",
      needsKey: true,
      models: [
        { id: "gemini-2.5-flash", name: "gemini-2.5-flash" },
        { id: "gemini-2.5-pro", name: "gemini-2.5-pro" },
        { id: "gemini-2.0-flash", name: "gemini-2.0-flash" }
      ],
      placeholder: "AIzaSy... (Google AI Studio API key)",
      note: "<strong>Session-only privacy:</strong> Your key stays in this browser tab's memory (<code>sessionStorage</code>) and is wiped when the tab closes.",
    },
    anthropic: {
      label: "Anthropic Claude (BYOK)",
      needsKey: true,
      models: [
        { id: "claude-sonnet-4-20250514", name: "claude-sonnet-4" },
        { id: "claude-3-5-haiku-latest", name: "claude-3-5-haiku" }
      ],
      placeholder: "sk-ant-... (Anthropic API key)",
      note: "<strong>Session-only privacy:</strong> Your key stays in this browser tab's memory (<code>sessionStorage</code>) and connects directly to Anthropic.",
    },
    openai: {
      label: "OpenAI ChatGPT (BYOK)",
      needsKey: true,
      models: [
        { id: "gpt-4.1-mini", name: "gpt-4.1-mini" },
        { id: "gpt-4.1", name: "gpt-4.1" },
        { id: "o4-mini", name: "o4-mini" }
      ],
      placeholder: "sk-... (OpenAI API key)",
      note: "<strong>Session-only privacy:</strong> Your key stays in this browser tab's memory (<code>sessionStorage</code>) and connects directly to OpenAI.",
    },
    openrouter: {
      label: "OpenRouter (BYOK)",
      needsKey: true,
      models: [
        { id: "google/gemini-2.5-flash", name: "gemini-2.5-flash" },
        { id: "anthropic/claude-sonnet-4", name: "claude-sonnet-4" },
        { id: "openai/gpt-4.1-mini", name: "gpt-4.1-mini" }
      ],
      placeholder: "sk-or-... (OpenRouter API key)",
      note: "<strong>Session-only privacy:</strong> Your key stays in this browser tab's memory (<code>sessionStorage</code>) and connects directly to OpenRouter.",
    },
  };

  const SYSTEM_PROMPT = `You are a concise, rigorous robotics and control-theory study tutor embedded inside an interactive textbook chapter on Hamilton-Jacobi Reachability (HJR), Control Barrier Functions (CBFs / CBVFs), Artificial Potential Fields (APFs), Physics-Informed Neural Networks (DeepReach), and aerial robotics planning (FaSTrack, Reach-Avoid dock landing, Dubins Air3D).
Guidelines:
- Explain from first principles with clear physical intuition first, then crisp equations using \\( ... \\) for inline LaTeX and \\[ ... \\] for display LaTeX.
- Keep answers focused (2–4 short paragraphs unless asked for a full derivation).
- Reference the interactive labs on the page when helpful so the reader can test ideas visually.`;

  window.addEventListener("DOMContentLoaded", () => {
    const HJR = window.HJR;
    if (!HJR) return;
    const { $, $$ } = HJR;

    const openBtn = $("#tutor-open");
    const closeBtn = $("#tutor-close");
    const drawer = $("#tutor-drawer");
    const backdrop = $("#tutor-backdrop");
    if (!openBtn || !drawer) return;

    const provSel = $("#tutor-provider");
    const modSel = $("#tutor-model");
    const keyRow = $("#tutor-key-row");
    const keyInp = $("#tutor-key");
    const clearBtn = $("#tutor-clear-key");
    const privacyNote = $("#tutor-privacy-note");
    const msgBox = $("#tutor-messages");
    const txtInp = $("#tutor-input");
    const sendBtn = $("#tutor-send");

    const history = [];
    let webllmEngine = null;
    let webllmLoadedModel = null;

    const toggleDrawer = (show) => {
      drawer.classList.toggle("open", show);
      backdrop.classList.toggle("open", show);
      if (show) txtInp.focus();
    };
    openBtn.addEventListener("click", () => toggleDrawer(true));
    closeBtn.addEventListener("click", () => toggleDrawer(false));
    backdrop.addEventListener("click", () => toggleDrawer(false));

    function populateModels() {
      const p = PROVIDERS[provSel.value] || PROVIDERS.public_cloud;
      modSel.innerHTML = p.models.map((m) => `<option value="${m.id}">${m.name}</option>`).join("");
      if (keyRow) keyRow.style.display = p.needsKey ? "flex" : "none";
      if (privacyNote) privacyNote.innerHTML = p.note;
      if (p.needsKey && keyInp) {
        keyInp.placeholder = p.placeholder || "Session API key...";
        try {
          keyInp.value = sessionStorage.getItem("hjr_byok_" + provSel.value) || "";
        } catch (_) {}
      }
    }

    provSel.addEventListener("change", populateModels);
    if (keyInp) {
      keyInp.addEventListener("input", () => {
        try { sessionStorage.setItem("hjr_byok_" + provSel.value, keyInp.value.trim()); } catch (_) {}
      });
    }
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        if (keyInp) keyInp.value = "";
        try {
          Object.keys(PROVIDERS).forEach((k) => sessionStorage.removeItem("hjr_byok_" + k));
        } catch (_) {}
        appendBubble("assistant", "Cleared session API keys from this browser tab.");
      });
    }
    populateModels();

    function getActiveSectionContext() {
      const activeNav = $(".toc-nav a.active");
      if (!activeNav) return "Reader is at Section 1 (Point of No Return).";
      const sec = $(activeNav.getAttribute("href"));
      if (!sec) return "";
      const h2 = $("h2, h1", sec);
      const readout = $(".hud-readout", sec);
      return `Reader is currently viewing section "${h2 ? h2.textContent.trim() : activeNav.textContent}"` +
        (readout ? `. Live simulator telemetry in this section: ${readout.textContent.replace(/\s+/g, " ").trim().slice(0, 320)}` : "");
    }

    function escapeHtml(s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function formatMarkdownLite(text) {
      let out = escapeHtml(text);
      out = out.replace(/```([\s\S]*?)```/g, "<pre style='overflow-x:auto;padding:0.5rem;background:var(--bg-hud);border-radius:6px;font-family:var(--mono);font-size:0.8rem'><code>$1</code></pre>");
      out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
      out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      out = out.replace(/\n\n+/g, "</p><p style='margin-top:0.55rem'>");
      return `<p>${out}</p>`;
    }

    function appendBubble(role, text) {
      const div = document.createElement("div");
      div.className = "tutor-msg " + role;
      div.innerHTML = role === "assistant" ? formatMarkdownLite(text) : escapeHtml(text);
      msgBox.appendChild(div);
      if (role === "assistant" && HJR.typeset) HJR.typeset(div);
      msgBox.scrollTop = msgBox.scrollHeight;
      return div;
    }

    async function callLLM(provider, model, apiKey, messages, sysPrompt, onStatus) {
      /* 1. Public Zero-Key Cloud Endpoint (GPT-OSS 20B Reasoning on Pollinations OVH) */
      if (provider === "public_cloud") {
        const res = await fetch("https://text.pollinations.ai/openai", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: model || "openai-fast",
            messages: [{ role: "system", content: sysPrompt }, ...messages],
            temperature: 0.35,
          }),
        });
        if (!res.ok) {
          throw new Error(`Public model temporarily busy (HTTP ${res.status}). Retry in a moment, switch to Local WebGPU, or add your own API key above.`);
        }
        const raw = await res.text();
        try {
          const j = JSON.parse(raw);
          return j.choices?.[0]?.message?.content || raw || "(Empty response)";
        } catch (_) {
          return raw || "(Empty response)";
        }
      }

      /* 2. Local In-Browser WebGPU Model via @mlc-ai/web-llm (No Key, Runs on User GPU) */
      if (provider === "webgpu") {
        if (!("gpu" in navigator)) {
          throw new Error("WebGPU is not enabled in this browser. Try Chrome/Edge 113+, or switch to 'Public Cloud (Free · No Key)' above.");
        }
        if (!webllmEngine || webllmLoadedModel !== model) {
          onStatus && onStatus("Loading WebLLM runtime…");
          const webllm = await import("https://esm.run/@mlc-ai/web-llm");
          webllmEngine = await webllm.CreateMLCEngine(model, {
            initProgressCallback: (rep) => {
              onStatus && onStatus(rep.text || `Downloading model weights (${Math.round((rep.progress || 0) * 100)}%)…`);
            },
          });
          webllmLoadedModel = model;
        }
        onStatus && onStatus("Generating locally on GPU…");
        const reply = await webllmEngine.chat.completions.create({
          messages: [{ role: "system", content: sysPrompt }, ...messages],
          temperature: 0.35,
          max_tokens: 700,
        });
        return reply.choices?.[0]?.message?.content || "(Empty response)";
      }

      /* 3. Google Gemini BYOK */
      if (provider === "gemini") {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
        const contents = messages.map((m) => ({
          role: m.role === "assistant" ? "model" : "user",
          parts: [{ text: m.content }],
        }));
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            systemInstruction: { parts: [{ text: sysPrompt }] },
            contents,
            generationConfig: { temperature: 0.35, maxOutputTokens: 900 },
          }),
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.error?.message || `Gemini HTTP ${res.status}`);
        return j.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "(Empty response)";
      }

      /* 4. Anthropic Claude BYOK */
      if (provider === "anthropic") {
        const res = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
          },
          body: JSON.stringify({
            model,
            max_tokens: 900,
            system: sysPrompt,
            messages,
          }),
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.error?.message || `Anthropic HTTP ${res.status}`);
        return j.content?.map((c) => c.text || "").join("") || "(Empty response)";
      }

      /* 5. OpenAI or OpenRouter BYOK */
      const endpoint = provider === "openrouter"
        ? "https://openrouter.ai/api/v1/chat/completions"
        : "https://api.openai.com/v1/chat/completions";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "system", content: sysPrompt }, ...messages],
          max_tokens: 900,
        }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error?.message || `HTTP ${res.status}`);
      return j.choices?.[0]?.message?.content || "(Empty response)";
    }

    async function sendMessage(textOverride) {
      const q = (textOverride ?? txtInp.value).trim();
      if (!q) return;
      const prov = provSel.value;
      const pCfg = PROVIDERS[prov] || PROVIDERS.public_cloud;
      const key = keyInp ? keyInp.value.trim() : "";

      if (pCfg.needsKey && !key) {
        appendBubble("assistant", "Paste a session API key above for this provider, or switch to **Public Cloud (Free · No Key)** to chat immediately without a key.");
        if (keyInp) keyInp.focus();
        return;
      }
      txtInp.value = "";
      appendBubble("user", q);
      history.push({ role: "user", content: q });
      const pending = appendBubble("assistant", "Thinking…");
      sendBtn.disabled = true;

      try {
        const sysWithCtx = `${SYSTEM_PROMPT}\n\n[Live Page Context]: ${getActiveSectionContext()}`;
        const reply = await callLLM(
          prov,
          modSel.value,
          key,
          history.slice(-10),
          sysWithCtx,
          (statusTxt) => { pending.innerHTML = `<em>${escapeHtml(statusTxt)}</em>`; }
        );
        history.push({ role: "assistant", content: reply });
        pending.innerHTML = formatMarkdownLite(reply);
        if (HJR.typeset) HJR.typeset(pending);
      } catch (err) {
        pending.innerHTML = `<strong>Connection notice:</strong> ${escapeHtml(err.message || String(err))}`;
      } finally {
        sendBtn.disabled = false;
        msgBox.scrollTop = msgBox.scrollHeight;
      }
    }

    sendBtn.addEventListener("click", () => sendMessage());
    txtInp.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    $$(".tutor-chip", drawer).forEach((chip) => {
      chip.addEventListener("click", () => sendMessage(chip.dataset.q));
    });
  });
})();
