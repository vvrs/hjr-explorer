/* ==========================================================================
   tutor.js — Ephemeral Session-Only BYOK ("Bring Your Own Key") Study Tutor
   Works 100% client-side on static GitHub Pages with:
     - Google Gemini (generativelanguage.googleapis.com)
     - Anthropic Claude (api.anthropic.com w/ direct-browser-access header)
     - OpenAI ChatGPT (api.openai.com)
     - OpenRouter (openrouter.ai — unified multi-model CORS API)
   Keys live strictly in sessionStorage / RAM for the current tab only.
   ========================================================================== */
(function () {
  "use strict";

  const PROVIDERS = {
    gemini: {
      label: "Google Gemini",
      models: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
      placeholder: "AIzaSy... (Google AI Studio API key)",
    },
    anthropic: {
      label: "Anthropic Claude",
      models: ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest"],
      placeholder: "sk-ant-... (Anthropic API key)",
    },
    openai: {
      label: "OpenAI",
      models: ["gpt-4.1-mini", "gpt-4.1", "o4-mini"],
      placeholder: "sk-... (OpenAI API key)",
    },
    openrouter: {
      label: "OpenRouter (Any Model)",
      models: ["google/gemini-2.5-flash", "anthropic/claude-sonnet-4", "openai/gpt-4.1-mini"],
      placeholder: "sk-or-... (OpenRouter API key)",
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
    const keyInp = $("#tutor-key");
    const clearBtn = $("#tutor-clear-key");
    const msgBox = $("#tutor-messages");
    const txtInp = $("#tutor-input");
    const sendBtn = $("#tutor-send");

    const history = []; // { role: "user" | "assistant", content: string }

    const toggleDrawer = (show) => {
      drawer.classList.toggle("open", show);
      backdrop.classList.toggle("open", show);
      if (show) txtInp.focus();
    };
    openBtn.addEventListener("click", () => toggleDrawer(true));
    closeBtn.addEventListener("click", () => toggleDrawer(false));
    backdrop.addEventListener("click", () => toggleDrawer(false));

    function populateModels() {
      const p = PROVIDERS[provSel.value] || PROVIDERS.gemini;
      modSel.innerHTML = p.models.map((m) => `<option value="${m}">${m}</option>`).join("");
      keyInp.placeholder = p.placeholder;
      try {
        keyInp.value = sessionStorage.getItem("hjr_byok_" + provSel.value) || "";
      } catch (_) {}
    }
    provSel.addEventListener("change", populateModels);
    keyInp.addEventListener("input", () => {
      try { sessionStorage.setItem("hjr_byok_" + provSel.value, keyInp.value.trim()); } catch (_) {}
    });
    clearBtn.addEventListener("click", () => {
      keyInp.value = "";
      try {
        Object.keys(PROVIDERS).forEach((k) => sessionStorage.removeItem("hjr_byok_" + k));
      } catch (_) {}
      appendBubble("assistant", "Cleared session API keys from this browser tab.");
    });
    populateModels();

    function getActiveSectionContext() {
      const activeNav = $(".toc-nav a.active");
      if (!activeNav) return "Reader is at the top overview (Lab 01: Point of No Return).";
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

    async function callLLM(provider, model, apiKey, messages, sysPrompt) {
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

      // OpenAI or OpenRouter (OpenAI-compatible chat completions)
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
      const key = keyInp.value.trim();
      if (!key) {
        appendBubble("assistant", "Paste a session API key above first (Google AI Studio, Anthropic, OpenAI, or OpenRouter). Your key stays exclusively in this tab's memory and is cleared as soon as the tab closes.");
        keyInp.focus();
        return;
      }
      txtInp.value = "";
      appendBubble("user", q);
      history.push({ role: "user", content: q });
      const pending = appendBubble("assistant", "Thinking…");
      sendBtn.disabled = true;

      try {
        const sysWithCtx = `${SYSTEM_PROMPT}\n\n[Live Page Context]: ${getActiveSectionContext()}`;
        const reply = await callLLM(provSel.value, modSel.value, key, history.slice(-10), sysWithCtx);
        history.push({ role: "assistant", content: reply });
        pending.innerHTML = formatMarkdownLite(reply);
        if (HJR.typeset) HJR.typeset(pending);
      } catch (err) {
        pending.innerHTML = `<strong>Connection error:</strong> ${escapeHtml(err.message || String(err))}`;
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
