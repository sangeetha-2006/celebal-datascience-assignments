// Talks to the FastAPI backend. Same-origin by default (served via
// `uvicorn app.api.main:app`, mounted at /ui) — if you're serving this
// page a different way, change API_BASE below.
const API_BASE = window.location.origin;

const SESSION_ID = "web_" + Math.random().toString(36).slice(2, 8);
document.getElementById("session-id").textContent = SESSION_ID;

const consoleBody = document.getElementById("console-body");
const input = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const connStatus = document.getElementById("conn-status");

function addMessage(text, kind) {
  const div = document.createElement("div");
  div.className = "msg msg-" + kind;
  div.textContent = text;
  consoleBody.appendChild(div);
  consoleBody.scrollTop = consoleBody.scrollHeight;
  return div;
}

function setLight(name, on) {
  const light = document.getElementById("light-" + name);
  const row = document.getElementById("row-" + name);
  if (!light) return;
  light.classList.toggle("on", !!on);
  row.classList.toggle("active-label", !!on);
}

function resetLights() {
  ["memory", "rag", "kg", "tool"].forEach((n) => setLight(n, false));
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      connStatus.textContent = "● connected";
      connStatus.style.color = "var(--active)";
    } else {
      throw new Error("bad status");
    }
  } catch (e) {
    connStatus.textContent = "● backend unreachable";
    connStatus.style.color = "var(--danger)";
  }
}

async function sendMessage() {
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  sendBtn.disabled = true;
  input.disabled = true;

  addMessage(query, "user");
  const thinking = addMessage("thinking…", "system");
  resetLights();
  setLight("memory", true); // memory always loads

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: SESSION_ID, query }),
    });

    thinking.remove();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addMessage(err.detail || `Error ${res.status}`, "error");
      return;
    }

    const data = await res.json();
    addMessage(data.answer, "bot");
    setLight("rag", data.used_rag);
    setLight("kg", data.used_kg);
    setLight("tool", data.used_tool);
  } catch (e) {
    thinking.remove();
    addMessage(
      "Could not reach the backend. Is `uvicorn app.api.main:app --reload` running?",
      "error"
    );
    connStatus.textContent = "● backend unreachable";
    connStatus.style.color = "var(--danger)";
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

checkHealth();
input.focus();
